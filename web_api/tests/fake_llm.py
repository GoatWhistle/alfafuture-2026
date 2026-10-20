"""Fake OpenAI-compatible endpoint for web e2e: every request gets one propose_experiment call.

Usage: python fake_llm.py PORT LOGFILE. Env SLEEP adds a per-request delay in seconds,
TASK_KIND sets the proposed task kind (binary, multiclass, regression).
The parent is taken from the last parent/best id mentioned in the request context.
"""

import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_PARENT = re.compile(r'"?(?:parent_id|active_parent_id|best_id)"?\W+(baseline[\w-]*|[\w-]+-\d+)')


class Handler(BaseHTTPRequestHandler):
    count = 0
    log_path = ""

    def log_message(self, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open(self.log_path, "a", encoding="utf-8") as log:
            tools = [t["function"]["name"] for t in body.get("tools") or []]
            log.write(json.dumps({"path": self.path, "tools": tools}) + "\n")
        time.sleep(float(os.environ.get("SLEEP", "0")))
        Handler.count += 1
        n = Handler.count
        found = _PARENT.findall(json.dumps(body["messages"], ensure_ascii=False))
        arguments = {
            "parent_id": found[-1] if found else "baseline",
            "task_kind": os.environ.get("TASK_KIND", "binary"),
            "backend": "catboost",
            "target_error": "Fake: ошибки на редких категориях f2",
            "rationale": "Fake: уменьшить глубину и число итераций",
            "expected_effect": "Fake: небольшой прирост AUC",
            "parameters": {"depth": 3 + n % 4, "iterations": 40 + 10 * n},
            "transforms": [],
            "python_transform": "",
            "tune": False,
        }
        call = {
            "id": f"call{n}",
            "type": "function",
            "function": {"name": "propose_experiment", "arguments": json.dumps(arguments)},
        }
        response = {
            "id": f"fake{n}",
            "object": "chat.completion",
            "created": 0,
            "model": "fake",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                }
            ],
            "usage": {"prompt_tokens": 400, "completion_tokens": 80, "total_tokens": 480},
        }
        data = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    Handler.log_path = sys.argv[2]
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
