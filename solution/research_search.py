"""Bounded research over replayable recipes, with fresh OOF diagnostics.

Only training-fold labels reach model fitting. The outer validation labels are
used to score completed predictions, never for early stopping. The portfolio
keeps a baseline, the champion and a few alternatives, including near losers.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import eda_detectors
import fe_dsl
import fe_search
import models
from agent_utils import StageTimeout, log, time_limit


def recipe_key(specs):
    return tuple(sorted(fe_dsl.spec_key(s) for s in fe_dsl.unique_specs(specs)))


def recipe_id(specs):
    return hashlib.sha256(json.dumps(["lightgbm", recipe_key(specs)]).encode()).hexdigest()[:16]


def extend(parent, spec):
    """Model operations replace their previous version; ordinary features accumulate."""
    kept = [s for s in parent if not (spec["op"] in fe_dsl.MODEL_OPS and s["op"] == spec["op"])]
    return sorted(fe_dsl.unique_specs(kept + [spec]), key=fe_dsl.spec_key)


class RecipeEvaluator(fe_search.Evaluator):
    """Fixed iteration count: outer fold labels cannot affect fitting or stopping."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.cache = OrderedDict()
        self.evaluations = OrderedDict()
        self.last_seconds = 0.0
        self.screen_result = None

    def _spec_feats(self, spec):
        result = super()._spec_feats(spec)
        self.cache.move_to_end(fe_dsl.spec_key(spec))
        while len(self.cache) > 16:
            self.cache.popitem(last=False)
        return result

    def _predict(self, Xtr, ytr, Xva, deadline, objective=None):
        import lightgbm as lgb

        ctx = self.ctx
        params = fe_search._lgb_params(ctx)
        if objective:
            params["objective"] = objective["name"]
            if "alpha" in objective:
                params["alpha" if objective["name"] == "huber" else "fair_c"] = objective["alpha"]
        classes = ctx["classes"]
        if classes and len(np.unique(ytr)) == 1:
            pred = np.zeros((len(Xva), len(classes)))
            pred[:, classes.index(ytr[0])] = 1
            return pred
        log_target = not classes and ctx["prof"].get("log_target", False)
        fit_y = np.log1p(np.clip(ytr, 0, None)) if log_target else ytr
        model = (lgb.LGBMClassifier if classes else lgb.LGBMRegressor)(**params)
        model.fit(Xtr, fit_y, callbacks=[models.time_guard(deadline)])
        if time.time() >= deadline:
            raise StageTimeout()
        if classes:
            return fe_search._align(ctx, model, model.predict_proba(Xva))
        pred = model.predict(Xva)
        return np.clip(np.expm1(pred), 0, None) if log_target else pred

    def evaluate(self, specs, deadline, screen=False, extra=None):
        """Screen on one fixed fold; promotion always needs all folds, at the same fidelity."""
        key = (recipe_key(specs), screen)
        if extra is None and key in self.evaluations:
            self.evaluations.move_to_end(key)
            return self.evaluations[key]
        if time.time() >= deadline:
            raise StageTimeout()
        t0 = time.time()
        ctx, df = self.ctx, self.ctx["df"]
        folds = ctx["folds"][:1] if screen else ctx["folds"]
        scores, predictions, rows = [], [], []
        seg = next((s for s in specs if s["op"] == "segment_model"), None)
        obj = next((s["params"] for s in specs if s["op"] == "objective"), None)
        for i, (tr, va) in enumerate(folds):
            if time.time() >= deadline:
                raise StageTimeout()
            if extra is None and not screen and i == 0 and self.screen_result is not None and self.screen_result[0] == key[0]:
                first = self.screen_result[1]
                scores.extend(first["scores"])
                predictions.append(first["oof"])
                rows.append(first["rows"])
                continue
            mats = self.matrices(i, specs, extra)
            if mats is None or not mats[0].shape[1]:
                raise ValueError("recipe cannot be materialized")
            Xtr, Xva = mats
            ytr = np.asarray(ctx["y"])[tr]
            pred = self._predict(Xtr, ytr, Xva, deadline, obj)
            if seg:
                ctr, cva = fe_dsl.segment_codes(seg, df.loc[tr], df.loc[va])
                for code in np.unique(cva):
                    fit, val = ctr == code, cva == code
                    if fit.sum() >= fe_search.SEG_MIN_ROWS:
                        pred[val] = self._predict(Xtr.loc[fit], ytr[fit], Xva.loc[val], deadline, obj)
            if not np.isfinite(pred).all():
                raise ValueError("non-finite predictions")
            scores.append(fe_search.score_pred(ctx, np.asarray(ctx["y"])[va], pred))
            predictions.append(pred)
            rows.append(va)
        if not scores or not np.isfinite(scores).all():
            raise ValueError("non-finite validation metric")
        self.n_evals += 1
        self.last_seconds = time.time() - t0
        self.spent += self.last_seconds
        result = dict(scores=scores, rows=np.concatenate(rows), oof=np.concatenate(predictions),
                      seconds=self.last_seconds)
        # Bound prediction memory as well as feature memory; seen recipes in Research
        # prevent old trials from being retrained after cache eviction.
        if extra is None:
            self.evaluations[key] = result
            while len(self.evaluations) > 4:
                self.evaluations.popitem(last=False)
        if screen and extra is None:
            self.screen_result = (key[0], result)
        return result


def diagnostic_context(ctx, candidate):
    """Only rows with this recipe's OOF are available to residual detectors."""
    rows = candidate["rows"]
    sub = {**ctx, "df": ctx["df"].iloc[rows].reset_index(drop=True),
           "y": np.asarray(ctx["y"])[rows], "oof": candidate["oof"].copy(),
           "facts": {k: deepcopy(v) for k, v in ctx["facts"].items()
                     if k not in ("tail", "segments", "perm_importance")}}
    if ctx["classes"]:
        lookup = {v: i for i, v in enumerate(ctx["classes"])}
        sub["y_idx"] = np.array([lookup[v] for v in sub["y"]])
        sub["resid"] = 1 - sub["oof"][np.arange(len(rows)), sub["y_idx"]]
    else:
        sub["resid"] = sub["y"] - sub["oof"]
    sub["oof_score"] = fe_search.score_pred(sub, sub["y"], sub["oof"])
    sub["facts"]["current_recipe"] = dict(id=candidate["id"], score=sub["oof_score"],
                                           specs=candidate["specs"], covered_rows=len(rows))
    return sub


def propose_from_errors(ctx, deadline):
    """Revisit error-dependent detectors without changing the fixed validation scheme."""
    specs = []
    for fn in (eda_detectors.detect_residuals, eda_detectors.detect_tail,
               eda_detectors.detect_segments, eda_detectors.detect_missingness,
               eda_detectors.detect_categoricals, eda_detectors.detect_transforms):
        if time.time() >= deadline:
            break
        try:
            for raw in fn(ctx):
                spec = fe_dsl.validate(raw, ctx["features"], set(eda_detectors._numeric(ctx)))
                if spec is not None:
                    specs.append(spec)
        except Exception as exc:
            log(f"Research detector {fn.__name__}: {type(exc).__name__}")
    return fe_dsl.unique_specs(specs)


class Research:
    def __init__(self, ctx, register=None, evaluator=None, proposer=propose_from_errors, llm=None,
                 width=4, max_rounds=24, batch_size=6, max_trials=144, max_specs=8):
        self.ctx, self.register = ctx, register
        self.ev = evaluator or RecipeEvaluator(ctx)
        self.proposer, self.llm = proposer, llm
        self.width, self.max_rounds, self.batch_size = max(3, width), max_rounds, batch_size
        self.max_trials, self.max_specs = max_trials, max_specs
        self.portfolio, self.seen, self.visits, self.queues = {}, set(), {}, {}
        self.events, self.history = [], []
        self.t0, self.stop_reason = time.time(), "not_started"
        self.baseline = self.champion = None
        self.direction = 1 if models.HIGHER_BETTER[ctx["metric"]] else -1
        self.llm_calls = 0
        self.round_count = 0
        self.rel_min = fe_search.REL_MIN

    def emit(self, event, **fields):
        row = dict(event=event, seconds=round(time.time() - self.t0, 3), **fields)
        self.events.append(row)
        if self.register and event in ("trial", "stop", "round"):
            self.register("research_" + event, json.dumps(fields, ensure_ascii=False, default=str),
                          fields.get("score"), fields.get("accepted", False))
        self.save()

    def save(self):
        workdir = self.ctx["prof"].get("workdir")
        if workdir is None:
            return
        directory = Path(workdir) / "research"
        directory.mkdir(exist_ok=True)
        name = hashlib.sha256(self.ctx["target"].encode()).hexdigest()[:12]
        path = directory / (name + ".json")
        data = dict(version=1, family="lightgbm", metric=self.ctx["metric"], cv=self.ctx["cv"],
                    champion=self.champion, stop_reason=self.stop_reason, history=self.history,
                    portfolio=[{k: c[k] for k in ("id", "parent", "specs", "scores", "score", "seconds")}
                               for c in self.portfolio.values()], events=self.events)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str, allow_nan=False))
        os.replace(tmp, path)

    def candidate(self, specs, result, parent=None):
        return dict(id=recipe_id(specs), parent=parent, specs=deepcopy(specs),
                    score=float(np.mean(result["scores"])), **result)

    def retain(self, candidate):
        """Keep champion and baseline; reserve one slot for a distinct error pattern."""
        self.portfolio[candidate["id"]] = candidate
        if len(self.portfolio) <= self.width:
            return
        fixed = {self.baseline, self.champion}
        rest = sorted((c for k, c in self.portfolio.items() if k not in fixed),
                      key=lambda c: -self.direction * c["score"])
        keep = set(fixed)
        if rest:
            keep.add(rest[0]["id"])
        if len(keep) < self.width and len(rest) > 1:
            champion = self.portfolio[self.champion]
            # Diversity only preserves an exploration branch; it never wins a submission.
            diverse = max(rest[1:], key=lambda c: float(np.mean((c["oof"] - champion["oof"]) ** 2)))
            keep.add(diverse["id"])
        for c in rest:
            if len(keep) >= self.width:
                break
            keep.add(c["id"])
        self.portfolio = {k: c for k, c in self.portfolio.items() if k in keep}

    def feedback(self, parent):
        return dict(metric=self.ctx["metric"], scheme=self.ctx["fold_scheme"],
                    threshold_rel=self.rel_min, current_recipe=parent["specs"],
                    current_score=parent["score"], champion=self.champion,
                    alternatives=[dict(id=c["id"], score=c["score"], specs=c["specs"])
                                  for c in self.portfolio.values()],
                    recent_trials=[e for e in self.events if e["event"] == "trial"][-12:])

    def run(self, initial_specs, deadline):
        try:
            with time_limit(max(.001, deadline - time.time())):
                self._run(initial_specs, deadline)
        except StageTimeout:
            self.stop_reason = "budget_exhausted"
        except Exception as exc:
            self.stop_reason = "search_error:" + type(exc).__name__
            log(f"Research: {type(exc).__name__}: {exc}")
        self.emit("stop", reason=self.stop_reason, trials=len(self.seen), rounds=self.round_count,
                  champion=self.champion)
        if self.champion is None:
            return []
        winner = self.portfolio[self.champion]
        self.ctx["search_base"] = self.portfolio[self.baseline]["score"]
        self.ctx["search_final"] = winner["score"]
        return deepcopy(winner["specs"])

    def _run(self, initial_specs, deadline):
        root = self.candidate([], self.ev.evaluate([], deadline))
        self.baseline = self.champion = root["id"]
        self.portfolio[root["id"]] = root
        self.seen.add(recipe_key([]))
        self.history.append(dict(seconds=time.time() - self.t0, recipe=root["id"], specs=[], score=root["score"]))
        noise = []
        if hasattr(self.ev, "null_feats"):
            top = sorted(self.ctx["features"], key=lambda c: -self.ctx["importances"].get(c, 0))
            for i, col in enumerate(top[:fe_search.N_NULL]):
                if not self.ev.fits(deadline):
                    break
                null = self.ev.evaluate([], deadline, extra=self.ev.null_feats(col, 100 + i))
                noise.append(abs(fe_search._gain(self.ctx["metric"], null["scores"], root["scores"])[0]))
        self.rel_min = max(fe_search.REL_MIN, max(noise, default=0))
        self.emit("calibration", threshold_rel=self.rel_min, null_gains=noise)
        self.stop_reason = "round_limit"
        stale = 0
        refresh_champion = False
        for round_no in range(self.max_rounds):
            if len(self.seen) >= self.max_trials:
                self.stop_reason = "trial_limit"
                break
            if not self.ev.fits(deadline):
                self.stop_reason = "insufficient_time_for_evaluation"
                break
            # Alternate exploitation with least-visited alternatives, including near losers.
            parent = self.portfolio[self.champion] if round_no % 2 == 0 or refresh_champion else min(
                self.portfolio.values(), key=lambda c: (self.visits.get(c["id"], 0), -self.direction * c["score"]))
            refresh_champion = False
            self.round_count += 1
            self.visits[parent["id"]] = self.visits.get(parent["id"], 0) + 1
            diag = diagnostic_context(self.ctx, parent)
            self.emit("round", round=round_no, parent=parent["id"], score=parent["score"],
                      covered_rows=len(diag["df"]), residual_mean=float(np.mean(diag["resid"])))
            proposed = self.proposer(diag, min(deadline, time.time() + 20))
            proposals = list(initial_specs) + proposed + self.queues.get(parent["id"], [])
            if self.llm and self.llm_calls < 8 and deadline - time.time() > max(5, self.ev.last_seconds * 2):
                self.llm_calls += 1
                try:
                    proposals += self.llm(diag, self.feedback(parent),
                                          [fe_dsl.describe(s) for s in proposals],
                                          min(190, max(1, (deadline - time.time()) * .3))) or []
                except Exception as exc:
                    self.emit("llm_error", error=type(exc).__name__)
            queue = fe_search.interleave(fe_dsl.unique_specs(proposals))
            tried, improved = 0, False
            self.queues[parent["id"]] = []
            for i, spec in enumerate(queue):
                specs = extend(parent["specs"], spec)
                key = recipe_key(specs)
                if key in self.seen or len(specs) > self.max_specs:
                    continue
                if tried >= self.batch_size or len(self.seen) >= self.max_trials or not self.ev.fits(deadline):
                    self.queues[parent["id"]] = queue[i:]
                    break
                self.seen.add(key)
                tried += 1
                try:
                    screen = self.ev.evaluate(specs, deadline, screen=True)
                    screen_gain, _ = fe_search._gain(self.ctx["metric"], screen["scores"], parent["scores"][:1])
                    # One in four trials bypasses screening to explore false negatives.
                    if screen_gain < -.05 and len(self.seen) % 4:
                        self.emit("trial", recipe=recipe_id(specs), parent=parent["id"], specs=specs,
                                  score=float(np.mean(screen["scores"])), accepted=False, reason="screen_only",
                                  fidelity="one_fold")
                        continue
                    result = self.ev.evaluate(specs, deadline)
                    candidate = self.candidate(specs, result, parent["id"])
                    champion = self.portfolio[self.champion]
                    gain, lcb = fe_search._gain(self.ctx["metric"], result["scores"], champion["scores"])
                    leaky = fe_search._looks_leaky(self.ctx["metric"], result["scores"], root["scores"])
                    accepted = gain > self.rel_min and lcb > 0 and not leaky
                    if accepted:
                        self.champion = candidate["id"]
                        self.history.append(dict(seconds=time.time() - self.t0, recipe=candidate["id"],
                                                 specs=deepcopy(specs), score=candidate["score"]))
                        improved = True
                        refresh_champion = True
                    if not leaky:
                        self.retain(candidate)
                    self.emit("trial", recipe=candidate["id"], parent=parent["id"], specs=specs,
                              score=candidate["score"], accepted=accepted, gain=gain, lcb=lcb,
                              reason="suspected_leakage" if leaky else "champion" if accepted else "alternative",
                              fidelity="all_folds", retained=candidate["id"] in self.portfolio)
                    if accepted:
                        self.queues[parent["id"]] = queue[i + 1:]
                        break  # Immediately diagnose the changed winner before its next experiment.
                except Exception as exc:
                    self.emit("trial", recipe=recipe_id(specs), parent=parent["id"], accepted=False,
                              reason="error:" + type(exc).__name__)
            stale = 0 if tried or improved else stale + 1
            if stale >= max(3, len(self.portfolio) * 2):
                self.stop_reason = "no_new_experiments"
                break
