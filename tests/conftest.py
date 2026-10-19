"""Offline tests cannot accidentally contact a configured real provider."""

import socket

import pytest


@pytest.fixture(autouse=True)
def deny_external_network_in_offline_tests(request, monkeypatch):
    if request.node.get_closest_marker("live"):
        return
    connect = socket.socket.connect

    def guarded_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError("Offline test attempted an external network connection")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
