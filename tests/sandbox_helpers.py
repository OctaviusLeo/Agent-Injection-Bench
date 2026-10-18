"""Test helpers: a socket guard that records any network attempt, and a frozen clock for the tools."""

from __future__ import annotations

import contextlib
import datetime as _dt
import socket
import types

from agentdojo.default_suites.v1.tools import cloud_drive_client, email_client

FROZEN_NOW = _dt.datetime(2024, 5, 15, 12, 0, 0)


class _FrozenDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):  # noqa: D102
        return FROZEN_NOW if tz is None else FROZEN_NOW.replace(tzinfo=tz)


_FROZEN_MODULE = types.SimpleNamespace(datetime=_FrozenDatetime, date=_dt.date, timedelta=_dt.timedelta, time=_dt.time)


@contextlib.contextmanager
def frozen_tool_clock():
    """The email and drive tools stamp new objects with datetime.now(); freeze it for exact comparisons."""
    saved = (email_client.datetime, cloud_drive_client.datetime)
    email_client.datetime = _FROZEN_MODULE
    cloud_drive_client.datetime = _FROZEN_MODULE
    try:
        yield
    finally:
        email_client.datetime, cloud_drive_client.datetime = saved


@contextlib.contextmanager
def socket_guard():
    """Make every outbound connection or DNS lookup raise, and record the attempt (tools may swallow errors)."""
    trips: list[str] = []
    saved = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo)

    def trip(name):
        def _raise(*args, **kwargs):
            trips.append(name)
            raise ConnectionRefusedError(f"network access attempted inside the Sandbox: {name}")
        return _raise

    socket.socket.connect = trip("socket.connect")
    socket.socket.connect_ex = trip("socket.connect_ex")
    socket.create_connection = trip("socket.create_connection")
    socket.getaddrinfo = trip("socket.getaddrinfo")
    try:
        yield trips
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection, socket.getaddrinfo = saved
