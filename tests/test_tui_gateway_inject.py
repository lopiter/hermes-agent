"""External message injection reaches dashboard/desktop sessions.

``PluginContext.inject_message`` used to be CLI-only: in a headless serving
process (``hermes serve`` / the desktop app) ``_cli_ref`` is ``None``, so a
plugin's completion report was logged as a warning and dropped — dispatch
watchers finished work nobody ever heard about. The gateway now exposes
``tui_gateway.server.inject_external_message``, and ``inject_message`` routes
to it when no interactive CLI is attached: busy sessions get the message
merged into their queued next-turn prompt (never interrupting in-flight
work), idle sessions start a relay turn immediately.
"""

import threading
import time
import types

from tui_gateway import server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "transport": None,
        "attached_images": [],
        **extra,
    }


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


# ── inject_external_message ────────────────────────────────────────────────

def test_idle_session_fires_a_turn_immediately(monkeypatch):
    fired = {}

    def fake_run_prompt_submit(rid, sid, session, text, **kwargs):
        fired["sid"] = sid
        fired["text"] = text

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run_prompt_submit)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda s: False)
    session = _session(last_active=100.0)
    monkeypatch.setattr(server, "_sessions", {"sid-1": session})

    assert server.inject_external_message("run finished") is True
    assert _wait_until(lambda: fired.get("text") == "run finished")
    assert fired["sid"] == "sid-1"


def test_busy_session_queues_without_interrupting(monkeypatch):
    calls = {"interrupt": 0}
    agent = types.SimpleNamespace(
        interrupt=lambda *a, **k: calls.__setitem__("interrupt", calls["interrupt"] + 1)
    )
    session = _session(agent=agent, running=True, last_active=100.0)
    monkeypatch.setattr(server, "_sessions", {"sid-1": session})

    assert server.inject_external_message("run finished") is True
    assert session["queued_prompt"]["text"] == "run finished"
    assert calls["interrupt"] == 0


def test_busy_session_merges_behind_queued_user_prompt(monkeypatch):
    session = _session(running=True, last_active=100.0)
    server._enqueue_prompt(session, "user typed this", "ws-1")
    monkeypatch.setattr(server, "_sessions", {"sid-1": session})

    assert server.inject_external_message("run finished") is True
    assert session["queued_prompt"]["text"] == "user typed this\n\nrun finished"
    # The user's transport binding survives the merge.
    assert session["queued_prompt"]["transport"] == "ws-1"


def test_most_recently_active_session_wins(monkeypatch):
    fired = {}

    def fake_run_prompt_submit(rid, sid, session, text, **kwargs):
        fired["sid"] = sid

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run_prompt_submit)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda s: False)
    monkeypatch.setattr(server, "_sessions", {
        "old": _session(last_active=10.0),
        "new": _session(last_active=200.0),
    })

    assert server.inject_external_message("report") is True
    assert _wait_until(lambda: fired.get("sid") == "new")


def test_target_session_beats_more_recent_activity(monkeypatch):
    fired = {}

    def fake_run_prompt_submit(rid, sid, session, text, **kwargs):
        fired["sid"] = sid

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run_prompt_submit)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda s: False)
    monkeypatch.setattr(server, "_sessions", {
        "origin": _session(last_active=10.0),
        "chatty": _session(last_active=200.0),
    })

    assert server.inject_external_message("report", target_sid="origin") is True
    assert _wait_until(lambda: fired.get("sid") == "origin")


def test_closed_target_session_drops_instead_of_rerouting(monkeypatch):
    other = _session(last_active=200.0)
    monkeypatch.setattr(server, "_sessions", {"other": other})

    assert server.inject_external_message("report", target_sid="gone") is False
    assert other.get("queued_prompt") is None


def test_lazy_target_session_drops_instead_of_rerouting(monkeypatch):
    other = _session(last_active=200.0)
    monkeypatch.setattr(server, "_sessions", {
        "watch": _session(lazy=True, last_active=10.0),
        "other": other,
    })

    assert server.inject_external_message("report", target_sid="watch") is False
    assert other.get("queued_prompt") is None


def test_lazy_watch_sessions_are_skipped(monkeypatch):
    monkeypatch.setattr(server, "_sessions", {
        "watch": _session(lazy=True, last_active=200.0),
    })
    assert server.inject_external_message("report") is False


def test_no_sessions_returns_false(monkeypatch):
    monkeypatch.setattr(server, "_sessions", {})
    assert server.inject_external_message("report") is False


def test_blank_text_is_rejected(monkeypatch):
    monkeypatch.setattr(server, "_sessions", {"sid-1": _session(last_active=1.0)})
    assert server.inject_external_message("   ") is False
    assert server.inject_external_message(None) is False


# ── PluginContext.inject_message routing ───────────────────────────────────

def _plugin_ctx():
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    pm = PluginManager()
    manifest = PluginManifest(
        name="testplugin", version="1.0.0", description="test", source="user",
    )
    return PluginContext(manifest, pm)


def test_inject_message_routes_to_gateway_when_no_cli(monkeypatch):
    session = _session(running=True, last_active=100.0)
    monkeypatch.setattr(server, "_sessions", {"sid-1": session})

    ctx = _plugin_ctx()
    assert ctx._manager._cli_ref is None
    assert ctx.inject_message("run finished") is True
    assert session["queued_prompt"]["text"] == "run finished"


def test_inject_message_session_id_targets_origin_session(monkeypatch):
    origin = _session(running=True, last_active=10.0)
    chatty = _session(running=True, last_active=200.0)
    monkeypatch.setattr(server, "_sessions", {"origin": origin, "chatty": chatty})

    ctx = _plugin_ctx()
    assert ctx.inject_message("run finished", session_id="origin") is True
    assert origin["queued_prompt"]["text"] == "run finished"
    assert chatty.get("queued_prompt") is None


def test_inject_message_session_id_gone_drops(monkeypatch):
    chatty = _session(running=True, last_active=200.0)
    monkeypatch.setattr(server, "_sessions", {"chatty": chatty})

    ctx = _plugin_ctx()
    assert ctx.inject_message("run finished", session_id="gone") is False
    assert chatty.get("queued_prompt") is None


def test_inject_message_still_false_without_cli_or_sessions(monkeypatch):
    monkeypatch.setattr(server, "_sessions", {})
    ctx = _plugin_ctx()
    assert ctx.inject_message("run finished") is False


def test_inject_message_prefers_cli_when_attached(monkeypatch):
    session = _session(last_active=100.0)
    monkeypatch.setattr(server, "_sessions", {"sid-1": session})

    class _Queue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    cli = types.SimpleNamespace(
        _agent_running=False, _pending_input=_Queue(), _interrupt_queue=_Queue(),
    )
    ctx = _plugin_ctx()
    ctx._manager._cli_ref = cli

    assert ctx.inject_message("hello") is True
    assert cli._pending_input.items == ["hello"]
    assert session.get("queued_prompt") is None


# ── surviving a disconnect ─────────────────────────────────────────────────
#
# A remote desktop client's WebSocket dropping reaps the live session
# (_ws_orphan_reap) while the conversation survives in the session store and
# returns on reconnect under a NEW sid. A report finishing in that window used
# to be dropped: its target sid no longer existed. It is now routed by the
# durable session key, and parked until the conversation comes back.

def _park_isolated(monkeypatch):
    """Give a test its own park store."""
    monkeypatch.setattr(server, "_parked_injections", {})
    monkeypatch.setattr(server, "_parked_injections_lock", threading.Lock())


def test_durable_key_delivers_when_the_sid_is_already_gone(monkeypatch):
    _park_isolated(monkeypatch)
    # Reconnected before the report landed: new sid, same durable key.
    reborn = _session(running=True, session_key="conv-1", last_active=100.0)
    monkeypatch.setattr(server, "_sessions", {"sid-new": reborn})

    assert server.inject_external_message(
        "run finished", target_sid="sid-old-and-reaped", target_key="conv-1"
    ) is True
    assert reborn["queued_prompt"]["text"] == "run finished"
    assert server._parked_injections == {}


def test_report_parks_when_nothing_is_live(monkeypatch):
    _park_isolated(monkeypatch)
    monkeypatch.setattr(server, "_sessions", {})

    assert server.inject_external_message(
        "run finished", target_sid="sid-old", target_key="conv-1"
    ) is True
    assert [text for _ts, text in server._parked_injections["conv-1"]] == [
        "run finished"
    ]


def test_parked_report_is_delivered_when_the_session_returns(monkeypatch):
    """The case this exists for: the report lands BEFORE the client reconnects."""
    _park_isolated(monkeypatch)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_register_session_cwd", lambda s: None)
    fired = {}

    def fake_run_prompt_submit(rid, sid, session, text, **kwargs):
        fired["sid"] = sid
        fired["text"] = text

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run_prompt_submit)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda s: False)

    assert server.inject_external_message(
        "run finished", target_sid="sid-old", target_key="conv-1"
    ) is True

    # ... now the desktop reconnects and resumes the same conversation.
    reborn = _session(session_key="conv-1", last_active=200.0)
    assert server._claim_or_reuse_live("sid-new", "conv-1", reborn, None) is None

    assert _wait_until(lambda: fired.get("text") == "run finished")
    assert fired["sid"] == "sid-new"
    assert server._parked_injections == {}


def test_parked_reports_drain_in_arrival_order(monkeypatch):
    _park_isolated(monkeypatch)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_register_session_cwd", lambda s: None)

    for text in ("first report", "second report"):
        server.inject_external_message(text, target_key="conv-1")

    # Busy on return, so both merge into the queued prompt deterministically.
    reborn = _session(running=True, session_key="conv-1")
    server._claim_or_reuse_live("sid-new", "conv-1", reborn, None)

    assert reborn["queued_prompt"]["text"] == "first report\n\nsecond report"


def test_parking_never_reroutes_to_a_sibling_session(monkeypatch):
    """The no-rerouting rule still holds: an unreachable target parks, it does
    not fall back to whichever other conversation happens to be open."""
    _park_isolated(monkeypatch)
    sibling = _session(running=True, session_key="conv-other", last_active=999.0)
    monkeypatch.setattr(server, "_sessions", {"sid-sibling": sibling})

    assert server.inject_external_message(
        "session A's report", target_sid="sid-a", target_key="conv-a"
    ) is True
    assert sibling.get("queued_prompt") is None
    assert "conv-a" in server._parked_injections


def test_parked_reports_expire(monkeypatch):
    _park_isolated(monkeypatch)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_register_session_cwd", lambda s: None)

    server.inject_external_message("stale report", target_key="conv-1")
    # Backdate past the TTL, as if the conversation never came back.
    server._parked_injections["conv-1"] = [
        (time.time() - server._PARKED_INJECTION_TTL_S - 1.0, "stale report")
    ]

    reborn = _session(running=True, session_key="conv-1")
    server._claim_or_reuse_live("sid-new", "conv-1", reborn, None)

    assert reborn.get("queued_prompt") is None
    assert server._parked_injections == {}


def test_park_depth_is_bounded_per_key(monkeypatch):
    _park_isolated(monkeypatch)
    monkeypatch.setattr(server, "_sessions", {})

    for i in range(server._PARKED_INJECTIONS_PER_KEY + 5):
        server.inject_external_message(f"report {i}", target_key="conv-1")

    parked = [text for _ts, text in server._parked_injections["conv-1"]]
    assert len(parked) == server._PARKED_INJECTIONS_PER_KEY
    # Oldest evicted, newest kept.
    assert parked[-1] == f"report {server._PARKED_INJECTIONS_PER_KEY + 4}"
    assert "report 0" not in parked


def test_untargeted_messages_are_never_parked(monkeypatch):
    """No durable key means no destination to hold it for — keep the old
    contract so the caller runs its degraded path instead of silently
    accumulating messages nobody will ever claim."""
    _park_isolated(monkeypatch)
    monkeypatch.setattr(server, "_sessions", {})

    assert server.inject_external_message("to whoever is listening") is False
    assert server._parked_injections == {}


def test_inject_message_forwards_the_durable_key(monkeypatch):
    _park_isolated(monkeypatch)
    reborn = _session(running=True, session_key="conv-1", last_active=100.0)
    monkeypatch.setattr(server, "_sessions", {"sid-new": reborn})

    ctx = _plugin_ctx()
    assert ctx._manager._cli_ref is None
    assert ctx.inject_message(
        "run finished", session_id="sid-reaped", session_key="conv-1"
    ) is True
    assert reborn["queued_prompt"]["text"] == "run finished"
