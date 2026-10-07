"""``preferred_pool_entry``: a session pin (gateway ``/credentials``) is a preselection, not a
select-then-swap. Inside the scope ``select()`` serves the pinned entry and charges the request to
it; the strategy's own pick is never counted and round-robin order never rotates. A benched pin
falls back to the normal strategy, which then charges the entry that actually serves.
"""

from __future__ import annotations

import base64
import json
import time

import pytest

from agent.credential_pool import load_pool
from agent.credential_pool_admin import preferred_pool_entry


def _jwt(sub: str) -> str:
    def part(payload: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part({'exp': int(time.time()) + 86400, 'sub': sub})}.sig"


def _entry(entry_id: str, priority: int, **extra) -> dict:
    return {"id": entry_id, "label": entry_id, "auth_type": "oauth", "priority": priority,
            "source": f"manual:{entry_id}", "access_token": _jwt(entry_id), "refresh_token": f"rt-{entry_id}",
            "base_url": "https://chatgpt.com/backend-api/codex", **extra}


@pytest.fixture
def pool_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def write(strategy: str, entries: list) -> None:
        (home / "config.yaml").write_text(
            f"credential_pool_strategies:\n  openai-codex: {strategy}\n", encoding="utf-8")
        (home / "auth.json").write_text(json.dumps(
            {"version": 1, "credential_pool": {"openai-codex": entries}}), encoding="utf-8")
    return write


def _counts(pool) -> dict:
    return {e.id: e.request_count for e in pool.entries()}


@pytest.mark.parametrize("strategy", ["least_used", "round_robin", "fill_first"])
def test_pinned_select_charges_only_the_pinned_entry_and_keeps_order(pool_home, strategy):
    pool_home(strategy, [_entry("aaa", 0), _entry("bbb", 1), _entry("ccc", 2)])
    pool = load_pool("openai-codex")
    order = [e.id for e in pool.entries()]

    with preferred_pool_entry("openai-codex", "ccc"):
        served = [pool.select().id for _ in range(3)]

    assert served == ["ccc", "ccc", "ccc"]
    assert _counts(pool) == {"aaa": 0, "bbb": 0, "ccc": 3}
    assert [e.id for e in pool.entries()] == order  # no round-robin rotation
    assert pool.current().id == "ccc"


def test_least_used_pick_after_a_pinned_turn_is_not_skewed(pool_home):
    pool_home("least_used", [_entry("aaa", 0), _entry("bbb", 1)])
    pool = load_pool("openai-codex")
    with preferred_pool_entry("openai-codex", "bbb"):
        pool.select()
    # bbb served (count 1); aaa was never charged, so it is the least used.
    assert pool.select().id == "aaa"
    assert _counts(pool) == {"aaa": 1, "bbb": 1}


def test_round_robin_rotation_continues_from_where_automatic_sessions_left_it(pool_home):
    pool_home("round_robin", [_entry("aaa", 0), _entry("bbb", 1), _entry("ccc", 2)])
    pool = load_pool("openai-codex")
    assert pool.select().id == "aaa"
    with preferred_pool_entry("openai-codex", "ccc"):
        assert pool.select().id == "ccc"
    assert pool.select().id == "bbb"  # the pinned turn did not consume a rotation slot


def test_benched_pin_falls_back_to_the_strategy_which_charges_the_server(pool_home):
    now = time.time()
    pool_home("least_used", [
        _entry("aaa", 0), _entry("bbb", 1),
        _entry("ccc", 2, last_status="exhausted", last_status_at=now, last_error_code=429,
               last_error_reset_at=now + 3600)])
    pool = load_pool("openai-codex")
    with preferred_pool_entry("openai-codex", "ccc"):
        served = pool.select()
    assert served.id in {"aaa", "bbb"}
    assert _counts(pool)[served.id] == 1 and _counts(pool)["ccc"] == 0


def test_preference_is_scoped_to_its_provider_and_block(pool_home):
    pool_home("fill_first", [_entry("aaa", 0), _entry("bbb", 1)])
    pool = load_pool("openai-codex")
    with preferred_pool_entry("anthropic", "bbb"):
        assert pool.select().id == "aaa"
    with preferred_pool_entry("openai-codex", None):
        assert pool.select().id == "aaa"
    with preferred_pool_entry("openai-codex", "bbb"):
        assert pool.select().id == "bbb"
    assert pool.select().id == "aaa"
