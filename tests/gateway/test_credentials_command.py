"""Gateway ``/credentials``: per-session OpenAI-Codex pool-entry pin with inline buttons.

Runs against a real ``load_pool`` over a temp ``HERMES_HOME`` auth store, so pin resolution goes
through the production pool (locks, cooldown checks, ``current`` bookkeeping), not a fake.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.slash_commands_credentials import (
    apply_session_credential_pin,
    safe_credential_label,
    session_credential_pin,
    session_credential_usage_view,
)


def _jwt(claims: dict) -> str:
    def _part(payload: dict) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}.sig"


TOKEN_A = _jwt({"exp": int(time.time()) + 86400, "sub": "a", "nonce": "secret-a-material"})
TOKEN_B = _jwt({"exp": int(time.time()) + 86400, "sub": "b", "nonce": "secret-b-material"})
REFRESH_A, REFRESH_B = "rt-secret-refresh-a-0123456789", "rt-secret-refresh-b-0123456789"
SECRETS = (TOKEN_A, TOKEN_B, REFRESH_A, REFRESH_B)


def _entry(entry_id, label, token, refresh, priority, **extra):
    return {"id": entry_id, "label": label, "auth_type": "oauth", "priority": priority,
            "source": "manual:device_code", "access_token": token, "refresh_token": refresh,
            "base_url": "https://chatgpt.com/backend-api/codex", **extra}


@pytest.fixture
def codex_pool(tmp_path, monkeypatch):
    """Two-entry openai-codex pool (fill_first → ``aaa111`` is the automatic pick)."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(gateway_run, "_hermes_home", home)

    def write(entries):
        (home / "auth.json").write_text(json.dumps(
            {"version": 1, "credential_pool": {"openai-codex": entries}}, indent=2))

    write([_entry("aaa111", "work@example.com", TOKEN_A, REFRESH_A, 0),
           _entry("bbb222", "home@example.com", TOKEN_B, REFRESH_B, 1)])
    return write


class _PickerAdapter:
    def __init__(self):
        self.calls = []

    async def send_choice_picker(self, **kwargs):
        self.calls.append(kwargs)
        return SendResult(success=True, message_id="m1")


class _NoPickerAdapter:
    """No ``send_choice_picker`` on the type: plain-text platform."""


def _event(text="/credentials", chat_id="67890", user_id="12345"):
    return MessageEvent(text=text, source=SessionSource(
        platform=Platform.TELEGRAM, user_id=user_id, chat_id=chat_id, user_name="u"))


def _runner(adapter):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._running_agents = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._spawn_release_thread = lambda *a, **k: None
    runner._delivery_adapter_for = lambda source: adapter
    runner._thread_metadata_for_source = lambda source, anchor=None: {}
    runner._reply_anchor_for_event = lambda event: None
    return runner


def _codex_runtime(pool=None, api_key=TOKEN_A):
    return {"provider": "openai-codex", "api_key": api_key, "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses", "credential_pool": pool}


def _assert_no_secrets(*texts):
    blob = json.dumps([str(t) for t in texts])
    for secret in SECRETS:
        assert secret not in blob


# ---------------------------------------------------------------- registry


def test_credentials_is_a_registered_gateway_command_in_the_telegram_menu():
    from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, gateway_help_lines, resolve_command
    from hermes_cli.commands_platforms import telegram_bot_commands, telegram_menu_commands

    cmd = resolve_command("credentials")
    assert cmd is not None and cmd.gateway_only
    assert "credentials" in GATEWAY_KNOWN_COMMANDS
    assert any("/credentials" in line for line in gateway_help_lines())
    assert "credentials" in dict(telegram_bot_commands(include_plugins=False))
    menu, _hidden = telegram_menu_commands(max_commands=60)
    assert "credentials" in dict(menu)
    assert [name for name, _description in menu].index("credentials") < 10
    runner = _runner(None)
    assert runner._gateway_idle_command_handlers()["credentials"] == runner._handle_credentials_command


# ---------------------------------------------------------------- buttons / text


@pytest.mark.asyncio
async def test_bare_command_sends_automatic_plus_safe_entry_buttons(codex_pool):
    adapter = _PickerAdapter()
    runner = _runner(adapter)

    assert await runner._handle_credentials_command(_event()) is None

    choices = adapter.calls[0]["choices"]
    assert [c["value"] for c in choices] == ["auto", "aaa111", "bbb222"]
    assert choices[0]["is_current"] is True
    assert choices[1]["label"] == "work@example.com · aaa111"
    _assert_no_secrets(adapter.calls[0]["title"], choices)


@pytest.mark.asyncio
async def test_platform_without_buttons_gets_text_list_and_typed_selection(codex_pool):
    runner = _runner(_NoPickerAdapter())

    text = await runner._handle_credentials_command(_event())
    assert "work@example.com · aaa111" in text and "/credentials <id|number|auto>" in text

    reply = await runner._handle_credentials_command(_event("/credentials 2"))
    key = runner._session_key_for_source(_event().source)
    assert "home@example.com · bbb222" in reply
    assert runner._peek_session_state(key).conversation.credential_pin == "bbb222"
    _assert_no_secrets(text, reply)


@pytest.mark.asyncio
async def test_unknown_selection_never_echoes_the_argument(codex_pool):
    runner = _runner(_NoPickerAdapter())
    reply = await runner._handle_credentials_command(_event(f"/credentials {TOKEN_A}"))
    _assert_no_secrets(reply)
    assert runner._peek_session_state(runner._session_key_for_source(_event().source)) is None


@pytest.mark.asyncio
async def test_telegram_buttons_carry_opaque_picker_id_and_tap_pins_entry(codex_pool, monkeypatch):
    from gateway.config import PlatformConfig
    import plugins.platforms.telegram.adapter as tg
    from plugins.platforms.telegram.adapter import TelegramAdapter

    # Record the keyboard the adapter builds whether or not python-telegram-bot is installed.
    monkeypatch.setattr(tg, "InlineKeyboardButton", lambda text, callback_data: SimpleNamespace(
        text=text, callback_data=callback_data))
    monkeypatch.setattr(tg, "InlineKeyboardMarkup", lambda rows: SimpleNamespace(inline_keyboard=rows))

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = MagicMock()
    adapter._send_control_message = AsyncMock(return_value=SimpleNamespace(message_id=7))
    runner = _runner(adapter)
    event = _event()

    assert await runner._handle_credentials_command(event) is None

    markup = adapter._send_control_message.await_args.kwargs["reply_markup"]
    data = [b.callback_data for row in markup.inline_keyboard for b in row]
    picker_ids = {d.split(":")[1] for d in data}
    assert len(picker_ids) == 1 and [d.rsplit(":", 1)[1] for d in data] == ["0", "1", "2"]
    assert all(len(d.encode()) <= 64 for d in data)  # Telegram callback_data cap
    _assert_no_secrets(data, [b.text for row in markup.inline_keyboard for b in row],
                       adapter._send_control_message.await_args)

    query = SimpleNamespace(answer=AsyncMock(), edit_message_text=AsyncMock(),
                            message=SimpleNamespace(message_id=7))
    await adapter._handle_choice_picker_callback(query, data[2], event.source.chat_id)

    key = runner._session_key_for_source(event.source)
    assert runner._peek_session_state(key).conversation.credential_pin == "bbb222"
    edited = query.edit_message_text.await_args_list[-1].kwargs["text"]
    assert "bbb222" in edited
    _assert_no_secrets(edited)


@pytest.mark.asyncio
async def test_picker_pool_reload_failure_with_secret_never_reaches_chat_or_logs(codex_pool, monkeypatch, caplog):
    import gateway.slash_commands_credentials as creds

    adapter = _PickerAdapter()
    runner = _runner(adapter)
    await runner._handle_credentials_command(_event())

    def _boom():
        raise RuntimeError(f"refresh failed: refresh_token={REFRESH_A} access={TOKEN_A}")

    monkeypatch.setattr(creds, "load_codex_choices", _boom)
    with caplog.at_level("DEBUG"):
        reply = await adapter.calls[0]["on_choice_selected"]("67890", "bbb222")

    assert "Couldn't read the OpenAI Codex credential pool" in reply
    _assert_no_secrets(reply, caplog.text)


# ---------------------------------------------------------------- pin / unpin / isolation


@pytest.mark.asyncio
async def test_pick_and_automatic_evict_cached_agent_and_are_session_scoped(codex_pool):
    adapter = _PickerAdapter()
    runner = _runner(adapter)
    mine, other = _event(chat_id="1"), _event(chat_id="2")
    key, other_key = (runner._session_key_for_source(e.source) for e in (mine, other))
    runner._agent_cache = {key: (MagicMock(), "sig"), other_key: (MagicMock(), "sig")}

    await runner._handle_credentials_command(mine)
    on_choice = adapter.calls[0]["on_choice_selected"]
    await on_choice(mine.source.chat_id, "bbb222")

    assert key not in runner._agent_cache  # rebuilt on the next turn
    assert other_key in runner._agent_cache
    assert runner._peek_session_state(key).conversation.credential_pin == "bbb222"
    assert runner._peek_session_state(other_key) is None

    runner._agent_cache[key] = (MagicMock(), "sig")
    reply = await on_choice(mine.source.chat_id, "auto")
    assert "Automatic" in reply
    assert key not in runner._agent_cache
    assert runner._peek_session_state(key).conversation.credential_pin is None


def test_turn_runtime_uses_pinned_entry_and_keeps_pool(codex_pool):
    from agent.credential_pool import load_pool

    runner = _runner(None)
    runner._session_state("s1").conversation.credential_pin = "bbb222"
    pool = load_pool("openai-codex")
    assert pool.select().id == "aaa111"  # what automatic resolution picked

    runtime = apply_session_credential_pin(runner, "s1", "gpt-5.5", _codex_runtime(pool))

    assert runtime["api_key"] == TOKEN_B
    assert runtime["credential_pool"] is pool
    assert pool.current().id == "bbb222"
    assert pool.entry_id_for_api_key(TOKEN_B) == "bbb222"  # 429 recovery attributes the pin
    # Another session, and a non-codex route, are untouched.
    assert apply_session_credential_pin(runner, "s2", "gpt-5.5", _codex_runtime(pool))["api_key"] == TOKEN_A
    other = {"provider": "openrouter", "api_key": "or-key"}
    assert apply_session_credential_pin(runner, "s1", "x", other) is other


def test_pin_attaches_codex_pool_when_runtime_had_none(codex_pool):
    runner = _runner(None)
    runner._session_state("s1").conversation.credential_pin = "bbb222"
    runtime = apply_session_credential_pin(runner, "s1", None, _codex_runtime(pool=None))
    assert runtime["api_key"] == TOKEN_B
    assert runtime["credential_pool"].provider == "openai-codex"


def test_deleted_pinned_entry_falls_back_to_automatic_and_clears_pin(codex_pool):
    runner = _runner(None)
    runner._session_state("s1").conversation.credential_pin = "bbb222"
    codex_pool([_entry("aaa111", "work@example.com", TOKEN_A, REFRESH_A, 0)])
    base = _codex_runtime(pool=None)

    assert apply_session_credential_pin(runner, "s1", None, base) is base
    assert runner._peek_session_state("s1").conversation.credential_pin is None


def test_benched_pinned_entry_serves_automatic_but_keeps_pin(codex_pool):
    from agent.credential_pool import load_pool

    codex_pool([
        _entry("aaa111", "work@example.com", TOKEN_A, REFRESH_A, 0),
        _entry("bbb222", "home@example.com", TOKEN_B, REFRESH_B, 1, last_status="exhausted",
               last_status_at=time.time(), last_error_code=429, last_error_reset_at=time.time() + 3600),
    ])
    runner = _runner(None)
    runner._session_state("s1").conversation.credential_pin = "bbb222"
    base = _codex_runtime(load_pool("openai-codex"))

    assert apply_session_credential_pin(runner, "s1", None, base) is base
    assert runner._peek_session_state("s1").conversation.credential_pin == "bbb222"


def test_new_conversation_boundary_clears_pin():
    from gateway.session_state import ConversationState

    state = ConversationState(credential_pin="bbb222")
    state.clear()
    assert state.credential_pin is None


# ---------------------------------------------------------------- /usage


def test_usage_view_names_resident_agents_real_entry(codex_pool):
    from agent.credential_pool import load_pool

    runner = _runner(None)
    runner._session_state("s1").conversation.credential_pin = "bbb222"
    agent = SimpleNamespace(_credential_pool=load_pool("openai-codex"), _credential_pool_entry_id="bbb222")

    key, url, line = session_credential_usage_view(runner, "s1", agent, "openai-codex")
    assert (key, url) == (None, None)
    assert line == "Session credential: home@example.com · bbb222 (pinned)"

    agent._credential_pool_entry_id = "aaa111"  # pin benched; strategy served the turn
    line = session_credential_usage_view(runner, "s1", agent, "openai-codex")[2]
    assert "aaa111" in line and "cools down" in line


def test_usage_view_without_agent_fetches_with_pinned_entry(codex_pool):
    runner = _runner(None)
    runner._session_state("s1").conversation.credential_pin = "bbb222"

    key, url, line = session_credential_usage_view(runner, "s1", None, "openai-codex")
    assert key == TOKEN_B and url == "https://chatgpt.com/backend-api/codex"
    _assert_no_secrets(line)
    assert session_credential_usage_view(runner, "s2", None, "openai-codex") == (None, None, None)
    assert session_credential_usage_view(runner, "s1", None, "anthropic") == (None, None, None)


@pytest.mark.asyncio
async def test_usage_command_shows_pinned_credential_and_fetches_its_quota(codex_pool, monkeypatch):
    import gateway.slash_commands_status as status_mod

    seen = {}

    def _fetch(provider, base_url=None, api_key=None):
        seen.update(provider=provider, api_key=api_key)
        return None

    monkeypatch.setattr(status_mod, "fetch_account_usage", _fetch)
    monkeypatch.setattr("agent.account_usage.nous_credits_lines", lambda markdown=True: [])
    monkeypatch.setattr(status_mod, "_configured_provider", lambda: "openai-codex")
    runner = _runner(_NoPickerAdapter())
    runner._resident_agent_for = lambda key: None
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, load_transcript=AsyncMock(return_value=[]),
        get_or_create_session=AsyncMock(return_value=SimpleNamespace(session_id="sid")))
    event = _event("/usage")
    runner._session_state(runner._session_key_for_source(event.source)).conversation.credential_pin = "bbb222"

    reply = await runner._handle_usage_command(event)

    assert seen == {"provider": "openai-codex", "api_key": TOKEN_B}
    assert "Session credential: home@example.com · bbb222 (pinned)" in reply
    _assert_no_secrets(reply)


# ---------------------------------------------------------------- label safety


@pytest.mark.parametrize("label", [TOKEN_A, f"key {TOKEN_A}", REFRESH_A, "x" * 40, ""])
def test_token_like_or_overlapping_labels_are_never_displayed(label):
    entry = SimpleNamespace(id="aaa111", label=label, access_token=TOKEN_A, refresh_token=REFRESH_A)
    shown = safe_credential_label(entry)
    assert shown == "credential"
    _assert_no_secrets(shown)


# ---------------------------------------------------------------- strategy accounting (real resolver)


def test_pinned_turn_resolution_never_charges_or_rotates_the_automatic_pick(codex_pool, tmp_path):
    """The real runtime ladder selects from the pool inside the pin's preference scope: the pinned
    entry serves and is charged; the strategy's pick is untouched (least_used + round_robin)."""
    from agent.credential_pool import load_pool

    home = tmp_path / "hermes"
    for strategy in ("least_used", "round_robin"):
        (home / "config.yaml").write_text(
            "model:\n  provider: openai-codex\n  default: gpt-5.5\n"
            f"credential_pool_strategies:\n  openai-codex: {strategy}\n", encoding="utf-8")
        codex_pool([_entry("aaa111", "work@example.com", TOKEN_A, REFRESH_A, 0),
                    _entry("bbb222", "home@example.com", TOKEN_B, REFRESH_B, 1)])
        runner = _runner(None)
        runner._session_state("s1").conversation.credential_pin = "bbb222"

        _model, runtime = runner._resolve_session_agent_runtime(session_key="s1")

        assert runtime["api_key"] == TOKEN_B, strategy
        pool = runtime["credential_pool"]
        assert pool.current().id == "bbb222"
        on_disk = {e.id: (e.request_count, e.priority) for e in load_pool("openai-codex").entries()}
        live = {e.id: (e.request_count, e.priority) for e in pool.entries()}
        assert live["aaa111"] == (0, 0), strategy  # never charged, never rotated
        assert live["bbb222"][0] == 1 and on_disk["aaa111"][1] == 0, strategy
        # An unpinned session still runs the strategy normally.
        assert runner._resolve_session_agent_runtime(session_key="s2")[1]["api_key"] == TOKEN_A


# ---------------------------------------------------------------- persistence across restart


def _store(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    return SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())


@pytest.mark.asyncio
async def test_pin_survives_gateway_restart_and_new_clears_it(codex_pool, tmp_path):
    store = _store(tmp_path)
    event = _event()
    entry = store.get_or_create_session(event.source)
    runner = _runner(_NoPickerAdapter())
    runner.session_store = store

    await runner._handle_credentials_command(_event("/credentials 2"))
    assert store.get_credential_pin(entry.session_key) == "bbb222"
    store._db.close()

    # Restart: fresh store over the same directory, fresh runner with no in-memory state.
    restarted_store = _store(tmp_path)
    restarted = _runner(None)
    restarted.session_store = restarted_store
    runtime = apply_session_credential_pin(restarted, entry.session_key, None, _codex_runtime(pool=None))
    assert runtime["api_key"] == TOKEN_B
    assert restarted._peek_session_state(entry.session_key).conversation.credential_pin == "bbb222"
    persisted = json.dumps(restarted_store._entries[entry.session_key].to_dict())
    _assert_no_secrets(persisted)

    # /new: the conversation funnel clears memory and the reset entry carries no pin.
    restarted._clear_conversation_scope(entry.session_key, reason="session_reset")
    restarted_store.reset_session(entry.session_key)
    assert session_credential_pin(restarted, entry.session_key) is None
    assert apply_session_credential_pin(restarted, entry.session_key, None, _codex_runtime())["api_key"] == TOKEN_A
    restarted_store._db.close()


def test_store_pin_is_sanitized_and_follows_only_non_boundary_repoints(tmp_path):
    store = _store(tmp_path)
    entry = store.get_or_create_session(_event().source)
    key = entry.session_key

    assert store.set_credential_pin(key, TOKEN_A)  # a token is never a valid pin
    assert store.get_credential_pin(key) is None
    store.set_credential_pin(key, "bbb222")

    other_sid = store.get_or_create_session(_event(chat_id="other").source).session_id
    store.switch_session(key, other_sid)  # compression-tip walk style repoint
    assert store.get_credential_pin(key) == "bbb222"
    resumed_sid = store.get_or_create_session(_event(chat_id="third").source).session_id
    store.switch_session(key, resumed_sid, preserve_prompt_pin=False)  # /resume
    assert store.get_credential_pin(key) is None
    store._db.close()


# ---------------------------------------------------------------- /usage


def test_usage_view_clears_a_deleted_pin_like_the_next_turn(codex_pool, tmp_path):
    store = _store(tmp_path)
    key = store.get_or_create_session(_event().source).session_key
    runner = _runner(None)
    runner.session_store = store
    runner._session_state(key).conversation.credential_pin = "bbb222"
    store.set_credential_pin(key, "bbb222")
    codex_pool([_entry("aaa111", "work@example.com", TOKEN_A, REFRESH_A, 0)])

    api_key, url, line = session_credential_usage_view(runner, key, None, "openai-codex")

    assert (api_key, url) == (None, None)
    assert line == "Session credential: automatic (the pinned credential was removed)"
    assert session_credential_pin(runner, key) is None and store.get_credential_pin(key) is None
    store._db.close()


def test_usage_view_for_benched_pin_reports_cooling_and_queries_the_pinned_account(codex_pool):
    codex_pool([
        _entry("aaa111", "work@example.com", TOKEN_A, REFRESH_A, 0),
        _entry("bbb222", "home@example.com", TOKEN_B, REFRESH_B, 1, last_status="exhausted",
               last_status_at=time.time(), last_error_code=429, last_error_reset_at=time.time() + 3600),
    ])
    runner = _runner(None)
    runner._session_state("s1").conversation.credential_pin = "bbb222"

    api_key, _url, line = session_credential_usage_view(runner, "s1", None, "openai-codex")

    assert api_key == TOKEN_B  # /usage reset redeems the pinned (benched) account
    assert "home@example.com · bbb222" in line and "cooling down" in line
    assert runner._peek_session_state("s1").conversation.credential_pin == "bbb222"
    _assert_no_secrets(line)


def test_usage_view_refreshes_an_expiring_pinned_token_through_the_pool(codex_pool, monkeypatch, caplog):
    import hermes_cli.auth as auth_mod

    expiring = _jwt({"exp": int(time.time()) + 30, "sub": "b", "nonce": "secret-b-expiring"})
    minted = _jwt({"exp": int(time.time()) + 86400, "sub": "b", "nonce": "secret-b-minted"})
    codex_pool([_entry("aaa111", "work@example.com", TOKEN_A, REFRESH_A, 0),
                _entry("bbb222", "home@example.com", expiring, REFRESH_B, 1)])
    posted = []

    def _refresh(access_token, refresh_token):
        posted.append(refresh_token)
        return {"access_token": minted, "refresh_token": "rt-secret-refresh-b-rotated", "last_refresh": None}

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", _refresh)
    runner = _runner(None)
    runner._session_state("s1").conversation.credential_pin = "bbb222"

    with caplog.at_level("DEBUG"):
        api_key, _url, line = session_credential_usage_view(runner, "s1", None, "openai-codex")

    assert posted == [REFRESH_B] and api_key == minted
    assert line == "Session credential: home@example.com · bbb222 (pinned)"
    for secret in (expiring, minted, REFRESH_B, "rt-secret-refresh-b-rotated"):
        assert secret not in line and secret not in caplog.text


@pytest.mark.asyncio
async def test_usage_command_reads_the_pin_under_the_normalized_topic_key(codex_pool, monkeypatch):
    import gateway.slash_commands_status as status_mod

    seen = {}
    monkeypatch.setattr(status_mod, "fetch_account_usage",
                        lambda provider, base_url=None, api_key=None: seen.update(api_key=api_key))
    monkeypatch.setattr("agent.account_usage.nous_credits_lines", lambda markdown=True: [])
    monkeypatch.setattr(status_mod, "_configured_provider", lambda: "openai-codex")
    runner = _runner(_NoPickerAdapter())
    runner._resident_agent_for = lambda key: None
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, load_transcript=AsyncMock(return_value=[]),
        get_or_create_session=AsyncMock(return_value=SimpleNamespace(session_id="sid")))
    # A lobby reply recovered to the user's active topic, as the next turn would key it.
    runner._recover_telegram_topic_thread_id = lambda source: "77"
    event = _event("/usage")
    await runner._handle_credentials_command(_event("/credentials 2"))
    topic_key = runner._session_key_for_source(replace(event.source, thread_id="77"))
    assert runner._peek_session_state(topic_key).conversation.credential_pin == "bbb222"

    reply = await runner._handle_usage_command(event)

    assert seen == {"api_key": TOKEN_B}
    assert "home@example.com · bbb222 (pinned)" in reply


# ---------------------------------------------------------------- cached agent after a 429 on the pin


class _LiveAgent:
    """The credential surface of a gateway-cached AIAgent that turn-start restore touches."""

    def __init__(self, pool, entry):
        self._credential_pool, self.model = pool, "gpt-5.5"
        self._fallback_activated, self._fallback_index, self._credential_pool_revert_id = False, 0, None
        self._swap_credential(entry)

    def _swap_credential(self, entry):
        self.api_key, self._credential_pool_entry_id = entry.runtime_api_key, entry.id


def test_cached_agent_returns_to_pin_once_eligible_after_429_rotation(codex_pool):
    """Pinned entry 429s → the cached agent rotates within the pool (automatic serves while the pin
    cools). Once the pin is eligible the resolved runtime hashes to the SAME cache signature as the
    agent's build, so the agent is reused; turn-start restore must move its live client back."""
    from agent.agent_runtime_helpers import restore_primary_runtime
    from agent.credential_pool import load_pool
    from gateway.slash_commands_credentials import agent_credential_pin

    runner = _runner(None)
    runner._session_state("s1").conversation.credential_pin = "bbb222"
    pool = load_pool("openai-codex")
    built = apply_session_credential_pin(runner, "s1", "gpt-5.5", _codex_runtime(pool))
    agent = _LiveAgent(pool, pool.current())
    built_sig = runner._agent_config_signature("gpt-5.5", built, [], "")
    assert agent.api_key == TOKEN_B

    # 429 on the pin: 429 recovery rotates the live agent to the automatic entry. The benched pin
    # does not outrank it, so the generic quota-revert hook is NOT armed (the gap this covers).
    rotated = pool.mark_exhausted_and_rotate(status_code=429, credential_id="bbb222",
                                            error_context={"reset_at": time.time() + 3600})
    agent._swap_credential(rotated)
    assert agent._credential_pool_entry_id == "aaa111" and agent._credential_pool_revert_id is None

    # Next turn, pin still cooling: automatic keeps serving; the agent stays on the rotated entry.
    cooling = apply_session_credential_pin(runner, "s1", "gpt-5.5", _codex_runtime(pool))
    assert cooling["api_key"] == TOKEN_A
    agent._credential_pool_pin = agent_credential_pin(runner, "s1")
    restore_primary_runtime(agent)
    assert agent._credential_pool_entry_id == "aaa111" and agent.api_key == TOKEN_A

    # The pin's window reopens: the turn resolves onto it and matches the cached signature...
    pool.reset_status("bbb222")
    eligible = apply_session_credential_pin(runner, "s1", "gpt-5.5", _codex_runtime(pool))
    assert eligible["api_key"] == TOKEN_B
    assert runner._agent_config_signature("gpt-5.5", eligible, [], "") == built_sig  # → cached agent reused
    # ...so the reused agent's turn start must put it back on the pin.
    agent._credential_pool_pin = agent_credential_pin(runner, "s1")
    restore_primary_runtime(agent)
    assert agent._credential_pool_entry_id == "bbb222" and agent.api_key == TOKEN_B
    assert pool.current().id == "bbb222"  # a later 429 is attributed to the pin

    # Automatic (unpinned) sessions keep the existing priority-gated revert behaviour.
    unpinned = _LiveAgent(pool, next(e for e in pool.entries() if e.id == "aaa111"))
    unpinned._credential_pool_pin = agent_credential_pin(runner, "other")
    restore_primary_runtime(unpinned)
    assert unpinned._credential_pool_entry_id == "aaa111"


# ---------------------------------------------------------------- handoff boundary


@pytest.mark.asyncio
async def test_handoff_drops_in_memory_pin_but_keeps_model_override(codex_pool, tmp_path):
    store = _store(tmp_path)
    source = _event().source
    key = store.get_or_create_session(source).session_key
    cli_sid = store.get_or_create_session(_event(chat_id="cli").source).session_id
    runner = _runner(None)
    runner.session_store = store
    await runner._handle_credentials_command(_event("/credentials 2"))  # memory + routing entry
    runner._session_state(key).conversation.model_override = {"model": "gpt-5.5", "provider": "openai-codex"}
    runner._handoff_resolve_destination = AsyncMock(return_value=SimpleNamespace(
        source=source, platform_name="telegram", home=SimpleNamespace(chat_id="67890"), effective_thread_id=None))
    runner._handoff_session_key = lambda dest, profile: key
    runner._release_running_agent_state = lambda session_key: None
    runner._handle_message = AsyncMock(return_value="")
    assert session_credential_pin(runner, key) == "bbb222"

    await runner._process_handoff({"id": cli_sid, "title": "t"})

    assert store.get_credential_pin(key) is None
    assert session_credential_pin(runner, key) is None  # in-memory pin no longer outlives the boundary
    assert runner._peek_session_state(key).conversation.model_override["model"] == "gpt-5.5"
    assert apply_session_credential_pin(runner, key, None, _codex_runtime())["api_key"] == TOKEN_A
    store._db.close()
