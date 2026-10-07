"""Telegram generic choice picker (``cp:`` callbacks): every tap is bound to the picker message it
was sent as, so stale keyboards, sibling forum topics and other pickers in the same chat can never
dispatch into a different picker's callback or session; callback failures never echo their text."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig

SECRET = "rt-secret-refresh-token-0123456789abcdef"


@pytest.fixture
def adapter(monkeypatch):
    import plugins.platforms.telegram.adapter as tg

    monkeypatch.setattr(tg, "InlineKeyboardButton", lambda text, callback_data: SimpleNamespace(
        text=text, callback_data=callback_data))
    monkeypatch.setattr(tg, "InlineKeyboardMarkup", lambda rows: SimpleNamespace(inline_keyboard=rows))
    adapter = tg.TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = MagicMock()
    message_ids = iter(range(100, 200))
    adapter._send_control_message = AsyncMock(side_effect=lambda *a, **k: SimpleNamespace(message_id=next(message_ids)))
    return adapter


async def _send(adapter, session_key, chat_id="-100", thread_id=None):
    """Send a two-choice picker; returns (callback_data list, message_id, recorded selections)."""
    picked = []

    async def on_choice(chat, value):
        picked.append((chat, value))
        return f"{session_key} -> {value}"

    result = await adapter.send_choice_picker(
        chat_id=chat_id, title="Pick", session_key=session_key, on_choice_selected=on_choice,
        choices=[{"value": "low", "label": "Low"}, {"value": "high", "label": "High"}],
        metadata={"thread_id": thread_id} if thread_id else None)
    markup = adapter._send_control_message.await_args.kwargs["reply_markup"]
    data = [b.callback_data for row in markup.inline_keyboard for b in row]
    return data, int(result.message_id), picked


def _query(message_id):
    return SimpleNamespace(answer=AsyncMock(), edit_message_text=AsyncMock(),
                           message=SimpleNamespace(message_id=message_id))


def _expired(query) -> bool:
    return query.answer.await_args.kwargs.get("text", "").startswith("Picker expired")


@pytest.mark.asyncio
async def test_old_keyboard_is_rejected_after_a_newer_picker_for_the_same_session(adapter):
    old_data, old_msg, old_picked = await _send(adapter, "s1")
    new_data, new_msg, new_picked = await _send(adapter, "s1")

    stale = _query(old_msg)
    await adapter._handle_choice_picker_callback(stale, old_data[1], "-100")
    assert _expired(stale) and not old_picked and not new_picked
    # Pre-upgrade keyboards (``cp:<index>``) are expired too, never routed by index.
    legacy = _query(new_msg)
    await adapter._handle_choice_picker_callback(legacy, "cp:1", "-100")
    assert _expired(legacy) and not new_picked

    fresh = _query(new_msg)
    await adapter._handle_choice_picker_callback(fresh, new_data[1], "-100")
    assert new_picked == [("-100", "high")] and not old_picked
    # One tap, one apply: a second tap on the claimed picker is expired.
    again = _query(new_msg)
    await adapter._handle_choice_picker_callback(again, new_data[0], "-100")
    assert _expired(again) and new_picked == [("-100", "high")]


@pytest.mark.asyncio
async def test_forum_topics_in_one_chat_keep_their_own_pickers(adapter):
    a_data, a_msg, a_picked = await _send(adapter, "topic-a", thread_id="11")
    b_data, b_msg, b_picked = await _send(adapter, "topic-b", thread_id="22")

    tap_a = _query(a_msg)
    await adapter._handle_choice_picker_callback(tap_a, a_data[0], "-100")
    assert a_picked == [("-100", "low")] and not b_picked
    assert tap_a.edit_message_text.await_args.kwargs["text"].startswith("topic")

    tap_b = _query(b_msg)
    await adapter._handle_choice_picker_callback(tap_b, b_data[1], "-100")
    assert b_picked == [("-100", "high")] and a_picked == [("-100", "low")]


@pytest.mark.asyncio
async def test_callback_data_never_crosses_pickers_messages_or_chats(adapter):
    a_data, a_msg, a_picked = await _send(adapter, "s-a")
    b_data, b_msg, b_picked = await _send(adapter, "s-b")
    assert a_data[0] != b_data[0]  # same index, distinct opaque ids

    for data, msg_id, chat in ((a_data[0], b_msg, "-100"),   # A's button data on B's message
                               (b_data[0], a_msg, "-100"),   # B's button data on A's message
                               (a_data[0], a_msg, "-999")):  # right message id, other chat
        query = _query(msg_id)
        await adapter._handle_choice_picker_callback(query, data, chat)
        assert _expired(query)
    assert not a_picked and not b_picked


@pytest.mark.asyncio
async def test_secret_bearing_callback_error_never_reaches_chat_or_logs(adapter, caplog):
    async def boom(chat, value):
        raise RuntimeError(f"pool refresh failed: refresh_token={SECRET}")

    result = await adapter.send_choice_picker(
        chat_id="-100", title="Pick", session_key="s1", on_choice_selected=boom,
        choices=[{"value": "x", "label": "X"}])
    markup = adapter._send_control_message.await_args.kwargs["reply_markup"]
    query = _query(int(result.message_id))
    with caplog.at_level("DEBUG"):
        await adapter._handle_choice_picker_callback(query, markup.inline_keyboard[0][0].callback_data, "-100")

    shown = " ".join(str(c) for c in query.edit_message_text.await_args_list)
    assert "RuntimeError" in shown
    assert SECRET not in shown and SECRET not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_index", ["-1", "-2", "+1", " 1", "2", "99", "", "¹", "1_0"])
async def test_out_of_range_or_signed_index_is_rejected_not_indexed_from_the_end(adapter, bad_index):
    data, msg_id, picked = await _send(adapter, "s1")
    picker_id = data[0].split(":")[1]

    bad = _query(msg_id)
    await adapter._handle_choice_picker_callback(bad, f"cp:{picker_id}:{bad_index}", "-100")
    assert "Invalid selection" in bad.answer.await_args.kwargs["text"]
    assert not picked  # "-1" must never select the last choice

    # The rejected tap did not claim the picker: a real button still applies once.
    good = _query(msg_id)
    await adapter._handle_choice_picker_callback(good, data[0], "-100")
    assert picked == [("-100", "low")]
