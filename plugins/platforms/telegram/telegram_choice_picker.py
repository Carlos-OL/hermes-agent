"""Telegram generic choice picker (/reasoning, /fast, /credentials, ...): one tap -> one value.

Each picker gets an opaque random id carried in its buttons' ``callback_data`` (``cp:<id>:<index>``,
well inside Telegram's 64-byte cap) and its state is keyed by that id, never by chat. A tap is
honoured only when the id is live and the tap arrives on the very message the picker was sent as,
in the same chat; a newer picker for the same session retires the older keyboard. Anything else (an
old keyboard, a sibling forum topic's picker, a keyboard from before a restart) answers "expired",
so a tap can never dispatch into a different picker's callback or session.
"""

import logging
import secrets
from typing import Any, Dict, Optional

from agent.i18n import t
from gateway.platforms.base import SendResult

logger = logging.getLogger("plugins.platforms.telegram.adapter")

_PICKER_ID_BYTES = 6  # token_urlsafe -> 8 chars of [A-Za-z0-9_-]; no ":" so the index parses cleanly
_MAX_LIVE_PICKERS = 64  # unanswered pickers kept; the oldest is dropped first


class TelegramChoicePickerMixin:
    """``send_choice_picker`` + its ``cp:`` callback for ``TelegramAdapter``."""

    async def send_choice_picker(
        self, chat_id: str, title: str, choices: list, session_key: str, on_choice_selected,
        metadata: Optional[Dict[str, Any]] = None) -> SendResult:
        """Flat inline-keyboard picker (one tap -> one value).

        Args:
            chat_id (str): Destination chat.
            title (str): Picker message body (Markdown, formatted here).
            choices (list): ``{"value": str, "label": str, "is_current": bool}`` dicts.
            session_key (str): Session the picker belongs to; a newer picker for it retires this one.
            on_choice_selected: ``async (chat_id, value) -> str`` result text.
            metadata (dict | None): Thread routing metadata.

        Returns:
            SendResult: The send outcome.

        Called by:
            - GatewayModelCommandsMixin._try_send_choice_picker()

        Calls:
            - _send_prompt() - routed send + state hook
        """
        from plugins.platforms.telegram import adapter as tg

        picker_id = secrets.token_urlsafe(_PICKER_ID_BYTES)

        def build():
            buttons = []
            for i, choice in enumerate(choices):
                label = str(choice.get("label") or choice.get("value") or "")
                if choice.get("is_current"):
                    label = f"✓ {label}"
                buttons.append(tg.InlineKeyboardButton(label, callback_data=f"cp:{picker_id}:{i}"))
            if not buttons:
                return SendResult(success=False, error="No choices")
            keyboard = tg.InlineKeyboardMarkup(self._rows_of_two(buttons))

            def _remember(msg):
                self._remember_choice_picker(picker_id, {
                    "chat_id": str(chat_id), "msg_id": msg.message_id, "choices": choices,
                    "session_key": session_key, "on_choice_selected": on_choice_selected})
            return self.format_message(title), keyboard, _remember
        return await self._send_prompt(
            "send_choice_picker", chat_id, metadata, build, thread_id=metadata.get("thread_id") if metadata else None,
            reply_to_mode=self._reply_to_mode)

    def _remember_choice_picker(self, picker_id: str, state: dict) -> None:
        """Register a sent picker, retiring older pickers of the same session and capping the table."""
        pickers = self._choice_picker_state
        if state["session_key"]:
            for stale in [pid for pid, s in pickers.items() if s["session_key"] == state["session_key"]]:
                del pickers[stale]
        pickers[picker_id] = state
        while len(pickers) > _MAX_LIVE_PICKERS:
            del pickers[next(iter(pickers))]

    async def _handle_choice_picker_callback(self, query, data: str, chat_id: str) -> None:
        """Handle a ``cp:<picker id>:<index>`` tap: claim the picker, run its callback, edit in place.

        Args:
            query: Telegram CallbackQuery.
            data (str): The button's callback_data.
            chat_id (str): Chat the tapped message lives in.

        Called by:
            - TelegramAdapter._handle_callback_query() - after the callback auth gate
        """
        from plugins.platforms.telegram.adapter import _toast

        picker_id, _sep, index = data[3:].rpartition(":")
        state = self._choice_picker_state.get(picker_id) if picker_id else None
        tapped_msg_id = getattr(getattr(query, "message", None), "message_id", None)
        if state is None or state["chat_id"] != chat_id or state["msg_id"] != tapped_msg_id:
            await query.answer(text=_toast("platform.telegram.picker.expired_rerun"))
            return
        # Plain ASCII digits in range only: int() would accept "-1" / "+1" / " 1" and a negative
        # index would silently select from the end of the list.
        if not (index.isascii() and index.isdigit() and int(index) < len(state["choices"])):
            await query.answer(text=_toast("platform.telegram.picker.invalid_selection"))
            return
        choice = state["choices"][int(index)]
        # Claim before dispatch so a double tap cannot apply the selection twice.
        del self._choice_picker_state[picker_id]
        try:
            result_text = await state["on_choice_selected"](chat_id, str(choice.get("value") or ""))
        except Exception as exc:  # health: allow BLE001 -- a picker callback must not crash the update loop; type name only: callback errors can carry auth/token payloads
            logger.error("Choice picker selection failed (%s)", type(exc).__name__)
            result_text = t("platform.telegram.picker.apply_error", error=type(exc).__name__)
        await self._edit_result_text(query, result_text)
        await query.answer()
