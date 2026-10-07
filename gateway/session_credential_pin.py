"""SessionStore credential pins: the ``/credentials`` pool-entry choice, durable across a restart.

Only the pool entry's stable id is stored (short random hex, never a token); the token is always
re-read from the pool. The pin belongs to one conversation: ``/new`` and every other reset build a
fresh routing entry without it, ``/resume`` and handoffs repoint with ``preserve_prompt_pin=False``
(which drops it too), and non-boundary repoints (compression tip walks) carry it.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Optional

# Pool ids are ``uuid4().hex[:6]``; anything outside this shape (a pasted token, a JSON blob) is refused.
_PIN_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def sanitize_credential_pin(pin: Any) -> Optional[str]:
    """*pin* when it is a plausible pool entry id, else None."""
    return pin if isinstance(pin, str) and _PIN_RE.fullmatch(pin) else None


class SessionCredentialPinMixin:
    """Read/write of ``SessionEntry.credential_pin`` on the routing index."""

    def set_credential_pin(self, session_key: str, pin: Optional[str]) -> bool:
        """Persist (or clear, with None) the session's ``/credentials`` pin.

        Args:
            session_key (str): Routing key.
            pin (str | None): Pool entry id; None (or an id that fails sanitizing) clears.

        Returns:
            bool: False when the session has no routing entry yet (nothing persisted).

        Called by:
            - GatewayCredentialsCommandsMixin._set_session_credential_pin()
            - gateway.slash_commands_credentials._clear_missing_pin()
        """
        cleaned = sanitize_credential_pin(pin)
        with self._lock:
            entry = self._entry_locked(session_key)
            if entry is None:
                return False
            if entry.credential_pin != cleaned:
                self._save_entry(
                    session_key, entry_data=replace(entry, credential_pin=cleaned).to_dict(), lock_held=True)
                entry.credential_pin = cleaned
            return True

    def get_credential_pin(self, session_key: str) -> Optional[str]:
        """The persisted ``/credentials`` pin for *session_key*, if any.

        Called by:
            - gateway.slash_commands_credentials.session_credential_pin() - lazy rehydrate after restart
        """
        with self._lock:
            entry = self._entry_locked(session_key)
            return entry.credential_pin if entry is not None else None
