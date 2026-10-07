"""``/credentials``: pin one session to an OpenAI-Codex pool entry, or put it back on Automatic.

The pin is the entry's stable id (never a token) on ``ConversationState.credential_pin``, written
through to ``SessionEntry.credential_pin`` so it survives a gateway restart and dropped at
conversation boundaries (/new, /resume). Tokens never leave the pool: picker values, replies and logs
carry only a safe label and a short id. Each turn resolves inside a ``preferred_pool_entry`` scope, so
the pool serves (and charges) the pinned row through its own locks and refresh, never rotating or
counting the strategy's pick; the pool stays attached to the runtime, so 429 recovery and ``/usage``
attribute requests to the row that actually served them.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, NamedTuple, Optional, Tuple

from agent.i18n import t

logger = logging.getLogger("gateway.run")

CODEX_PROVIDER = "openai-codex"
AUTOMATIC_CHOICE = "auto"
_AUTOMATIC_ARGS = frozenset({AUTOMATIC_CHOICE, "automatic", "off", "clear", "reset", "none"})
_SHORT_ID_LEN = 6
_LABEL_MAX = 48
# A label with no whitespace or "@" this long reads as a pasted secret, never as a name.
_SECRET_LIKE_MIN_LEN = 32


class CredentialChoice(NamedTuple):
    """Display-safe view of one pool entry: the only shape that leaves the pool loader."""

    id: str
    display: str


class CredentialUsageTarget(NamedTuple):
    """Internal-only quota probe target. ``api_key`` is never rendered or logged."""

    display: str
    api_key: str
    base_url: Optional[str]


def short_credential_id(entry_id: Any) -> str:
    """First characters of a pool entry id (pool ids are short random hex, never secrets)."""
    return str(entry_id or "")[:_SHORT_ID_LEN]


def _looks_like_secret(label: str) -> bool:
    return len(label) >= _SECRET_LIKE_MIN_LEN and "@" not in label and not any(c.isspace() for c in label)


def safe_credential_label(entry: Any) -> str:
    """User-visible label for a pool entry that can never be (or contain) one of its tokens.

    Args:
        entry (PooledCredential): Pool entry.

    Returns:
        str: The entry's label (whitespace-collapsed, truncated), or a generic name when the label
        is empty, overlaps one of the entry's tokens, or looks like a secret.

    Called by:
        - credential_display()
    """
    label = " ".join(str(getattr(entry, "label", "") or "").split())
    secrets = [
        s for s in (getattr(entry, name, None) for name in ("access_token", "refresh_token", "agent_key"))
        if isinstance(s, str) and len(s) >= 8
    ]
    overlaps = any(s in label or (len(label) >= 8 and label in s) for s in secrets)
    if not label or overlaps or _looks_like_secret(label):
        return t("gateway.credentials.unnamed")
    return label if len(label) <= _LABEL_MAX else label[: _LABEL_MAX - 1] + "…"


def credential_display(entry: Any) -> str:
    """``label · shortid`` for replies, buttons and /usage."""
    return f"{safe_credential_label(entry)} · {short_credential_id(entry.id)}"


def load_codex_pool():
    """The live openai-codex credential pool (profile-scoped by the caller's HERMES_HOME)."""
    from agent.credential_pool import load_pool
    return load_pool(CODEX_PROVIDER)


def load_codex_choices() -> Tuple[List[CredentialChoice], str]:
    """``(choices, strategy)`` for the picker; tokens stay inside this function.

    Returns:
        tuple: Display-safe choices in pool order, and the configured pool strategy name.

    Called by:
        - GatewayCredentialsCommandsMixin._handle_credentials_command() - off the event loop
    """
    from agent.credential_pool import get_pool_strategy
    choices = [CredentialChoice(e.id, credential_display(e)) for e in load_codex_pool().entries()]
    return choices, get_pool_strategy(CODEX_PROVIDER)


def load_codex_usage_targets() -> List[CredentialUsageTarget]:
    """Fresh quota-probe targets for every Codex pool row, without selecting or rotating.

    Refreshing an expiring OAuth token is allowed and persisted through the pool's normal locks;
    the strategy's current entry and request counters are untouched. Returned tokens are consumed
    only by ``/usage``'s read-only fetcher and must never be included in user-visible output.
    """
    pool = load_codex_pool()
    targets: List[CredentialUsageTarget] = []
    for listed in pool.entries():
        entry = pool.fresh_entry(listed.id) or listed
        api_key = str(entry.runtime_api_key or "").strip()
        if api_key:
            targets.append(CredentialUsageTarget(
                credential_display(entry), api_key, entry.runtime_base_url or None))
    return targets


def _session_store(runner: Any):
    """The runner's sync SessionStore, or None (bare test runners, stores without pin support)."""
    store = getattr(runner, "session_store", None)
    return store if hasattr(store, "get_credential_pin") else None


def session_credential_pin(runner: Any, session_key: Optional[str]) -> Optional[str]:
    """The session's pinned pool entry id, or None (Automatic).

    In-memory first; otherwise the pin persisted on the session's routing entry is rehydrated
    (first use after a gateway restart). A boundary (/new, /resume) replaces that entry, so a
    cleared conversation never resurrects an old pin.

    Called by:
        - GatewayTurnRuntimeMixin._resolve_session_agent_runtime() - the turn's pool preference
        - apply_session_credential_pin(), session_credential_usage_view(), /credentials
    """
    if not session_key:
        return None
    state = runner._peek_session_state(session_key)
    pin = state.conversation.credential_pin if state is not None else None
    if pin:
        return pin
    store = _session_store(runner)
    try:
        pin = store.get_credential_pin(session_key) if store is not None else None
    except Exception:  # health: allow BLE001 -- a store read failure leaves the session on Automatic
        logger.debug("Failed to read the persisted /credentials pin", exc_info=True)
        return None
    if pin:
        runner._session_state(session_key).conversation.credential_pin = pin
        logger.info("Rehydrated /credentials pin %s for session=%s", short_credential_id(pin), session_key)
    return pin or None


def agent_credential_pin(runner: Any, session_key: Optional[str]) -> Optional[Tuple[str, str]]:
    """``(provider, entry id)`` for ``AIAgent._credential_pool_pin``, or None (Automatic).

    The gateway caches agents by a signature that hashes the key the agent was built with, so a
    reused agent can still be on the entry a 429 rotated it to after its pin became eligible again.
    The agent's turn-start restore (``credential_pool_admin.return_to_pinned_entry``) reads this
    to move back onto the pin; while the pin cools down, automatic selection keeps serving.

    Args:
        runner (GatewayRunner): Owner of the per-session state.
        session_key (str | None): Session key of the turn.

    Returns:
        tuple | None: ``(CODEX_PROVIDER, pin)`` when the session is pinned.

    Called by:
        - GatewayTurnRunner._wire_turn_agent_callbacks() - every turn, fresh or reused agent
    """
    pin = session_credential_pin(runner, session_key)
    return (CODEX_PROVIDER, pin) if pin else None


def _clear_missing_pin(runner: Any, session_key: str, pin: str) -> None:
    """Drop a pin whose pool entry was deleted (in memory and on the routing entry)."""
    state = runner._peek_session_state(session_key)
    if state is not None and state.conversation.credential_pin == pin:
        state.conversation.credential_pin = None
    store = _session_store(runner)
    if store is not None and store.get_credential_pin(session_key) == pin:
        store.set_credential_pin(session_key, None)
    logger.warning(
        "Pinned credential %s is no longer in the %s pool; session back on automatic selection",
        short_credential_id(pin), CODEX_PROVIDER,
    )


def apply_session_credential_pin(runner: Any, session_key: Optional[str], model: Optional[str], runtime: dict) -> dict:
    """Put a resolved openai-codex runtime on the session's pinned pool entry.

    A runtime the resolution already took from the pin (``preferred_pool_entry`` scope) is returned
    as-is. Otherwise (the ``/model`` fast path never selects) the pinned row is selected through
    ``CredentialPool.select_pinned`` (cooldown clearing and token refresh under the normal
    pool/auth-store locks, no strategy accounting) and the pool stays attached, so 429 recovery
    rotates within it and ``entry_id_for_api_key`` names the pinned row. A pin whose entry was
    deleted is cleared (Automatic); a benched pin keeps the strategy's pick for this turn and is
    retried next turn.

    Args:
        runner (GatewayRunner): Owner of the per-session state.
        session_key (str | None): Session whose pin applies.
        model (str | None): Resolved model (per-model cooldowns).
        runtime (dict): Resolved runtime kwargs.

    Returns:
        dict: ``runtime`` unchanged, or a copy with the pinned ``api_key`` and its ``credential_pool``.

    Called by:
        - GatewayTurnRuntimeMixin._resolve_session_agent_runtime()

    Calls:
        - CredentialPool.select_pinned() - refresh + make the pin ``current``
    """
    pin = session_credential_pin(runner, session_key)
    if pin is None or str(runtime.get("provider") or "").strip().lower() != CODEX_PROVIDER:
        return runtime
    pool = runtime.get("credential_pool")
    try:
        if getattr(pool, "provider", None) != CODEX_PROVIDER:
            pool = load_codex_pool()
        if not any(e.id == pin for e in pool.entries()):
            _clear_missing_pin(runner, session_key, pin)
            return runtime
        current = pool.current()
        if (pool is runtime.get("credential_pool") and current is not None and current.id == pin
                and current.runtime_api_key == runtime.get("api_key")):
            return runtime
        entry = pool.select_pinned(pin, model=model or None)
    except Exception as exc:  # health: allow BLE001 -- a pin must never fail the turn; no traceback: refresh errors can carry auth payloads
        logger.warning("Pinned credential %s could not be resolved (%s); using automatic selection",
                       short_credential_id(pin), type(exc).__name__)
        return runtime
    api_key = getattr(entry, "runtime_api_key", "") if entry is not None else ""
    if not api_key:
        logger.info("Pinned credential %s is cooling down; automatic selection serves this turn",
                    short_credential_id(pin))
        return runtime
    return {**runtime, "api_key": api_key, "credential_pool": pool}


def session_credential_usage_view(
    runner: Any, session_key: Optional[str], agent: Any, provider: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``(api_key, base_url, line)`` for /usage: which credential the session is really on.

    With a resident agent the line names the entry the agent dispatched with
    (``_credential_pool_entry_id``). With none, a pinned session's quota is fetched (and a
    ``/usage reset`` redeemed) with the pinned entry, read through the pool: token refreshed under
    the pool/auth-store locks when expiring, cooldown state reported, never charged to the strategy.
    A pin whose entry was deleted is cleared exactly as the next turn would clear it. ``api_key``
    only feeds the quota fetch — it is never rendered.

    Args:
        runner (GatewayRunner): Owner of the per-session state.
        session_key (str | None): Session key, normalized like the turn's.
        agent (AIAgent | None): The session's resident agent, if any.
        provider (str | None): The provider /usage reports on.

    Returns:
        tuple: Quota-fetch overrides (None = keep /usage's own) and a display line (or None).

    Called by:
        - GatewayStatusCommandsMixin._handle_usage_command() - off the event loop

    Calls:
        - CredentialPool.fresh_entry() / reclaim() - refresh, then availability
    """
    if str(provider or "").strip().lower() != CODEX_PROVIDER:
        return None, None, None
    pin = session_credential_pin(runner, session_key)
    if agent is not None:
        entry_id = getattr(agent, "_credential_pool_entry_id", None)
        pool = getattr(agent, "_credential_pool", None)
        entry = next((e for e in pool.entries() if e.id == entry_id), None) if pool and entry_id else None
        if entry is None:
            return None, None, None
        mode = "pinned" if entry.id == pin else ("pin_waiting" if pin else "automatic")
        return None, None, t("gateway.credentials.usage_line", credential=credential_display(entry),
                             mode=t(f"gateway.credentials.mode_{mode}"))
    if pin is None:
        return None, None, None
    pool = load_codex_pool()
    listed = next((e for e in pool.entries() if e.id == pin), None)
    if listed is None:
        _clear_missing_pin(runner, session_key, pin)
        return None, None, t("gateway.credentials.usage_pin_removed")
    entry = pool.fresh_entry(pin)
    available = entry is not None and pool.reclaim(pin) is not None
    line = t("gateway.credentials.usage_line", credential=credential_display(entry or listed),
             mode=t("gateway.credentials.mode_pinned" if available else "gateway.credentials.mode_pinned_cooling"))
    if entry is None:
        return None, None, line
    return entry.runtime_api_key or None, entry.runtime_base_url or None, line


def _match_choice(arg: str, choices: List[CredentialChoice]) -> Optional[CredentialChoice]:
    """Exact entry id first (ids can be all digits), then a 1-based list index."""
    match = next((c for c in choices if c.id == arg), None)
    if match is None and arg.isdigit() and 1 <= int(arg) <= len(choices):
        match = choices[int(arg) - 1]
    return match


class GatewayCredentialsCommandsMixin:
    """``/credentials`` on GatewayRunner."""

    async def _set_session_credential_pin(self, session_key: str, entry_id: Optional[str]) -> None:
        """Pin (or with None unpin) the session, write the id through to its routing entry (survives
        a restart; never a token), and evict its cached agent so the next turn rebuilds on the new
        credential instead of reusing the old client."""
        self._session_state(session_key).conversation.credential_pin = entry_id
        if _session_store(self) is not None:
            try:
                await self.async_session_store.set_credential_pin(session_key, entry_id)
            except Exception:  # health: allow BLE001 -- the in-memory pin still applies this process
                logger.warning("Failed to persist the /credentials pin for session=%s", session_key, exc_info=True)
        self._evict_cached_agent(session_key)
        logger.info("Session %s credential selection -> %s", session_key,
                    short_credential_id(entry_id) if entry_id else "automatic")

    async def _apply_credential_selection(self, session_key: str, arg: str, choices: List[CredentialChoice],
                                          strategy: str) -> str:
        """Apply a typed or picked /credentials argument and return the reply (never echoes *arg*,
        which may be a pasted token)."""
        arg = arg.strip()
        if arg.lower() in _AUTOMATIC_ARGS:
            await self._set_session_credential_pin(session_key, None)
            return t("gateway.credentials.automatic", strategy=strategy)
        match = _match_choice(arg, choices)
        if match is None:
            return t("gateway.credentials.unknown")
        await self._set_session_credential_pin(session_key, match.id)
        return t("gateway.credentials.pinned", credential=match.display)

    def _credentials_text(self, choices: List[CredentialChoice], pin: Optional[str], current: str) -> str:
        """Plain-text /credentials body for platforms without buttons."""
        lines = [t("gateway.credentials.status_header", current=current), ""]
        lines.append(t("gateway.credentials.automatic_line", marker=" ✓" if pin is None else ""))
        lines += [
            t("gateway.credentials.entry_line", index=i, credential=c.display, marker=" ✓" if c.id == pin else "")
            for i, c in enumerate(choices, start=1)
        ]
        lines += ["", t("gateway.credentials.usage")]
        return "\n".join(lines)

    async def _handle_credentials_command(self, event) -> Optional[str]:
        """Handle /credentials — pick this session's OpenAI-Codex pool credential (buttons where the
        adapter supports a choice picker, else a text list + ``/credentials <id|number|auto>``)."""
        # Normalized like /reasoning (#30479) so the pin lands under the key the next turn reads.
        source = await asyncio.to_thread(self._normalize_source_for_session_key, event.source)
        session_key = self._session_key_for_source(source)
        profile_home = None
        if getattr(getattr(self, "config", None), "multiplex_profiles", False):
            profile_home = self._resolve_profile_home_for_source(event.source)
        try:
            choices, strategy = await asyncio.to_thread(load_codex_choices)
        except Exception as exc:  # health: allow BLE001 -- reply instead of erroring; no traceback: pool load errors can carry auth payloads
            logger.warning("/credentials could not load the %s pool: %s", CODEX_PROVIDER, type(exc).__name__)
            return t("gateway.credentials.load_failed")
        raw_args = event.get_command_args().strip()
        if raw_args:
            return await self._apply_credential_selection(session_key, raw_args, choices, strategy)
        if not choices:
            return t("gateway.credentials.none")
        pin = session_credential_pin(self, session_key)
        pinned = next((c for c in choices if c.id == pin), None)
        current = pinned.display if pinned else t("gateway.credentials.current_automatic", strategy=strategy)

        async def _apply_picked(value: str) -> str:
            # Re-read the pool: the entry may have been removed while the buttons were up.
            try:
                fresh, fresh_strategy = await asyncio.to_thread(load_codex_choices)
            except Exception as exc:  # health: allow BLE001 -- reply instead of erroring; no traceback: pool load errors can carry auth payloads
                logger.warning("/credentials could not reload the %s pool: %s", CODEX_PROVIDER, type(exc).__name__)
                return t("gateway.credentials.load_failed")
            return await self._apply_credential_selection(session_key, value, fresh, fresh_strategy)

        async def _on_credential_choice(_chat_id: str, value: str) -> str:
            if profile_home is None:
                return await _apply_picked(value)
            from gateway.run import _profile_runtime_scope
            with _profile_runtime_scope(profile_home):
                return await _apply_picked(value)

        picker_sent = await self._try_send_choice_picker(
            event, session_key,
            title=t("gateway.credentials.picker_title", current=current),
            choices=[
                {"value": AUTOMATIC_CHOICE, "label": t("gateway.credentials.choice_automatic", strategy=strategy),
                 "is_current": pin is None},
                *({"value": c.id, "label": c.display, "is_current": c.id == pin} for c in choices),
            ],
            on_choice_selected=_on_credential_choice,
        )
        if picker_sent:
            return None  # Picker sent — adapter handles the response
        return self._credentials_text(choices, pin, current)
