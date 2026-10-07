"""Locked credential-pool administration and target resolution."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Iterator, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.credential_pool import PooledCredential

logger = logging.getLogger(__name__)

def _cleared_status_copy(entry: PooledCredential) -> PooledCredential:
    from agent.credential_pool import _CLEAR_STATUS

    # The reset marker lets a live pool in another process tell "reset after my cooldown" from
    # "never had a status" — both read as bare None on disk (#89415).
    return replace(entry, **_CLEAR_STATUS, model_cooldowns=None, status_cleared_at=time.time(),
                   extra={k: v for k, v in entry.extra.items() if k != "failure_reason"})


# (provider, entry id) that ``select`` serves first inside a ``preferred_pool_entry`` scope.
_PREFERRED_ENTRY: ContextVar[Optional[Tuple[str, str]]] = ContextVar("credential_pool_preferred_entry", default=None)


@contextmanager
def preferred_pool_entry(provider: str, credential_id: Optional[str]) -> Iterator[None]:
    """Within the block, ``select()`` on *provider*'s pools serves *credential_id* whenever it is
    available (no strategy rotation, the request charged to that entry); when it is benched or
    gone, ``select()`` runs the strategy as usual. ``credential_id=None`` is a no-op scope.

    Args:
        provider (str): Pool provider the preference applies to (e.g. ``openai-codex``).
        credential_id (str | None): Stable pool entry id (never a token).

    Called by:
        - gateway.run_turn.GatewayTurnMixin._resolve_session_agent_runtime() - /credentials pin
    """
    token = _PREFERRED_ENTRY.set((provider, credential_id) if credential_id else None)
    try:
        yield
    finally:
        _PREFERRED_ENTRY.reset(token)


def agent_pool_pin(agent: Any) -> Tuple[str, Optional[str]]:
    """``(provider, entry id)`` of the session pin (``agent._credential_pool_pin``, set per turn by
    the gateway's ``/credentials``) when it applies to the agent's live pool, else ``("", None)``.

    Called by:
        - return_to_pinned_entry()
        - agent.agent_runtime_helpers.restore_primary_runtime() - post-fallback pool re-select
    """
    pin = getattr(agent, "_credential_pool_pin", None)
    pool = getattr(agent, "_credential_pool", None)
    if not pin or pool is None or str(getattr(pool, "provider", "") or "").strip().lower() != pin[0]:
        return "", None
    return pin[0], pin[1]


def return_to_pinned_entry(agent: Any) -> bool:
    """Put a live (e.g. gateway-cached) agent back on its session's pinned pool entry at turn start.

    A 429 on the pin rotates the agent within the pool and automatic selection serves while the pin
    cools down. The gateway's cached-agent signature hashes the key the agent was BUILT with, so once
    the pin is eligible again the resolved runtime matches the cache and the agent is reused while
    its client is still on the rotated entry; this swaps it back (``select_pinned``: cooldown
    clearing and token refresh under the pool/auth-store locks, ``current`` pointed at the pin).
    While the agent is on its pin, rotation reverts armed by earlier benches are dropped.

    Args:
        agent (AIAgent): Agent whose ``_credential_pool`` / ``_credential_pool_entry_id`` are live.

    Returns:
        bool: True when the agent is on its pin (already, or swapped now); False when no pin
        applies or the pin is still cooling down (the caller's automatic revert logic then runs).

    Called by:
        - agent.agent_runtime_helpers._revert_credential_rotation() - every turn start

    Calls:
        - CredentialPool.select_pinned() - refresh + make the pin ``current``
    """
    _provider, pin = agent_pool_pin(agent)
    if pin is None:
        return False
    if getattr(agent, "_credential_pool_entry_id", None) != pin:
        try:
            entry = agent._credential_pool.select_pinned(pin, model=getattr(agent, "model", None) or None)
        except Exception as exc:  # health: allow BLE001 -- no traceback: refresh errors can carry auth payloads
            logger.warning("Pinned credential %s could not be restored (%s)", pin[:6], type(exc).__name__)
            return False
        if entry is None or agent._swap_credential(entry) is False:
            return False  # still cooling down (or its route cannot serve this model): automatic serves
        logger.info("Pinned credential %s available again — session back on it", pin[:6])
    agent._credential_pool_revert_id = None
    return True


class CredentialPoolAdminMixin:
    def reset_status(self, credential_id: str) -> Optional[PooledCredential]:
        """Clear only the target's local error state, preserving sibling cooldowns."""
        with self._lock:
            entry = self._find(lambda e: e.id == credential_id)
            if entry is None:
                return None
            cleared = _cleared_status_copy(entry)
            self._replace_entry(entry, cleared)
            self._persist(status_cleared_ids=[cleared.id])
            return cleared
    def reset_statuses(self) -> int:
        """Clear exhaustion state on every entry. Returns how many were cleared.

        ``failure_reason`` lives in ``extra``, not a dataclass field, so it is
        stripped explicitly. The persist declares the cleared ids because the
        disk-recency merge reads a cleared ``last_status_at`` (None -> epoch 0)
        as a stale snapshot and would copy a still-binding cooldown back.
        """
        from agent.credential_pool import _CLEAR_STATUS

        with self._lock:
            stale = [
                e for e in self._entries
                if e.last_status or e.last_status_at or e.last_error_code or e.failure_reason or e.model_cooldowns
            ]
            if stale:
                stale_ids = {e.id for e in stale}
                self._entries = [
                    _cleared_status_copy(e) if e.id in stale_ids else e
                    for e in self._entries
                ]
                self._persist(status_cleared_ids=list(stale_ids))
            return len(stale)

    def reclaim(self, credential_id: str, *, model: Optional[str] = None) -> Optional[PooledCredential]:
        """Entry *credential_id* once its cooldown has lifted (cleared and token-refreshed the way
        ``select`` would), else ``None``. Never bumps ``request_count`` or round-robin order: a
        live session asking "may I go back?" every turn is not a request."""
        with self._lock:
            available, pending = self._available_entries(clear_expired=True, refresh=True, model=model)
        if any(e.id == credential_id for e in pending):
            self._refresh_pending_entries([e for e in pending if e.id == credential_id])
            with self._lock:
                available, _pending = self._available_entries(clear_expired=True, refresh=True, model=model)
        return next((e for e in available if e.id == credential_id), None)

    def select_pinned(
        self, credential_id: str, *, model: Optional[str] = None, count: bool = False,
    ) -> Optional[PooledCredential]:
        """Make *credential_id* the pool's current entry for a session pin (gateway ``/credentials``).

        Cleared and token-refreshed exactly as ``reclaim`` does (pool lock + auth-store lock), so a
        pinned turn never reads a stale token; ``None`` while the entry is benched or gone. The
        strategy's order is never touched (no round-robin rotation, no other entry charged);
        pointing ``current`` at the pin keeps 429 recovery (``mark_exhausted_and_rotate``) and
        ``entry_id_for_api_key`` attributed to the pinned row.

        Args:
            credential_id (str): Stable pool entry id (never a token).
            model (str | None): Model for per-model cooldown checks.
            count (bool): Charge the selection to the pinned entry's ``request_count``, as
                ``select`` does for the entry it picks (True when the selection serves a request).

        Returns:
            PooledCredential | None: The refreshed pinned entry, or None when unavailable.

        Called by:
            - _select_preferred() - ``select`` inside a ``preferred_pool_entry`` scope (count=True)
            - gateway.slash_commands_credentials.apply_session_credential_pin() - runtimes that were
              resolved without a pool selection (count=False)

        Calls:
            - reclaim() - cooldown clearing + refresh under the normal locks
        """
        entry = self.reclaim(credential_id, model=model)
        if entry is None:
            return None
        with self._lock:
            entry = self._find(lambda e: e.id == credential_id) or entry
            if count:
                entry = self._adopt(entry, persist=False, request_count=entry.request_count + 1)
            self._current_id = entry.id
            return entry

    def _select_preferred(self, *, model: Optional[str] = None) -> Optional[PooledCredential]:
        """The ``preferred_pool_entry`` scope's entry for this pool's provider, if one is set and
        available (selected via ``select_pinned``, charged to that entry), else None.

        Called by:
            - CredentialPool.select() - before the strategy runs
        """
        preferred = _PREFERRED_ENTRY.get()
        if preferred is None or preferred[0] != self.provider:
            return None
        return self.select_pinned(preferred[1], model=model, count=True)

    def fresh_entry(self, credential_id: str) -> Optional[PooledCredential]:
        """Entry *credential_id* regardless of cooldown, its token refreshed first when it is
        expiring (same locks as ``select``). ``None`` when the entry is gone or the refresh failed.
        Never changes ``current``, ``request_count`` or the strategy's order.

        Called by:
            - gateway.slash_commands_credentials.session_credential_usage_view() - /usage for a
              pinned entry that is cooling down
        """
        with self._lock:
            entry = self._find(lambda e: e.id == credential_id)
        if entry is None or not self._entry_needs_refresh(entry):
            return entry
        return self._refresh_entry(entry, force=False)

    def remove_index(self, index: int) -> Optional[PooledCredential]:
        with self._lock:
            if index < 1 or index > len(self._entries):
                return None
            removed = self._entries.pop(index - 1)
            self._entries = [replace(e, priority=p) for p, e in enumerate(self._entries)]
            self._persist(removed_ids=[removed.id])
            if self._current_id == removed.id:
                self._current_id = None
            return removed

    def move_entry(self, credential_id: str, priority: int) -> Optional[PooledCredential]:
        """Place an entry at a clamped zero-based position and persist contiguous priorities."""
        from agent.credential_pool import _normalize_pool_priorities

        with self._lock:
            entry = self._find(lambda e: e.id == credential_id)
            if entry is None:
                return None
            others = [e for e in self._entries if e.id != credential_id]
            others.insert(max(0, min(int(priority), len(others))), entry)
            entries = [replace(e, priority=p) for p, e in enumerate(others)]
            # Apply load-time ordering now so the reported position survives reload.
            _normalize_pool_priorities(self.provider, entries)
            self._entries = sorted(entries, key=lambda e: e.priority)
            self._persist()
            return self._find(lambda e: e.id == credential_id)

    def resolve_target(self, target: Any) -> Tuple[Optional[int], Optional[PooledCredential], Optional[str]]:
        raw = str(target or "").strip()
        if not raw:
            return None, None, "No credential target provided."

        with self._lock:
            for idx, entry in enumerate(self._entries, start=1):
                if entry.id == raw:
                    return idx, entry, None

            label_matches = [
                (idx, entry)
                for idx, entry in enumerate(self._entries, start=1)
                if entry.label.strip().lower() == raw.lower()
            ]
            if len(label_matches) == 1:
                return label_matches[0][0], label_matches[0][1], None
            if len(label_matches) > 1:
                return None, None, f'Ambiguous credential label "{raw}". Use the numeric index or entry id instead.'
            if raw.isdigit():
                index = int(raw)
                if 1 <= index <= len(self._entries):
                    return index, self._entries[index - 1], None
                return None, None, f"No credential #{index}."
            return None, None, f'No credential matching "{raw}".'

    def add_entry(self, entry: PooledCredential) -> PooledCredential:
        from agent.credential_pool import _next_priority, write_credential_pool
        from hermes_cli import auth as auth_mod

        with self._lock:
            entry = replace(entry, priority=_next_priority(self._entries))
            self._entries.append(entry)
            borrowed_ids = getattr(self, "_borrowed_root_ids", None)
            if borrowed_ids:
                # ``hermes -p <profile> auth add <single-use provider>``: the
                # profile claims its OWN credential. Persist only profile-owned
                # rows — copying the borrowed root grant alongside would fork
                # its single-use refresh token (#100339). Once the profile owns
                # rows, the root fallback for this provider is shadowed.
                self._entries = [e for e in self._entries if e.id not in borrowed_ids]
                written = write_credential_pool(
                    self.provider, [e.to_dict() for e in self._entries],
                    token_bases=self._persisted_token_pairs,
                )
                self._persisted_token_pairs = auth_mod._token_pairs_by_id(written)
                self._borrowed_root_ids = set()
            else:
                self._persist()
            return entry
