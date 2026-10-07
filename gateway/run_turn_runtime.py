"""Session model/runtime resolution for gateway turns (``/model`` override, channel overrides,
config default) plus the ``/credentials`` pool pin layered over it."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Optional

from agent.credential_pool_admin import preferred_pool_entry
from gateway.session import SessionSource
from gateway.slash_commands_credentials import (
    CODEX_PROVIDER, apply_session_credential_pin, session_credential_pin,
)

logger = logging.getLogger("gateway.run")


class GatewayTurnRuntimeMixin:
    """``_resolve_session_agent_runtime`` for GatewayRunner (via GatewayTurnMixin)."""

    def _resolve_session_agent_runtime(
        self, *, source: Optional[SessionSource] = None, session_key: Optional[str] = None,
        user_config: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Resolve model/runtime for a session, on its ``/credentials`` pool entry when pinned.

        The pin is a pool preference for the whole resolution: every ``select()`` on the
        openai-codex pool serves (and is charged to) the pinned entry while it is available, so the
        strategy never rotates or counts an entry that is not going to serve. Runtimes resolved
        without a pool selection (the ``/model`` fast path) are moved onto the pin afterwards.

        Args:
            source (SessionSource | None): Message source (derives the session key when not given).
            session_key (str | None): Session key.
            user_config (dict | None): Loaded user config.

        Returns:
            tuple: ``(model, runtime_kwargs)``.

        Calls:
            - _resolve_session_agent_runtime_unpinned() - the resolution ladder
            - apply_session_credential_pin() - swap onto the pin when resolution did not select it
        """
        skey = self._resolve_session_key_or_none(source, session_key)
        with preferred_pool_entry(CODEX_PROVIDER, session_credential_pin(self, skey)):
            model, runtime_kwargs = self._resolve_session_agent_runtime_unpinned(
                source=source, session_key=skey, user_config=user_config)
        return model, apply_session_credential_pin(self, skey, model, runtime_kwargs)

    def _resolve_session_agent_runtime_unpinned(
        self, *, source: Optional[SessionSource] = None, session_key: Optional[str] = None,
        user_config: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Resolve model/runtime for a session.

        Priority (highest first): session ``/model`` → ``channel_overrides`` → global config/env
        (``_resolve_gateway_model(user_config)`` and default provider resolution)."""
        from gateway.run import (
            _credential_pool_for_provider, _get_channel_override, _resolve_gateway_model,
            _resolve_runtime_agent_kwargs, _resolve_runtime_agent_kwargs_for_provider,
        )
        skey = self._resolve_session_key_or_none(source, session_key)
        # Every exit path starts clean: the /model-override fast path returns before the pop below,
        # and hygiene/inbound callers resolve without a turn runner consuming the stash — a stale
        # notice must never attach to another session's next turn (#74349).
        self._pre_agent_fallback_notice = None

        model = _resolve_gateway_model(user_config)
        if skey:
            self._rehydrate_session_model_override(skey)
        _override_state = self._peek_session_state(skey) if skey else None
        override = _override_state.conversation.model_override if _override_state else None
        if override:
            override_model = override.get("model", model)
            override_runtime = {
                k: override.get(k) for k in (
                    "provider", "requested_provider", "api_key", "base_url", "api_mode",
                    "max_tokens", "credential_pool", "request_overrides", "capabilities",
                )
            }
            override_runtime["capabilities"] = dict(override_runtime["capabilities"] or {})
            if override_runtime.get("api_key"):
                if override_runtime.get("credential_pool") is None:
                    override_runtime["credential_pool"] = _credential_pool_for_provider(override.get("provider"))
                logger.debug(
                    "Session model override (fast): session=%s config_model=%s -> override_model=%s provider=%s",
                    skey or "", model, override_model, override_runtime.get("provider"),
                )
                return override_model, override_runtime
            # No api_key on the override (credentials failed to re-resolve at rehydrate): resolve them
            # for the override's own provider below, never layer it over the default provider's runtime.
            logger.debug(
                "Session model override (no api_key, fallback): session=%s config_model=%s override_model=%s",
                skey or "", model, override_model,
            )
        elif logger.isEnabledFor(logging.DEBUG):
            # The override_keys scan walks every session; only pay for it when DEBUG is on.
            logger.debug(
                "No session model override: session=%s config_model=%s override_keys=%s",
                skey or "", model,
                [
                    _key for _key, _st in list(self._sessions_map().items())
                    if _st.conversation.model_override is not None
                ][:5] or "[]",
            )

        runtime_kwargs, unavailable_override = None, None
        if override and override.get("provider"):
            try:
                runtime_kwargs = _resolve_runtime_agent_kwargs_for_provider(
                    override["provider"], target_model=override.get("model") or None)
            except Exception as exc:
                # Layering the override on the default runtime sent its model to the default provider's
                # endpoint (openai-codex on the Nous URL). Run this turn on the whole default route and say
                # so; the persisted override is kept, so the next turn retries it.
                logger.warning("Session /model override provider %s unavailable: %s", override["provider"], exc)
                unavailable_override, override = override, None
        if runtime_kwargs is None:
            runtime_kwargs = _resolve_runtime_agent_kwargs()
        # Private notice metadata must never reach an ``AIAgent(**runtime_kwargs)`` spread; the turn
        # runner surfaces it through the agent's one-shot fallback notice (#74349).
        self._pre_agent_fallback_notice = runtime_kwargs.pop("_fallback_notice", None)
        runtime_model = runtime_kwargs.pop("model", None)
        if runtime_model:
            logger.info("Runtime provider supplied explicit model override: %s -> %s", model, runtime_model)
            model = runtime_model
        if unavailable_override and not self._pre_agent_fallback_notice:
            from hermes_cli.fallback_config import pre_agent_fallback_notice
            self._pre_agent_fallback_notice = pre_agent_fallback_notice(
                unavailable_override["provider"], unavailable_override.get("model"), runtime_kwargs.get("provider"), model)

        cfg = getattr(self, "config", None)  # getattr: bare object.__new__ test runners
        if cfg and source is not None:
            ch = _get_channel_override(
                cfg, source.platform, str(source.chat_id) if source.chat_id else "",
                thread_id=str(source.thread_id) if getattr(source, "thread_id", None) else None,
                parent_id=str(source.parent_chat_id) if getattr(source, "parent_chat_id", None) else None,
            )
            if ch:
                if ch.model:
                    model = ch.model
                if ch.provider:
                    runtime_kwargs = _resolve_runtime_agent_kwargs_for_provider(ch.provider, target_model=model or None)
                    ch_runtime_model = runtime_kwargs.pop("model", None)
                    # Adopt the provider's bundled model only when the override named none.
                    if ch_runtime_model and not ch.model:
                        model = ch_runtime_model

        if override and skey:
            model, runtime_kwargs = self._apply_session_model_override(skey, model, runtime_kwargs)

        # Provider resolved but no model.default (`hermes auth add` without `hermes model`): use the
        # provider's first catalog model.
        if not model and runtime_kwargs.get("provider"):
            with suppress(Exception):
                from hermes_cli.models import get_default_model_for_provider
                model = get_default_model_for_provider(runtime_kwargs["provider"])
                if model:
                    logger.info(
                        "No model configured — defaulting to %s for provider %s", model, runtime_kwargs["provider"],
                    )

        # Final safety net: an empty model (transient config-cache miss) makes every API call 400 and
        # the session goes silent — reuse the last model resolved for this session, else process-wide.
        if not model:
            _lr_state = self._peek_session_state(skey) if skey else None
            _lr_star = self._peek_session_state("*")
            _recovered = (
                (_lr_state.conversation.last_resolved_model if _lr_state else "")
                or (_lr_star.conversation.last_resolved_model if _lr_star else "")
            )
            if _recovered:
                logger.warning(
                    "Empty model resolved for session=%s — recovering "
                    "last-known-good model %s (config read likely returned "
                    "empty; see #35314)", skey or "", _recovered,
                )
                model = _recovered
        else:
            # Cache the good resolution for future recovery turns.
            if skey:
                self._session_state(skey).conversation.last_resolved_model = model
            self._session_state("*").conversation.last_resolved_model = model

        return model, runtime_kwargs
