"""``session.usage`` RPC (Desktop usage feed) carries the provider account-limits block.

The CLI/TUI slash worker and gateway ``/usage`` render Codex quota windows via
``render_account_usage_lines``; the Desktop feed reads ``session.usage`` instead, so the RPC
must ship the same lines (``account_lines``) or that surface silently omits them.
"""
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import patch


def test_session_usage_rpc_ships_account_lines_for_every_codex_pool_entry():
    from tui_gateway import server

    agent = SimpleNamespace(provider="openai-codex", base_url="https://chatgpt.example/backend-api",
                            api_key="tok", model="gpt-5.3-codex")
    session = {"agent": agent, "history": [], "running": False, "session_key": "sess-usage"}
    sid = "sid-usage-account"
    server._sessions[sid] = session
    account_lines = [
        "📈 OpenAI Codex limits · Personal · abc123",
        "Weekly: 12% used",
        "",
        "📈 OpenAI Codex limits · Work · def456",
        "Weekly: 34% used",
    ]

    try:
        with (
            patch.object(server, "_get_usage", return_value={"calls": 1, "input": 10, "output": 20, "total": 30}),
            patch("gateway.slash_commands_credentials.codex_pool_usage_lines", return_value=account_lines),
            patch("agent.account_usage.nous_credits_lines", lambda **kw: []),
        ):
            r = server._methods["session.usage"]("r1", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert "error" not in r, r
    result = r["result"]
    assert result["total"] == 30 and "credits_lines" not in result
    assert result["account_lines"] == account_lines


def test_session_usage_rpc_keeps_active_non_codex_limits_before_codex_pool():
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
    from tui_gateway import server

    agent = SimpleNamespace(provider="anthropic", base_url="https://api.anthropic.com", api_key="tok")
    session = {"agent": agent, "history": [], "running": False, "session_key": "sess-anthropic"}
    snapshot = AccountUsageSnapshot(
        provider="anthropic", source="test", fetched_at=None,
        windows=(AccountUsageWindow(label="Current week", used_percent=20),),
    )

    with (
        patch("agent.account_usage.fetch_account_usage", return_value=snapshot),
        patch("gateway.slash_commands_credentials.codex_pool_usage_lines", return_value=["Codex pool"]),
    ):
        lines = server._account_usage_lines(session)

    assert "Provider: anthropic" in "\n".join(lines)
    assert lines[-2:] == ["", "Codex pool"]


def test_codex_pool_usage_lines_preserves_pool_order_and_omits_failed_accounts():
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
    from gateway import slash_commands_credentials as credentials

    targets = [
        credentials.CredentialUsageTarget(
            "Personal · abc123", "secret-one", "https://one.example", "fp:one"),
        credentials.CredentialUsageTarget(
            "Broken · bad999", "secret-bad", "https://bad.example", "fp:bad"),
        credentials.CredentialUsageTarget(
            "Work · def456", "secret-two", "https://two.example", "fp:two"),
    ]
    profile_marker = ContextVar("profile_marker", default="wrong-profile")
    profile_marker.set("session-profile")

    def _fetch(_provider, *, base_url=None, api_key=None, read_only=False, identity_id=None):
        assert read_only is True
        assert identity_id is not None
        assert profile_marker.get() == "session-profile"
        if api_key == "secret-bad":
            raise RuntimeError("quota endpoint unavailable")
        used = 12 if api_key == "secret-one" else 34
        return AccountUsageSnapshot(
            provider="openai-codex", source="test", fetched_at=None,
            windows=(AccountUsageWindow(label="Weekly", used_percent=used),),
        )

    with (
        patch.object(credentials, "load_codex_usage_targets", return_value=targets),
        patch("agent.account_usage.fetch_account_usage", side_effect=_fetch),
    ):
        lines = credentials.codex_pool_usage_lines()

    output = "\n".join(lines)
    assert output.index("Personal · abc123") < output.index("Work · def456")
    assert "Broken · bad999" not in output
    assert "secret-one" not in output and "secret-two" not in output and "secret-bad" not in output


def test_tui_usage_combines_session_tokens_with_all_subscription_limits():
    from tui_gateway import server

    session = {
        "agent": SimpleNamespace(model="gpt-5.5"),
        "history": [],
        "running": False,
        "session_key": "sess-usage",
        "_metadata_message_count": 2,
        "usage": {"calls": 99, "input": 123_456, "output": 7_890, "total": 131_346},
    }
    account_lines = [
        "📈 OpenAI Codex limits · Personal · abc123",
        "Weekly: 12% used",
        "",
        "📈 OpenAI Codex limits · Work · def456",
        "Weekly: 34% used",
    ]
    credits_lines = ["💳 Nous credits", "Balance: 42.00 credits"]

    with (
        patch.object(server, "_session_usage_snapshot", return_value=session["usage"]),
        patch("gateway.slash_commands_credentials.codex_pool_usage_lines", return_value=account_lines),
        patch("agent.account_usage.nous_credits_lines", return_value=credits_lines),
    ):
        output = server._format_live_usage_output("sid", session, "")

    assert output.startswith("Session Token Usage")
    assert "123,456" in output
    assert output.endswith("\n".join([*account_lines, "", *credits_lines]))


def test_tui_usage_without_agent_still_shows_zero_session_and_subscription_blocks():
    from tui_gateway import server

    session = {
        "agent": None,
        "history": [],
        "running": False,
        "session_key": "sess-empty",
        "_metadata_message_count": 0,
    }
    with (
        patch.object(server, "_session_usage_snapshot", return_value={}),
        patch.object(server, "_account_usage_lines", return_value=["Codex pool"]),
        patch("agent.account_usage.nous_credits_lines", return_value=["Nous credits"]),
    ):
        output = server._format_live_usage_output("sid", session, "")

    assert output.startswith("Session Token Usage")
    assert "Total tokens:" in output and "0" in output
    assert output.endswith("Codex pool\n\nNous credits")
