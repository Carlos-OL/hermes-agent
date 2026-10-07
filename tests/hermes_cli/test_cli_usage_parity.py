"""Plain CLI /usage retains session totals and all account blocks."""
from datetime import datetime
from types import SimpleNamespace

from cli import HermesCLI
import pytest


@pytest.fixture
def cli_obj(monkeypatch):
    obj = HermesCLI.__new__(HermesCLI)
    obj.agent = None
    obj.model = "gpt-5.5"
    obj.provider = "openai-codex"
    obj.base_url = obj.api_key = None
    obj.session_start = datetime.now()
    obj.conversation_history = []
    obj.verbose = False
    obj._print_nous_credits_block = lambda: False
    monkeypatch.setattr("gateway.slash_commands_credentials.codex_pool_usage_lines", lambda: [])
    return obj


def test_limits_include_all_codex_accounts_on_other_provider(cli_obj, monkeypatch, capsys):
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
    cli_obj.provider = "anthropic"
    monkeypatch.setattr("agent.account_usage.fetch_account_usage", lambda *a, **kw: AccountUsageSnapshot(
        provider="anthropic", source="test", fetched_at=datetime.now(),
        windows=(AccountUsageWindow("Weekly", 25),)))
    monkeypatch.setattr("gateway.slash_commands_credentials.codex_pool_usage_lines", lambda: [
        "OpenAI Codex limits · Account A", "Weekly: 80% remaining", "",
        "OpenAI Codex limits · Account B", "Weekly: 90% remaining"])
    cli_obj._handle_usage_command("/usage")
    out = capsys.readouterr().out
    assert "Provider: anthropic" in out
    assert "Account A" in out and "Account B" in out
    assert "Weekly: 80% remaining\n  \n  OpenAI Codex limits" in out


def test_zero_calls_still_show_session_tokens_and_context(cli_obj, capsys):
    cli_obj.agent = SimpleNamespace(
        model=cli_obj.model, session_api_calls=0, session_input_tokens=0, session_output_tokens=0,
        session_prompt_tokens=0, session_completion_tokens=0, session_total_tokens=0,
        get_rate_limit_state=lambda: None,
        context_compressor=SimpleNamespace(last_prompt_tokens=500, context_length=272000, compression_count=0))
    cli_obj._handle_usage_command("/usage")
    out = capsys.readouterr().out
    assert "Total tokens" in out and "API calls" in out
    assert "500 / 272,000" in out


def test_resumed_no_agent_uses_persisted_totals(cli_obj, capsys):
    cli_obj.session_id = "resumed"
    cli_obj._session_db = SimpleNamespace(get_session=lambda sid: {
        "input_tokens": 1200, "output_tokens": 300, "cache_read_tokens": 4000,
        "cache_write_tokens": 200, "reasoning_tokens": 80, "api_call_count": 4})
    cli_obj._handle_usage_command("/usage")
    out = capsys.readouterr().out
    assert "Total tokens" in out and "5,700" in out
    assert "Input tokens" in out and "1,200" in out
    assert "API calls" in out


def test_fresh_no_agent_shows_zero_session_totals(cli_obj, capsys):
    cli_obj._handle_usage_command("/usage")
    out = capsys.readouterr().out
    assert "Total tokens" in out and "API calls" in out
