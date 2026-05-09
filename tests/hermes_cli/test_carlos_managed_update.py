"""Carlos-specific update routing guards."""

from types import SimpleNamespace

import pytest


def test_terminal_update_routes_to_carlos_managed_script(monkeypatch, tmp_path):
    import hermes_cli.main as main

    script = tmp_path / "hermes-carlos-update.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(main, "_carlos_managed_update_script", lambda: script)
    monkeypatch.setattr(main.subprocess, "run", lambda cmd: calls.append(cmd) or SimpleNamespace(returncode=0))

    handled = main._run_carlos_managed_update(SimpleNamespace(check=False, gateway=False))

    assert handled is True
    assert calls == [[main.sys.executable, str(script), "apply"]]


def test_terminal_update_check_routes_to_carlos_dry_run(monkeypatch, tmp_path):
    import hermes_cli.main as main

    script = tmp_path / "hermes-carlos-update.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(main, "_carlos_managed_update_script", lambda: script)
    monkeypatch.setattr(main.subprocess, "run", lambda cmd: calls.append(cmd) or SimpleNamespace(returncode=0))

    handled = main._run_carlos_managed_update(SimpleNamespace(check=True, gateway=True))

    assert handled is True
    assert calls == [[main.sys.executable, str(script), "dry-run", "--gateway"]]


def test_carlos_managed_update_failure_exits(monkeypatch, tmp_path):
    import hermes_cli.main as main

    script = tmp_path / "hermes-carlos-update.py"
    script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    monkeypatch.setattr(main, "_carlos_managed_update_script", lambda: script)
    monkeypatch.setattr(main.subprocess, "run", lambda cmd: SimpleNamespace(returncode=7))

    with pytest.raises(SystemExit) as exc:
        main._run_carlos_managed_update(SimpleNamespace(check=False, gateway=False))

    assert exc.value.code == 7


def test_update_slash_command_is_available_in_cli():
    from hermes_cli.commands import resolve_command

    cmd = resolve_command("update")

    assert cmd is not None
    assert cmd.name == "update"
    assert cmd.gateway_only is False


def test_cli_slash_update_invokes_update_subcommand(monkeypatch):
    import subprocess
    import cli as cli_module

    rendered = []
    calls = []
    cli_obj = cli_module.HermesCLI.__new__(cli_module.HermesCLI)

    monkeypatch.setattr(cli_module, "_cprint", lambda msg: rendered.append(str(msg)))
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda cmd: calls.append(cmd) or SimpleNamespace(returncode=0),
    )

    assert cli_module.HermesCLI.process_command(cli_obj, "/update") is True

    assert calls == [[cli_module.sys.executable, "-m", "hermes_cli.main", "update"]]
    assert any("Update command completed" in line for line in rendered)
