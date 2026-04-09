"""Tests for Shell command routing and REPL."""

import asyncio
import sys
from io import StringIO
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from tui_chatbot.shell import (
    Shell,
    Command,
    DefaultCommand,
    ModelCommand,
    ClearCommand,
    HelpCommand,
    QuitCommand,
    create_shell,
)
from tui_chatbot.config import Config
from tui_chatbot.daemon import Daemon
from tui_chatbot.frontend import Colors


# ╭────────────────────────────────────────────────────────────╮
# │  Fixtures                                                    │
# ╰────────────────────────────────────────────────────────────╯


@pytest.fixture
def config():
    """Create a test config."""
    return Config(api_key="test-key", model="gpt-3.5-turbo")


@pytest.fixture
def shell(config):
    """Create a test shell."""
    return Shell(config)


@pytest.fixture
def mock_daemon():
    """Create a mock daemon."""
    daemon = MagicMock(spec=Daemon)
    daemon.model = "gpt-3.5-turbo"
    daemon.list_models = AsyncMock(return_value=["gpt-3.5-turbo", "gpt-4"])
    return daemon


# ╭────────────────────────────────────────────────────────────╮
# │  Shell Init Tests                                            │
# ╰────────────────────────────────────────────────────────────╯


class TestShellInit:
    """Shell initialization tests."""

    def test_init(self, config):
        """Shell initializes with config."""
        shell = Shell(config)

        assert shell.config == config
        assert shell.daemon is not None
        assert shell.PROMPT == ">>> "

    def test_init_registers_commands(self, config):
        """Shell registers all commands on init."""
        shell = Shell(config)

        assert "__default__" in shell.cmds
        assert "model" in shell.cmds
        assert "m" in shell.cmds
        assert "clear" in shell.cmds
        assert "c" in shell.cmds
        assert "help" in shell.cmds
        assert "h" in shell.cmds
        assert "quit" in shell.cmds
        assert "q" in shell.cmds
        assert "exit" in shell.cmds


class TestCreateShell:
    """create_shell factory tests."""

    def test_create_shell(self, config):
        """create_shell returns configured Shell."""
        shell = create_shell(config)
        assert isinstance(shell, Shell)
        assert shell.config == config


# ╭────────────────────────────────────────────────────────────╮
# │  Command Parsing Tests                                       │
# ╰────────────────────────────────────────────────────────────╯


class TestShellGetCmd:
    """Shell command parsing tests."""

    def test_get_cmd_default_chat(self, shell):
        """Non-command input routes to default chat."""
        cmd, argv = shell._get_cmd("Hello bot")

        assert isinstance(cmd, DefaultCommand)
        assert argv == ["Hello bot"]

    def test_get_cmd_chat_with_slash(self, shell):
        """Text starting with / but unknown command routes to chat."""
        cmd, argv = shell._get_cmd("/notacommand test")

        assert isinstance(cmd, DefaultCommand)
        assert argv == ["/notacommand test"]

    def test_get_cmd_model_list(self, shell):
        """/model routes to ModelCommand."""
        cmd, argv = shell._get_cmd("/model")

        assert isinstance(cmd, ModelCommand)
        assert argv == ["/model"]

    def test_get_cmd_model_switch(self, shell):
        """/model <name> routes to ModelCommand with args."""
        cmd, argv = shell._get_cmd("/model gpt-4")

        assert isinstance(cmd, ModelCommand)
        assert argv == ["/model", "gpt-4"]

    def test_get_cmd_model_alias(self, shell):
        """/m alias routes to ModelCommand."""
        cmd, argv = shell._get_cmd("/m gpt-4")

        assert isinstance(cmd, ModelCommand)
        assert argv == ["/m", "gpt-4"]

    def test_get_cmd_clear(self, shell):
        """/clear routes to ClearCommand."""
        cmd, argv = shell._get_cmd("/clear")

        assert isinstance(cmd, ClearCommand)
        assert argv == ["/clear"]

    def test_get_cmd_clear_alias(self, shell):
        """/c alias routes to ClearCommand."""
        cmd, argv = shell._get_cmd("/c")

        assert isinstance(cmd, ClearCommand)
        assert argv == ["/c"]

    def test_get_cmd_help(self, shell):
        """/help routes to HelpCommand."""
        cmd, argv = shell._get_cmd("/help")

        assert isinstance(cmd, HelpCommand)
        assert argv == ["/help"]

    def test_get_cmd_help_alias(self, shell):
        """/h alias routes to HelpCommand."""
        cmd, argv = shell._get_cmd("/h")

        assert isinstance(cmd, HelpCommand)
        assert argv == ["/h"]

    def test_get_cmd_quit(self, shell):
        """/quit routes to QuitCommand."""
        cmd, argv = shell._get_cmd("/quit")

        assert isinstance(cmd, QuitCommand)
        assert argv == ["/quit"]

    def test_get_cmd_quit_alias_q(self, shell):
        """/q alias routes to QuitCommand."""
        cmd, argv = shell._get_cmd("/q")

        assert isinstance(cmd, QuitCommand)
        assert argv == ["/q"]

    def test_get_cmd_exit(self, shell):
        """/exit routes to QuitCommand."""
        cmd, argv = shell._get_cmd("/exit")

        assert isinstance(cmd, QuitCommand)
        assert argv == ["/exit"]

    def test_get_cmd_case_insensitive(self, shell):
        """Command names are case insensitive."""
        cmd, argv = shell._get_cmd("/MODEL gpt-4")

        assert isinstance(cmd, ModelCommand)

    def test_get_cmd_shlex_parsing(self, shell):
        """Shlex parsing for structured commands."""
        cmd, argv = shell._get_cmd('/model "model name with spaces"')

        assert isinstance(cmd, ModelCommand)
        assert argv == ["/model", "model name with spaces"]

    def test_get_cmd_raw_mode(self, shell):
        """Raw mode keeps entire text."""
        cmd, argv = shell._get_cmd("Hello world")

        assert isinstance(cmd, DefaultCommand)
        assert argv == ["Hello world"]


# ╭────────────────────────────────────────────────────────────╮
# │  Command Tests                                               │
# ╰────────────────────────────────────────────────────────────╯


class TestDefaultCommand:
    """DefaultCommand (chat) tests."""

    @pytest.mark.asyncio
    async def test_run_chat(self, mock_daemon):
        """DefaultCommand runs chat via frontend."""
        from tui_chatbot.frontend import Frontend

        cmd = DefaultCommand(mock_daemon)

        # Mock frontend
        with patch.object(Frontend, "run", AsyncMock()) as mock_run:
            result = await cmd.run(["Hello"])

            assert result == 0
            mock_run.assert_called_once_with(["Hello"])


class TestModelCommand:
    """ModelCommand tests."""

    @pytest.mark.asyncio
    async def test_run_list_models(self, mock_daemon, capsys):
        """ModelCommand lists models when no argument."""
        cmd = ModelCommand(mock_daemon)

        result = await cmd.run(["/model"])

        assert result == 0
        mock_daemon.list_models.assert_called_once()

        captured = capsys.readouterr()
        assert "Models" in captured.out
        assert "gpt-3.5-turbo" in captured.out

    @pytest.mark.asyncio
    async def test_run_switch_model(self, mock_daemon, capsys):
        """ModelCommand switches model when name provided."""
        cmd = ModelCommand(mock_daemon)

        result = await cmd.run(["/model", "gpt-4"])

        assert result == 0
        mock_daemon.switch_model.assert_called_once_with("gpt-4")

        captured = capsys.readouterr()
        assert "Switched to: gpt-4" in captured.out


class TestClearCommand:
    """ClearCommand tests."""

    @pytest.mark.asyncio
    async def test_run_clear(self, mock_daemon, capsys):
        """ClearCommand clears history."""
        cmd = ClearCommand(mock_daemon)

        result = await cmd.run(["/clear"])

        assert result == 0
        mock_daemon.clear.assert_called_once()

        captured = capsys.readouterr()
        assert "cleared" in captured.out


class TestHelpCommand:
    """HelpCommand tests."""

    @pytest.mark.asyncio
    async def test_run_help(self, capsys):
        """HelpCommand displays help."""
        cmd = HelpCommand()

        result = await cmd.run(["/help"])

        assert result == 0

        captured = capsys.readouterr()
        assert "Commands:" in captured.out
        assert "/model" in captured.out
        assert "/clear" in captured.out
        assert "/help" in captured.out
        assert "/quit" in captured.out


class TestQuitCommand:
    """QuitCommand tests."""

    @pytest.mark.asyncio
    async def test_run_quit(self, capsys):
        """QuitCommand exits the program."""
        cmd = QuitCommand()

        with pytest.raises(SystemExit) as exc_info:
            await cmd.run(["/quit"])

        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "Bye!" in captured.out


# ╭────────────────────────────────────────────────────────────╮
# │  Shell Run Tests                                             │
# ╰────────────────────────────────────────────────────────────╯


class TestShellRun:
    """Shell REPL tests."""

    @pytest.mark.asyncio
    async def test_run_command(self, shell):
        """run_command executes a single command."""
        with patch("tui_chatbot.frontend.Frontend.run", AsyncMock()) as mock_run:
            result = await shell.run_command("Hello")

            assert result == 0
            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_command_with_command(self, shell):
        """run_command handles /commands."""
        with patch.object(ModelCommand, "run", AsyncMock(return_value=0)) as mock_run:
            result = await shell.run_command("/model")

            assert result == 0
            mock_run.assert_called_once()


# ╭────────────────────────────────────────────────────────────╮
# │  Command Base Tests                                          │
# ╰────────────────────────────────────────────────────────────╯


class TestCommandBase:
    """Command base class tests."""

    def test_command_meta(self):
        """Command has META dict."""
        assert Command.META == {"pass_mode": "shlex"}

    def test_default_command_meta(self):
        """DefaultCommand has raw pass mode."""
        from tui_chatbot.daemon import Daemon

        cmd = DefaultCommand(MagicMock(spec=Daemon))
        assert cmd.META == {"pass_mode": "raw"}

    def test_command_run_not_implemented(self):
        """Command.run raises NotImplementedError."""
        cmd = Command()

        with pytest.raises(NotImplementedError):
            asyncio.run(cmd.run([]))
