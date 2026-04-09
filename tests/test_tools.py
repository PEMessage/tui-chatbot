"""Tests for built-in agent tools.

Tests the echo_tool, calculator_tool, and datetime_tool implementations.
"""

import asyncio
import json
import re
from datetime import datetime

import pytest

from tui_chatbot.agent.tools import calculator_tool, datetime_tool, echo_tool
from tui_chatbot.agent.types import AgentTool


# ╭────────────────────────────────────────────────────────────╮
# │  Echo Tool Tests                                             │
# ╰────────────────────────────────────────────────────────────╯


def test_echo_tool_structure():
    """Test echo_tool is correctly structured."""
    assert isinstance(echo_tool, AgentTool)
    assert echo_tool.name == "echo"
    assert "echo" in echo_tool.description.lower()
    assert "type" in echo_tool.parameters
    assert echo_tool.parameters["type"] == "object"


@pytest.mark.asyncio
async def test_echo_tool_string_message():
    """Test echo_tool with string message."""
    result = await echo_tool.execute({"message": "Hello World"})
    assert result == "Echo: Hello World"


@pytest.mark.asyncio
async def test_echo_tool_empty_message():
    """Test echo_tool with empty message."""
    result = await echo_tool.execute({"message": ""})
    assert result == "Echo: "


@pytest.mark.asyncio
async def test_echo_tool_missing_message():
    """Test echo_tool with missing message key."""
    result = await echo_tool.execute({})
    assert result == "Echo: "


@pytest.mark.asyncio
async def test_echo_tool_dict_message():
    """Test echo_tool with dict message (serialized)."""
    result = await echo_tool.execute({"message": {"key": "value", "number": 42}})
    assert "Echo:" in result
    # Check that JSON is in the result
    parsed = json.loads(result.replace("Echo: ", ""))
    assert parsed["key"] == "value"
    assert parsed["number"] == 42


@pytest.mark.asyncio
async def test_echo_tool_list_message():
    """Test echo_tool with list message (serialized)."""
    result = await echo_tool.execute({"message": [1, 2, 3, "test"]})
    assert "Echo:" in result
    parsed = json.loads(result.replace("Echo: ", ""))
    assert parsed == [1, 2, 3, "test"]


@pytest.mark.asyncio
async def test_echo_tool_number_message():
    """Test echo_tool with number (converted to string)."""
    result = await echo_tool.execute({"message": 42})
    # Numbers are converted to string by str()
    assert "Echo:" in result


@pytest.mark.asyncio
async def test_echo_tool_special_characters():
    """Test echo_tool with special characters."""
    result = await echo_tool.execute({"message": "Hello! @#$%^&*()"})
    assert result == "Echo: Hello! @#$%^&*()"


@pytest.mark.asyncio
async def test_echo_tool_unicode():
    """Test echo_tool with unicode characters."""
    result = await echo_tool.execute({"message": "Hello 世界 🌍"})
    assert result == "Echo: Hello 世界 🌍"


@pytest.mark.asyncio
async def test_echo_tool_multiline():
    """Test echo_tool with multiline message."""
    message = "Line 1\nLine 2\nLine 3"
    result = await echo_tool.execute({"message": message})
    assert result == f"Echo: {message}"


# ╭────────────────────────────────────────────────────────────╮
# │  Calculator Tool Tests                                       │
# ╰────────────────────────────────────────────────────────────╯


def test_calculator_tool_structure():
    """Test calculator_tool is correctly structured."""
    assert isinstance(calculator_tool, AgentTool)
    assert calculator_tool.name == "calculator"
    assert "calculate" in calculator_tool.description.lower()
    assert calculator_tool.parameters["type"] == "object"
    assert "expression" in calculator_tool.parameters["properties"]


@pytest.mark.asyncio
async def test_calculator_addition():
    """Test calculator with addition."""
    result = await calculator_tool.execute({"expression": "2 + 2"})
    assert result == "4"


@pytest.mark.asyncio
async def test_calculator_subtraction():
    """Test calculator with subtraction."""
    result = await calculator_tool.execute({"expression": "10 - 3"})
    assert result == "7"


@pytest.mark.asyncio
async def test_calculator_multiplication():
    """Test calculator with multiplication."""
    result = await calculator_tool.execute({"expression": "6 * 7"})
    assert result == "42"


@pytest.mark.asyncio
async def test_calculator_division():
    """Test calculator with division."""
    result = await calculator_tool.execute({"expression": "15 / 3"})
    assert result == "5"


@pytest.mark.asyncio
async def test_calculator_floordiv():
    """Test calculator with floor division."""
    result = await calculator_tool.execute({"expression": "17 // 3"})
    assert result == "5"


@pytest.mark.asyncio
async def test_calculator_power():
    """Test calculator with exponentiation."""
    result = await calculator_tool.execute({"expression": "2 ** 3"})
    assert result == "8"


@pytest.mark.asyncio
async def test_calculator_modulo():
    """Test calculator with modulo."""
    result = await calculator_tool.execute({"expression": "17 % 5"})
    assert result == "2"


@pytest.mark.asyncio
async def test_calculator_complex_expression():
    """Test calculator with complex expression."""
    result = await calculator_tool.execute({"expression": "(2 + 3) * 4"})
    assert result == "20"


@pytest.mark.asyncio
async def test_calculator_decimal():
    """Test calculator with decimal numbers."""
    result = await calculator_tool.execute({"expression": "3.5 * 2"})
    assert result == "7"


@pytest.mark.asyncio
async def test_calculator_negative_numbers():
    """Test calculator with negative numbers."""
    result = await calculator_tool.execute({"expression": "-5 + 3"})
    assert result == "-2"


@pytest.mark.asyncio
async def test_calculator_nested_parentheses():
    """Test calculator with nested parentheses."""
    result = await calculator_tool.execute({"expression": "((1 + 2) * 3) + 4"})
    assert result == "13"


@pytest.mark.asyncio
async def test_calculator_empty_expression():
    """Test calculator with empty expression."""
    result = await calculator_tool.execute({"expression": ""})
    assert isinstance(result, dict)
    assert "error" in result


@pytest.mark.asyncio
async def test_calculator_missing_expression():
    """Test calculator with missing expression key."""
    result = await calculator_tool.execute({})
    assert isinstance(result, dict)
    assert "error" in result


@pytest.mark.asyncio
async def test_calculator_division_by_zero():
    """Test calculator handles division by zero."""
    result = await calculator_tool.execute({"expression": "10 / 0"})
    assert isinstance(result, dict)
    assert "error" in result
    assert "zero" in result["error"].lower()


@pytest.mark.asyncio
async def test_calculator_invalid_characters():
    """Test calculator rejects invalid characters."""
    result = await calculator_tool.execute({"expression": "2 + 'abc'"})
    assert isinstance(result, dict)
    assert "error" in result


@pytest.mark.asyncio
async def test_calculator_syntax_error():
    """Test calculator handles syntax errors."""
    result = await calculator_tool.execute({"expression": "2 + * 3"})
    assert isinstance(result, dict)
    assert "error" in result


@pytest.mark.asyncio
async def test_calculator_whitespace():
    """Test calculator handles various whitespace."""
    result = await calculator_tool.execute({"expression": "  2  +   3  "})
    assert result == "5"


# ╭────────────────────────────────────────────────────────────╮
# │  DateTime Tool Tests                                         │
# ╰────────────────────────────────────────────────────────────╯


def test_datetime_tool_structure():
    """Test datetime_tool is correctly structured."""
    assert isinstance(datetime_tool, AgentTool)
    assert datetime_tool.name == "datetime"
    assert (
        "date" in datetime_tool.description.lower()
        or "time" in datetime_tool.description.lower()
    )
    assert datetime_tool.parameters["type"] == "object"
    # format parameter is optional


@pytest.mark.asyncio
async def test_datetime_default_format():
    """Test datetime_tool with default (ISO) format."""
    result = await datetime_tool.execute({})
    # Should be ISO format
    # Example: 2024-01-15T09:30:00.123456
    assert isinstance(result, str)
    # Check ISO format pattern
    iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    assert re.match(iso_pattern, result)


@pytest.mark.asyncio
async def test_datetime_iso_format():
    """Test datetime_tool with explicit ISO format."""
    result = await datetime_tool.execute({"format": "iso"})
    iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    assert re.match(iso_pattern, result)


@pytest.mark.asyncio
async def test_datetime_date_format():
    """Test datetime_tool with date format."""
    result = await datetime_tool.execute({"format": "date"})
    # Should be YYYY-MM-DD format
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    assert re.match(date_pattern, result)


@pytest.mark.asyncio
async def test_datetime_time_format():
    """Test datetime_tool with time format."""
    result = await datetime_tool.execute({"format": "time"})
    # Should be HH:MM:SS format
    time_pattern = r"^\d{2}:\d{2}:\d{2}$"
    assert re.match(time_pattern, result)


@pytest.mark.asyncio
async def test_datetime_full_format():
    """Test datetime_tool with full format."""
    result = await datetime_tool.execute({"format": "full"})
    # Should be YYYY-MM-DD HH:MM:SS format
    full_pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    assert re.match(full_pattern, result)


@pytest.mark.asyncio
async def test_datetime_timestamp_format():
    """Test datetime_tool with timestamp format."""
    result = await datetime_tool.execute({"format": "timestamp"})
    # Should be a Unix timestamp (integer as string)
    assert result.isdigit()
    # Should be a reasonable timestamp (within last few years)
    timestamp = int(result)
    now = int(datetime.now().timestamp())
    assert abs(timestamp - now) < 60  # Within 60 seconds


@pytest.mark.asyncio
async def test_datetime_result_is_reasonable():
    """Test that datetime_tool returns a reasonable current time."""
    result = await datetime_tool.execute({"format": "timestamp"})
    timestamp = int(result)
    now = int(datetime.now().timestamp())

    # Should be very close to current time
    assert abs(timestamp - now) < 5  # Within 5 seconds


@pytest.mark.asyncio
async def test_datetime_different_formats_different():
    """Test that different formats return different looking results."""
    iso_result = await datetime_tool.execute({"format": "iso"})
    date_result = await datetime_tool.execute({"format": "date"})
    time_result = await datetime_tool.execute({"format": "time"})
    full_result = await datetime_tool.execute({"format": "full"})

    # All should be different formats
    assert iso_result != date_result
    assert date_result != time_result
    assert full_result != iso_result


# ╭────────────────────────────────────────────────────────────╮
# │  Integration Tests                                           │
# ╰────────────────────────────────────────────────────────────╯


@pytest.mark.asyncio
async def test_all_tools_are_callable():
    """Test that all built-in tools can be called."""
    tools = [echo_tool, calculator_tool, datetime_tool]

    for tool in tools:
        # Test with minimal args
        if tool.name == "echo":
            result = await tool.execute({"message": "test"})
            assert "test" in result
        elif tool.name == "calculator":
            result = await tool.execute({"expression": "1+1"})
            assert result == "2"
        elif tool.name == "datetime":
            result = await tool.execute({})
            assert isinstance(result, str)
            assert len(result) > 0


@pytest.mark.asyncio
async def test_tool_execution_timing():
    """Test that tools execute in reasonable time."""
    start = asyncio.get_event_loop().time()

    await echo_tool.execute({"message": "test"})
    await calculator_tool.execute({"expression": "1+1"})
    await datetime_tool.execute({})

    elapsed = asyncio.get_event_loop().time() - start
    assert elapsed < 1.0  # Should complete in under 1 second


def test_tool_parameter_schemas():
    """Test that all tools have valid parameter schemas."""
    tools = [echo_tool, calculator_tool, datetime_tool]

    for tool in tools:
        assert "type" in tool.parameters
        assert tool.parameters["type"] == "object"
        assert "properties" in tool.parameters

        # Check for required fields where applicable
        if tool.name == "echo":
            assert "message" in tool.parameters["properties"]
        elif tool.name == "calculator":
            assert "expression" in tool.parameters["properties"]
        elif tool.name == "datetime":
            # format is optional, so properties might be empty or have format
            pass
