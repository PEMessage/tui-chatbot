"""Built-in agent tools.

Basic tools useful for testing and simple agent implementations:
- echo_tool: Echoes back input
- calculator_tool: Simple math evaluation
- datetime_tool: Returns current date/time
"""

from __future__ import annotations

import asyncio
import operator
from datetime import datetime
from typing import Any, Union

from .types import AgentTool


async def echo_tool_execute(args: dict) -> str:
    """Echo back the input as a string.

    Args:
        args: Dictionary with optional 'message' key

    Returns:
        The echoed message
    """
    message = args.get("message", "")
    # Handle various input types
    if isinstance(message, (dict, list)):
        import json

        return f"Echo: {json.dumps(message)}"
    return f"Echo: {message}"


echo_tool = AgentTool(
    name="echo",
    description="Echo back the input message",
    parameters={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Message to echo back",
            }
        },
        "required": ["message"],
    },
    execute=echo_tool_execute,
)


# Calculator operators
_CALC_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "**": operator.pow,
    "%": operator.mod,
}


async def calculator_tool_execute(args: dict) -> Union[str, dict]:
    """Execute simple math calculation.

    Supports basic operations: +, -, *, /, //, **, %

    Args:
        args: Dictionary with 'expression' key containing math expression

    Returns:
        Result as string or error dict
    """
    expression = args.get("expression", "")

    if not expression:
        return {"error": "No expression provided"}

    try:
        # Validate expression - only allow safe characters
        allowed_chars = set("0123456789+-*/().% //** ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression"}

        # Simple evaluation for basic operations
        # Using eval with limited globals/locals for safety
        result = eval(expression, {"__builtins__": {}}, {})  # nosec B307 - limited to math

        # Format result
        if isinstance(result, float):
            # Avoid floating point artifacts
            if result == int(result):
                result = int(result)

        return str(result)

    except ZeroDivisionError:
        return {"error": "Division by zero"}
    except Exception as e:
        return {"error": f"Calculation error: {str(e)}"}


calculator_tool = AgentTool(
    name="calculator",
    description="Calculate simple mathematical expressions",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression to calculate (e.g., '2 + 2', '10 * 5')",
            }
        },
        "required": ["expression"],
    },
    execute=calculator_tool_execute,
)


async def datetime_tool_execute(args: dict) -> str:
    """Get current date and time.

    Args:
        args: Dictionary with optional 'format' key
            - 'iso': ISO format (default)
            - 'date': Date only
            - 'time': Time only
            - 'full': Full datetime string
            - 'timestamp': Unix timestamp

    Returns:
        Current datetime as formatted string
    """
    now = datetime.now()
    format_type = args.get("format", "iso")

    if format_type == "date":
        return now.strftime("%Y-%m-%d")
    elif format_type == "time":
        return now.strftime("%H:%M:%S")
    elif format_type == "full":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    elif format_type == "timestamp":
        return str(int(now.timestamp()))
    else:  # iso
        return now.isoformat()


datetime_tool = AgentTool(
    name="datetime",
    description="Get current date and time",
    parameters={
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": "Output format: 'iso', 'date', 'time', 'full', 'timestamp'",
                "enum": ["iso", "date", "time", "full", "timestamp"],
            }
        },
    },
    execute=datetime_tool_execute,
)
