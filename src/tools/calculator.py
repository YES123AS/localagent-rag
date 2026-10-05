"""A small, deterministic arithmetic evaluator with an AST allowlist."""

from __future__ import annotations

import ast
import math
import operator
from typing import Callable


class CalculatorError(ValueError):
    """Raised when an expression is invalid or outside the safe subset."""


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[int | float, int | float], int | float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[int | float], int | float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPRESSION_LENGTH = 200
_MAX_ABS_OPERAND = 10**100
_MAX_ABS_EXPONENT = 100


def _normalize_expression(expression: str) -> str:
    normalized = (
        expression.strip()
        .replace("×", "*")
        .replace("÷", "/")
        .replace("（", "(")
        .replace("）", ")")
    )
    # In calculator-style input, a single caret conventionally means exponentiation.
    normalized = normalized.replace("^", "**")
    if not normalized:
        raise CalculatorError("表达式不能为空。")
    if len(normalized) > _MAX_EXPRESSION_LENGTH:
        raise CalculatorError("表达式过长。")
    return normalized


def _validate_number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError("只支持整数和小数。")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("计算结果不是有限数值。")
    if abs(value) > _MAX_ABS_OPERAND:
        raise CalculatorError("数值超出安全计算范围。")
    return value


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        return _validate_number(node.value)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        operand = _evaluate(node.operand)
        return _validate_number(_UNARY_OPERATORS[type(node.op)](operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise CalculatorError("除数不能为零。")
        if isinstance(node.op, ast.Pow):
            if abs(right) > _MAX_ABS_EXPONENT:
                raise CalculatorError("指数超出安全计算范围。")
            if abs(left) > 10**12:
                raise CalculatorError("幂运算底数超出安全计算范围。")
        try:
            return _validate_number(_BINARY_OPERATORS[type(node.op)](left, right))
        except ZeroDivisionError as exc:
            raise CalculatorError("除数不能为零。") from exc
        except OverflowError as exc:
            raise CalculatorError("计算结果超出安全范围。") from exc

    raise CalculatorError("表达式包含不支持的操作。")


def calculate_expression(expression: str) -> int | float:
    """Evaluate arithmetic without ``eval`` or access to Python objects."""
    normalized = _normalize_expression(expression)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError("表达式语法无效。") from exc
    return _evaluate(tree)


def format_calculation_result(value: int | float) -> str:
    """Format a safe result without unnecessary trailing decimal places."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return format(value, ".15g")
    return str(value)
