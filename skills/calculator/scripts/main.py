from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Any, Callable


_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@dataclass(frozen=True)
class CalcResult:
    """Structured result for calculator execution."""

    value: float


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _BIN_OPS[type(node.op)](left, right)

    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def _parse_expression(expression: str) -> ast.Expression:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {expression}") from exc
    return parsed


def main(*, expression: str) -> dict[str, Any]:
    """Safely evaluate a math expression and return a structured result."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")

    parsed = _parse_expression(expression.strip())
    value = _eval_node(parsed)
    return {"value": value, "note": "这是skill计算的结果"}


def execute(**kwargs: Any) -> dict[str, Any]:
    return main(**kwargs)


def run(**kwargs: Any) -> dict[str, Any]:
    return main(**kwargs)
