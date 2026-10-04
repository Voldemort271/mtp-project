"""LibCST mutation operators and candidate discovery for NumPy pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

MOR_OPERATORS = {
    "+": cst.Add,
    "-": cst.Subtract,
    "*": cst.Multiply,
    "/": cst.Divide,
    "@": cst.MatrixMultiply,
}

ROOR_OPERATORS = {
    "==": cst.Equal,
    "!=": cst.NotEqual,
    "<": cst.LessThan,
    "<=": cst.LessThanEqual,
    ">": cst.GreaterThan,
    ">=": cst.GreaterThanEqual,
    "in": cst.In,
    "not in": cst.NotIn,
    "is": cst.Is,
    "is not": cst.IsNot,
}

COR_OPERATORS = {
    "and": cst.And,
    "or": cst.Or,
}

ASR_OPERATORS = {
    "+=": cst.AddAssign,
    "-=": cst.SubtractAssign,
    "*=": cst.MultiplyAssign,
    "/=": cst.DivideAssign,
    "//=": cst.FloorDivideAssign,
    "**=": cst.PowerAssign,
    "@=": cst.MatrixMultiplyAssign,
}

OPERATOR_SETS = (MOR_OPERATORS, ROOR_OPERATORS, COR_OPERATORS)

CALL_SWAPS = (
    ("np.sum", "np.prod"),
    ("np.prod", "np.sum"),
    ("np.linalg.inv", "np.linalg.pinv"),
    ("np.linalg.pinv", "np.linalg.inv"),
)


@dataclass
class Candidate:
    node: cst.CSTNode
    mutation_type: str
    detail: str | None
    function: str | None
    line: int
    index: int | None = None


class CandidateVisitor(cst.CSTVisitor):
    """Collect every applicable mutation candidate with its location context."""

    def __init__(self, module: cst.Module, wrapper: MetadataWrapper) -> None:
        self.module = module
        self.wrapper = wrapper
        self._positions = wrapper.resolve(PositionProvider)
        self._function_stack: list[str] = []
        self.candidates: list[Candidate] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._function_stack.append(node.name.value)
        return True

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._function_stack.pop()

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> bool:
        operator = self.module.code_for_node(node.operator).strip()
        if operator in MOR_OPERATORS:
            self._record(node, "MOR", operator, line=self._positions[node.operator].start.line)
        return True

    def visit_Comparison(self, node: cst.Comparison) -> bool:
        for index, target in enumerate(node.comparisons):
            operator = self.module.code_for_node(target.operator).strip()
            if operator in ROOR_OPERATORS:
                self._record(
                    node,
                    "ROOR",
                    operator,
                    index=index,
                    line=self._positions[target.operator].start.line,
                )
        return True

    def visit_BooleanOperation(self, node: cst.BooleanOperation) -> bool:
        operator = self.module.code_for_node(node.operator).strip()
        if operator in COR_OPERATORS:
            self._record(node, "COR", operator)
        return True

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        operator = self.module.code_for_node(node.operator).strip()
        if operator in ASR_OPERATORS:
            self._record(node, "ASR", operator, line=self._positions[node.operator].start.line)
        return True

    def visit_Call(self, node: cst.Call) -> bool:
        func = self.module.code_for_node(node.func)
        for old, new in CALL_SWAPS:
            if func == old:
                self._record(node, "FCS", func)
        if func.endswith(".reshape") and is_swapable_reshape(node):
            self._record(node, "TSM", None)
        positional = [i for i, arg in enumerate(node.args) if arg.keyword is None]
        if len(positional) > 1:
            for index in positional:
                self._record(node, "AAR", None, index=index)
        return True

    def visit_Integer(self, node: cst.Integer) -> bool:
        self._record(node, "LCR", node.value, line=self._positions[node].start.line)
        return True

    def visit_Float(self, node: cst.Float) -> bool:
        self._record(node, "LCR", node.value, line=self._positions[node].start.line)
        return True

    def visit_Name(self, node: cst.Name) -> bool:
        if node.value in ("True", "False"):
            self._record(node, "LCR", node.value, line=self._positions[node].start.line)
        return True

    def visit_If(self, node: cst.If) -> bool:
        self._record(node.test, "NOTR", "not", line=self._positions[node.test].start.line)
        return True

    def visit_While(self, node: cst.While) -> bool:
        self._record(node.test, "NOTR", "not", line=self._positions[node.test].start.line)
        return True

    def visit_Subscript(self, node: cst.Subscript) -> bool:
        index = perturbable_element_index(node.slice)
        if index is not None:
            self._record(node, "INX", None, index=index, line=self._positions[node].start.line)
        return True

    def _record(
        self,
        node: cst.CSTNode,
        mutation_type: str,
        detail: str | None,
        *,
        index: int | None = None,
        line: int | None = None,
    ) -> None:
        if line is None:
            line = self._positions[node].start.line
        function = self._function_stack[-1] if self._function_stack else None
        self.candidates.append(
            Candidate(
                node=node,
                mutation_type=mutation_type,
                detail=detail,
                function=function,
                line=line,
                index=index,
            )
        )


def is_swapable_reshape(node: cst.Call) -> bool:
    """True when a reshape call has two positional args and one is -1."""
    positional = [arg for arg in node.args if arg.keyword is None]
    if len(positional) != 2:
        return False
    return any(is_neg_one(arg.value) for arg in positional)


def is_neg_one(value: cst.BaseExpression) -> bool:
    return (
        isinstance(value, cst.UnaryOperation)
        and isinstance(value.operator, cst.Minus)
        and isinstance(value.expression, cst.Integer)
        and value.expression.value == "1"
    )


def perturbable_element_index(slice_elements) -> int | None:
    """Index of the first subscript element whose slice we can perturb."""
    for index, element in enumerate(slice_elements):
        if has_perturbable_index(element.slice):
            return index
    return None


def has_perturbable_index(slice_value) -> bool:
    """True when a subscript slice is an integer index we can perturb."""
    if isinstance(slice_value, cst.Index):
        return isinstance(slice_value.value, (cst.Integer, cst.UnaryOperation))
    if isinstance(slice_value, cst.Slice):
        for bound in (slice_value.lower, slice_value.upper):
            if isinstance(bound, cst.Integer):
                return True
    return False