"""LibCST mutation operators and candidate discovery for NumPy pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import libcst as cst
from libcst.metadata import MetadataWrapper, ParentNodeProvider, PositionProvider

MOR_OPERATORS = {
    "+": cst.Add,
    "-": cst.Subtract,
    "*": cst.Multiply,
    "/": cst.Divide,
    "@": cst.MatrixMultiply,
    "%": cst.Modulo,
    "//": cst.FloorDivide,
    "**": cst.Power,
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

BOR_OPERATORS = {
    "&": cst.BitAnd,
    "|": cst.BitOr,
    "^": cst.BitXor,
    "<<": cst.LeftShift,
    ">>": cst.RightShift,
}

OPERATOR_SETS = (MOR_OPERATORS, ROOR_OPERATORS, COR_OPERATORS, BOR_OPERATORS)

CALL_SWAPS = (
    ("np.sum", "np.prod"),
    ("np.prod", "np.sum"),
    ("np.linalg.inv", "np.linalg.pinv"),
    ("np.linalg.pinv", "np.linalg.inv"),
)

UNARY_MUTATABLE = (cst.Minus, cst.Plus, cst.BitInvert)

# RTR: replace ``return expr`` with a constant that changes the returned value.
RTR_REPLACEMENTS = (
    ("None", cst.Name("None")),
    ("0", cst.Integer("0")),
    ('""', cst.SimpleString('""')),
    ("False", cst.Name("False")),
    ("[]", cst.List(elements=[])),
)


@dataclass
class Candidate:
    node: cst.CSTNode
    mutation_type: str
    detail: str | None
    function: str | None
    line: int
    index: int | None = None
    context: dict | None = None


class CandidateVisitor(cst.CSTVisitor):
    """Collect every applicable mutation candidate with its location context."""

    def __init__(self, module: cst.Module, wrapper: MetadataWrapper) -> None:
        self.module = module
        self.wrapper = wrapper
        self._positions = wrapper.resolve(PositionProvider)
        self._parents = wrapper.resolve(ParentNodeProvider)
        self._function_stack: list[str] = []
        self._scope_stack: list[list[str]] = []  # names per function
        self._unary_depth: int = 0
        self.candidates: list[Candidate] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._function_stack.append(node.name.value)
        self._scope_stack.append([])
        return True

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        self._scope_stack.pop()
        self._function_stack.pop()

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> bool:
        operator = self.module.code_for_node(node.operator).strip()
        if operator in MOR_OPERATORS:
            self._record(node, "MOR", operator, line=self._positions[node.operator].start.line)
        if operator in BOR_OPERATORS:
            self._record(node, "BOR", operator, line=self._positions[node.operator].start.line)
        return True

    def visit_UnaryOperation(self, node: cst.UnaryOperation) -> bool:
        if isinstance(node.operator, UNARY_MUTATABLE):
            self._record(node, "UOI", "remove", line=self._positions[node].start.line)
        self._unary_depth += 1
        return True

    def leave_UnaryOperation(self, node: cst.UnaryOperation) -> None:
        self._unary_depth -= 1

    def visit_Return(self, node: cst.Return) -> bool:
        if node.value is not None:
            self._record(node, "RTR", "return", line=self._positions[node].start.line)
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
            # EXM: membership/equality expansion. A ``==``/``!=`` comparator
            # against a literal can become ``in (literal, sibling)``; an
            # ``in`` over a tuple/set can collapse to ``==`` a single member.
            if operator in ("==", "!=", "in", "not in"):
                self._record(
                    node,
                    "EXM",
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
        positional = [i for i, arg in enumerate(node.args) if arg.keyword is None and arg.star == ""]
        if len(positional) >= 2:
            self._record(node, "ARC", "argswap", index=positional[0])
        if len(positional) > 1:
            for index in positional:
                self._record(node, "AAR", None, index=index)
        # SVR: this call is an argument to an outer call -> pulling a
        # positional arg from the inner call up to the outer call is a
        # real fault family (e.g. enumerate(tqdm_class(iterable, start)) ->
        # enumerate(tqdm_class(iterable), start)).
        parent = self._parents.get(node)
        if isinstance(parent, cst.Arg):
            inner_positional = positional
            if len(inner_positional) >= 2:
                self._record(node, "SVR", None, line=self._positions[node].start.line,
                             context={"inner": list(inner_positional)})
        if self._unary_depth == 0 and self._unary_insertable(node):
            self._record(node, "UOI", "insert", line=self._positions[node].start.line)
        return True

    def visit_Integer(self, node: cst.Integer) -> bool:
        self._record(node, "LCR", node.value, line=self._positions[node].start.line)
        if self._unary_depth == 0 and self._unary_insertable(node):
            self._record(node, "UOI", "insert", line=self._positions[node].start.line)
        return True

    def visit_Float(self, node: cst.Float) -> bool:
        self._record(node, "LCR", node.value, line=self._positions[node].start.line)
        if self._unary_depth == 0 and self._unary_insertable(node):
            self._record(node, "UOI", "insert", line=self._positions[node].start.line)
        return True

    def visit_Name(self, node: cst.Name) -> bool:
        if node.value in ("True", "False"):
            self._record(node, "LCR", node.value, line=self._positions[node].start.line)
        if self._unary_depth == 0 and self._unary_insertable(node):
            self._record(node, "UOI", "insert", line=self._positions[node].start.line)
        # LVR: swap a plain name with another name seen in this function.
        if self._scope_stack and not node.value.startswith("__") and self._unary_insertable(node):
            siblings = [n for n in self._scope_stack[-1] if n != node.value]
            if siblings:
                self._record(
                    node, "LVR", node.value, line=self._positions[node].start.line,
                    context={"siblings": siblings},
                )
        if self._scope_stack:
            self._scope_stack[-1].append(node.value)
        return True

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        if self._unary_depth == 0 and self._unary_insertable(node):
            self._record(node, "UOI", "insert", line=self._positions[node].start.line)
        # LVR-attr: an attribute expression ``x.foo`` -> an in-scope name.
        if self._scope_stack:
            siblings = [n for n in self._scope_stack[-1] if n not in ("True", "False", "None")]
            if siblings:
                self._record(
                    node, "LVR", None, line=self._positions[node].start.line,
                    context={"siblings": siblings},
                )
        # ATTR: collapse ``x.foo`` to bare ``x`` (attribute removal). Only safe
        # for value-position attributes (not ``.foo`` in an assignment target).
        parent = self._parents.get(node)
        if self._unary_depth == 0 and not isinstance(
            parent, (cst.AssignTarget, cst.AnnAssign, cst.AugAssign)
        ) and isinstance(node.value, (cst.Name, cst.Attribute)):
            self._record(node, "ATTR", None, line=self._positions[node].start.line)
        return True

    def visit_If(self, node: cst.If) -> bool:
        self._record(node.test, "NOTR", "not", line=self._positions[node.test].start.line)
        self._record(node.test, "CDR", "constant", line=self._positions[node.test].start.line)
        return True

    def visit_While(self, node: cst.While) -> bool:
        self._record(node.test, "NOTR", "not", line=self._positions[node.test].start.line)
        self._record(node.test, "CDR", "constant", line=self._positions[node.test].start.line)
        return True

    def visit_IfExp(self, node: cst.IfExp) -> bool:
        self._record(node.test, "CDR", "constant", line=self._positions[node.test].start.line)
        return True

    def visit_SimpleString(self, node: cst.SimpleString) -> bool:
        self._record(node, "SCP", node.value, line=self._positions[node].start.line)
        return True

    def visit_Subscript(self, node: cst.Subscript) -> bool:
        index = perturbable_element_index(node.slice)
        if index is not None:
            self._record(node, "INX", None, index=index, line=self._positions[node].start.line)
        if self._unary_depth == 0 and self._unary_insertable(node):
            self._record(node, "UOI", "insert", line=self._positions[node].start.line)
        return True

    def _unary_insertable(self, node: cst.CSTNode) -> bool:
        """True when wrapping ``node`` in a unary minus/plus keeps valid syntax.

        Insertion is only valid in value/expression position inside a function
        body: not at module scope (which can break import), not on function or
        class names, parameters, imports, attribute selectors, assignment
        targets, keyword-argument names, or module-level statement keywords.
        """
        if not self._function_stack:
            return False
        parent = self._parents.get(node)
        if isinstance(parent, (cst.FunctionDef, cst.ClassDef, cst.Param, cst.Lambda)):
            return False
        if isinstance(parent, (cst.ImportAlias, cst.ImportFrom, cst.AsName)):
            return False
        # ``-a`` is not valid syntactically inside a dotted name; negation
        # belongs on the whole attribute expression, not its parts.
        if isinstance(parent, cst.Attribute):
            return False
        if isinstance(parent, (cst.Del, cst.Global, cst.Nonlocal, cst.Decorator)):
            return False
        if isinstance(parent, cst.Arg) and node is parent.keyword:
            return False
        if isinstance(parent, (cst.AssignTarget, cst.AnnAssign, cst.AugAssign)) and node is parent.target:
            return False
        if isinstance(parent, (cst.For, cst.CompFor)) and node is parent.target:
            return False
        if isinstance(parent, cst.With) and any(node is item.item for item in parent.items):
            return False
        return True

    def _record(
        self,
        node: cst.CSTNode,
        mutation_type: str,
        detail: str | None,
        *,
        index: int | None = None,
        line: int | None = None,
        context: dict | None = None,
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
                context=context,
            )
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