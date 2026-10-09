"""Generate mutant sources and their ledgers from a target module."""

from __future__ import annotations

import libcst as cst
import numpy as np
from libcst.metadata import MetadataWrapper

from .mutators import (
    ASR_OPERATORS,
    CALL_SWAPS,
    CandidateVisitor,
    MOR_OPERATORS,
    ROOR_OPERATORS,
    COR_OPERATORS,
    RTR_REPLACEMENTS,
    UNARY_MUTATABLE,
    Candidate,
)


def _parse(source_text: str) -> tuple[cst.Module, MetadataWrapper, CandidateVisitor]:
    wrapper = MetadataWrapper(cst.parse_module(source_text))
    visitor = CandidateVisitor(wrapper.module, wrapper)
    wrapper.module.visit(visitor)
    return wrapper.module, wrapper, visitor


def generate_mutants(
    source_text: str,
    *,
    limit: int | None = None,
    priority_lines: frozenset[int] = frozenset(),
) -> list[dict]:
    """Return one entry per distinct mutant.

    Each entry carries ``mutant_id``, ``line``, ``mutation_type``, ``detail``,
    ``function``, and ``source``. Mutants whose source is unchanged or already
    seen are dropped.

    ``limit`` stops generation early (candidates on ``priority_lines`` first),
    which matters for large files where full generation re-serializes the
    module once per candidate.
    """
    module, wrapper, visitor = _parse(source_text)

    candidates = sorted(
        visitor.candidates,
        key=lambda c: (
            c.line not in priority_lines,
            c.line,
            c.index if c.index is not None else -1,
        ),
    )

    mutants: list[dict] = []
    seen_sources: set[str] = set()
    for candidate in candidates:
        operator = candidate.detail or "reshape"
        for detail, replacement in _replacements(module, candidate):
            mutant_module = _apply(module, candidate.node, replacement)
            new_source = mutant_module.code
            if new_source == source_text or new_source in seen_sources:
                continue
            seen_sources.add(new_source)
            mutants.append(
                {
                    "mutant_id": (
                        f"L{candidate.line}-{candidate.mutation_type}-"
                        f"{detail or 'swap'}-{len(mutants)}"
                    ),
                    "line": candidate.line,
                    "mutation_type": candidate.mutation_type,
                    "operator": operator,
                    "detail": detail,
                    "function": candidate.function,
                    "source": new_source,
                }
            )
            if limit is not None and len(mutants) >= limit:
                return mutants
    return mutants


def _replacements(module: cst.Module, candidate: Candidate):
    node = candidate.node
    if candidate.mutation_type == "MOR":
        for name, operator_class in MOR_OPERATORS.items():
            yield name, cst.BinaryOperation(
                left=node.left,
                operator=operator_class(),
                right=node.right,
            )
    elif candidate.mutation_type == "ROOR":
        for name, operator_class in ROOR_OPERATORS.items():
            targets = [
                cst.ComparisonTarget(
                    operator=operator_class(), comparator=target.comparator
                )
                if index == candidate.index
                else target
                for index, target in enumerate(node.comparisons)
            ]
            yield name, cst.Comparison(left=node.left, comparisons=targets)
    elif candidate.mutation_type == "COR":
        for name, operator_class in COR_OPERATORS.items():
            yield name, cst.BooleanOperation(
                left=node.left,
                operator=operator_class(),
                right=node.right,
            )
    elif candidate.mutation_type == "ASR":
        for name, operator_class in ASR_OPERATORS.items():
            yield name, cst.AugAssign(
                target=node.target,
                operator=operator_class(),
                value=node.value,
            )
    elif candidate.mutation_type == "FCS":
        partner = _swap_partner(candidate.detail)
        yield partner, cst.Call(
            func=cst.parse_expression(partner),
            args=node.args,
        )
    elif candidate.mutation_type == "ARC":
        positional = [
            i for i, arg in enumerate(node.args)
            if arg.keyword is None and arg.star == ""
        ]
        if len(positional) >= 2:
            first, second = positional[0], positional[1]
            args = list(node.args)
            args[first], args[second] = args[second], args[first]
            yield "swap", cst.Call(func=node.func, args=args)
    elif candidate.mutation_type == "LCR":
        yield from _literal_replacements(node, candidate.detail)
    elif candidate.mutation_type == "NOTR":
        if isinstance(node, cst.UnaryOperation) and isinstance(
            node.operator, cst.Not
        ):
            yield "remove", node.expression
        yield "insert", cst.UnaryOperation(operator=cst.Not(), expression=node)
    elif candidate.mutation_type == "CDR":
        yield "True", cst.Name("True")
        yield "False", cst.Name("False")
    elif candidate.mutation_type == "SCP":
        yield from _string_replacements(node)
    elif candidate.mutation_type == "AAR":
        args = [
            arg for i, arg in enumerate(node.args) if i != candidate.index
        ]
        yield f"remove#{candidate.index}", cst.Call(func=node.func, args=args)
    elif candidate.mutation_type == "INX":
        yield from _index_replacements(node, candidate)
    elif candidate.mutation_type == "UOI":
        if isinstance(node, cst.UnaryOperation) and isinstance(
            node.operator, UNARY_MUTATABLE
        ):
            yield "remove", node.expression
        else:
            yield "-", cst.UnaryOperation(operator=cst.Minus(), expression=node)
            yield "+", cst.UnaryOperation(operator=cst.Plus(), expression=node)
    elif candidate.mutation_type == "RTR":
        for detail, value in RTR_REPLACEMENTS:
            yield detail, cst.Return(value=value)


def _string_replacements(node: cst.SimpleString):
    """Yield (detail, replacement) for string-literal mutants.

    Covers the fault family "wrong string constant" (empty string, whitespace
    collapse, prefix/suffix truncation, swapped quote style). Mutants whose
    source is unchanged are dropped by the caller's dedup.
    """
    original = node.value
    try:
        value = node.evaluated_value
    except Exception:
        return
    if value is None:
        return
    variants = {
        "empty": "",
    }
    stripped = value.strip()
    if stripped != value:
        variants["strip"] = stripped
    if len(value) > 1:
        variants["drop_first"] = value[1:]
        variants["drop_last"] = value[:-1]
    if value != "" and value.isspace():
        variants["single_space"] = " "
    for detail, replacement in variants.items():
        if replacement == value:
            continue
        yield detail, cst.SimpleString(repr(replacement))


def _int_node(value: int) -> cst.BaseExpression:
    """Build an integer expression node, handling negatives."""
    if value < 0:
        return cst.UnaryOperation(
            operator=cst.Minus(), expression=cst.Integer(str(abs(value)))
        )
    return cst.Integer(str(value))


def _float_node(value: float) -> cst.BaseExpression:
    """Build a float expression node, handling negatives."""
    if value < 0:
        return cst.UnaryOperation(
            operator=cst.Minus(), expression=cst.Float(repr(abs(value)))
        )
    return cst.Float(repr(value))


def _literal_replacements(node, original: str):
    """Yield (detail, replacement) for numeric/boolean literal mutants."""
    if isinstance(node, cst.Integer):
        try:
            value = int(original.replace("_", ""), 0)
        except ValueError:
            return
        for delta in (-1, 1):
            yield f"{value + delta}", _int_node(value + delta)
        if value == 0:
            yield "1", cst.Integer("1")
        elif value == 1:
            yield "0", cst.Integer("0")
    elif isinstance(node, cst.Float):
        try:
            value = float(original.replace("_", ""))
        except ValueError:
            return
        for delta in (-1.0, 1.0):
            new_value = value + delta
            yield repr(new_value), _float_node(new_value)
        # Small magnitude-scaled deltas express off-by-threshold / boundary
        # bugs (e.g. 1000.0 -> 999.95), which ±1.0 cannot reach.
        scale = 10 ** int(np.log10(abs(value))) if value != 0 else 1.0
        for fraction in (0.005, 0.0005):
            delta = scale * fraction
            for sign in (-1, 1):
                new_value = value + sign * delta
                yield repr(new_value), _float_node(new_value)
    elif isinstance(node, cst.Name):
        if node.value == "True":
            yield "False", cst.Name("False")
        elif node.value == "False":
            yield "True", cst.Name("True")


def _index_replacements(node, candidate):
    """Yield (detail, replacement) for perturbed subscript indices."""
    def integer_alt(value: cst.Integer):
        base = int(value.value, 0)
        for delta in (-1, 1):
            yield f"{base + delta}", _int_node(base + delta)

    target_element = node.slice[candidate.index]
    slice_value = target_element.slice

    def rebuilt(new_slice):
        new_element = cst.SubscriptElement(slice=new_slice)
        elements = [
            new_element if i == candidate.index else e
            for i, e in enumerate(node.slice)
        ]
        return cst.Subscript(value=node.value, slice=elements)

    if isinstance(slice_value, cst.Index) and isinstance(slice_value.value, cst.Integer):
        for detail, alt in integer_alt(slice_value.value):
            yield detail, rebuilt(cst.Index(alt))
    elif isinstance(slice_value, cst.Index) and isinstance(
        slice_value.value, cst.UnaryOperation
    ) and isinstance(slice_value.value.expression, cst.Integer):
        base = int(slice_value.value.expression.value, 0)
        for delta in (-1, 1):
            yield f"{-base + delta}", _int_node(-base + delta)
    elif isinstance(slice_value, cst.Slice):
        if isinstance(slice_value.lower, cst.Integer):
            for detail, alt in integer_alt(slice_value.lower):
                yield detail, rebuilt(
                    cst.Slice(lower=alt, upper=slice_value.upper, step=slice_value.step)
                )
        elif isinstance(slice_value.upper, cst.Integer):
            for detail, alt in integer_alt(slice_value.upper):
                yield detail, rebuilt(
                    cst.Slice(lower=slice_value.lower, upper=alt, step=slice_value.step)
                )


def _swap_partner(call_name: str) -> str:
    for old, new in CALL_SWAPS:
        if call_name == old:
            return new
    raise ValueError(f"No swap partner for {call_name!r}")


class _ReplaceTransformer(cst.CSTTransformer):
    def __init__(self, target: cst.CSTNode, replacement: cst.CSTNode) -> None:
        self.target = target
        self.replacement = replacement
        self.done = False

    def on_leave(self, original_node, updated_node):
        if original_node is self.target and not self.done:
            self.done = True
            return self.replacement
        return updated_node


def _apply(
    module: cst.Module,
    target: cst.CSTNode,
    replacement: cst.CSTNode,
) -> cst.Module:
    transformer = _ReplaceTransformer(target, replacement)
    return module.visit(transformer)