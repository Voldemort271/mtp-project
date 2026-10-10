"""Generate mutant sources and their ledgers from a target module."""

from __future__ import annotations

import random

import libcst as cst
import numpy as np
from libcst.metadata import MetadataWrapper

from .mutators import (
    ASR_OPERATORS,
    BOR_OPERATORS,
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


def candidate_lines(source_text: str) -> set[int]:
    """Return the set of source lines that have at least one mutation candidate.

    Used for triage/diagnostics; it does not serialize any mutant.
    """
    _module, _wrapper, visitor = _parse(source_text)
    return {candidate.line for candidate in visitor.candidates}


def generate_mutants(
    source_text: str,
    *,
    limit: int | None = None,
    seed: int = 0,
    mode: str = "uniform",
) -> list[dict]:
    """Return one entry per distinct mutant.

    Each entry carries ``mutant_id``, ``line``, ``mutation_type``, ``detail``,
    ``function``, and ``source``. Mutants whose source is unchanged or already
    seen are dropped.

    When ``limit`` is set, the retained mutant set is independent of any
    downstream fault labels and of source-line ordering (seeded by ``seed``),
    and is deterministic for a given source and seed.

    ``mode`` selects how the budget is spent when ``limit`` is set:
    - ``"uniform"``: candidates are shuffled and the first ``limit`` distinct
      mutants are kept. Lines with more mutation sites win more budget, which
      is legitimate (complex lines hold more real faults) but covers fewer
      distinct lines in large files.
    - ``"stratified_floor"``: candidates are round-robin'd across lines, taking
      at most one distinct mutant per candidate before moving on, so the budget
      covers as many distinct lines as possible before deepening any line. This
      maximizes fault-line coverage but reduces per-line mutant density.

    The default remains fault-agnostic uniform sampling.
    """
    module, wrapper, visitor = _parse(source_text)

    candidates = sorted(
        visitor.candidates,
        key=lambda c: (c.line, c.index if c.index is not None else -1),
    )
    if limit is None:
        candidates = list(candidates)
    elif mode == "uniform":
        random.Random(seed).shuffle(candidates)
    elif mode == "stratified_floor":
        candidates = _round_robin_candidates(candidates, seed)
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    mutants: list[dict] = []
    seen_sources: set[str] = set()
    for candidate in candidates:
        operator = candidate.detail or "reshape"
        yielded = False
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
            yielded = True
            if limit is not None and len(mutants) >= limit:
                return mutants
            if mode == "stratified_floor":
                # at most one distinct mutant per candidate: maximize line spread
                break
        _ = yielded
    return mutants


def _round_robin_candidates(
    candidates: list[Candidate], seed: int
) -> list[Candidate]:
    """Order candidates line-by-line, round-robin, deterministic per seed.

    Every mutatable line contributes one candidate before any line contributes
    a second, which spreads a fixed mutant budget across the most lines.
    """
    rng = random.Random(seed + 1)
    by_line: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        by_line.setdefault(candidate.line, []).append(candidate)
    lines = sorted(by_line)
    for line in lines:
        rng.shuffle(by_line[line])
    ordered: list[Candidate] = []
    index = 0
    while True:
        advanced = False
        for line in lines:
            if index < len(by_line[line]):
                ordered.append(by_line[line][index])
                advanced = True
        if not advanced:
            break
        index += 1
    return ordered


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
    elif candidate.mutation_type == "EXM":
        yield from _membership_replacements(module, node, candidate)
    elif candidate.mutation_type == "BOR":
        operator_name = candidate.detail
        for name, operator_class in BOR_OPERATORS.items():
            if name == operator_name:
                continue
            yield name, cst.BinaryOperation(
                left=node.left,
                operator=operator_class(),
                right=node.right,
            )
    elif candidate.mutation_type == "LVR":
        siblings = (candidate.context or {}).get("siblings", [])
        if isinstance(node, cst.Attribute):
            for sib in siblings:
                yield f"name-{sib}", cst.Name(sib)
        else:
            for sib in siblings:
                yield f"name-{sib}", cst.Name(sib)
    elif candidate.mutation_type == "ATTR":
        # collapse ``x.foo`` -> ``x``
        yield "drop-attr", node.value
    elif candidate.mutation_type == "SVR":
        # pull the (first two) positional args out of the inner call up to the
        # enclosing call. candidate.node is the inner call; its parent Arg is
        # removed and replaced by a bare call of the remaining args.
        inner = len((candidate.context or {}).get("inner", []))
        if inner >= 2:
            # keep only arg[0] inside inner; the rest would be moved (multi-arg
            # move is complex), so we move the LAST inner positional to the end
            # of the outer call.
            positional = [
                i for i, arg in enumerate(node.args)
                if arg.keyword is None and arg.star == ""
            ]
            if len(positional) >= 2:
                dropped = positional[-1]
                args = [arg for i, arg in enumerate(node.args) if i != dropped]
                yield "shift-arg-inner", cst.Call(func=node.func, args=args)


def _membership_replacements(module: cst.Module, node, candidate):
    """Expand ``== lit`` to membership and collapse ``in (...)`` to equality.

    Expresses the common fault family "single equality should have been a set
    membership" (e.g. ``token == 'def'`` -> ``token in ('def', 'for')``) and
    its reverse (``in (a, b)`` -> ``== a``).
    """
    target = node.comparisons[candidate.index]
    operator = module.code_for_node(target.operator).strip()
    comparator = target.comparator
    # left operand of this comparison step (node.left for the first, else the
    # previous step's comparator)
    if candidate.index == 0:
        left = node.left
    else:
        left = node.comparisons[candidate.index - 1].comparator

    def rebuilt(op, right):
        new_target = cst.ComparisonTarget(operator=op, comparator=right)
        comparisons = list(node.comparisons)
        comparisons[candidate.index] = new_target
        return cst.Comparison(left=node.left, comparisons=comparisons)

    if operator in ("==", "!="):
        # right-hand side is a literal / name / tuple: expand to membership
        if isinstance(comparator, (cst.Integer, cst.Float, cst.SimpleString, cst.Name)):
            member = comparator
            in_op = cst.In() if operator == "==" else cst.NotIn()
            # (x) == lit  ->  x in (lit,)
            yield "in-single", rebuilt(in_op, _as_tuple(module, member))
        elif isinstance(comparator, (cst.Tuple, cst.List, cst.Set)):
            elements = _container_elements(comparator)
            if len(elements) == 1 and operator == "==":
                yield "in-single", rebuilt(cst.In(), elements[0])
            elif len(elements) >= 2:
                # membership of x among some members -> equality with first
                yield "==first", rebuilt(cst.Equal(), elements[0])
    elif operator in ("in", "not in"):
        if isinstance(comparator, (cst.Tuple, cst.List, cst.Set)):
            elements = _container_elements(comparator)
            if len(elements) == 1:
                yield "==member", rebuilt(cst.Equal(), elements[0])
            elif len(elements) >= 2:
                yield "==first", rebuilt(cst.Equal(), elements[0])
        elif isinstance(comparator, (cst.Integer, cst.Float, cst.SimpleString, cst.Name)):
            # x in lit  ->  x == lit
            yield "==member", rebuilt(cst.Equal(), comparator)


def _as_tuple(module: cst.Module, expression) -> cst.BaseExpression:
    if isinstance(expression, cst.Tuple):
        return expression
    # wrap a single lit/name into a one-element tuple: (lit,)
    return cst.Tuple(elements=[cst.Element(value=expression)])


def _container_elements(container) -> list:
    if isinstance(container, cst.Tuple):
        return [e.value for e in container.elements]
    if isinstance(container, cst.List):
        return [e.value for e in container.elements]
    if isinstance(container, cst.Set):
        return [e.value for e in container.elements]
    return []


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

    # Regex-aware string mutations: if the literal looks like a regex pattern
    # (anchors, character classes, escapes, quantifiers), perturb its
    # metacharacters. This expresses the thefuck regex faults (e.g. ``[a-z]+``
    # vs ``[^"]+``, ``^mkdir`` vs ``\\bmkdir``).
    if _looks_like_regex(value):
        yield from _regex_string_replacements(value)


def _looks_like_regex(value: str) -> bool:
    return any(ch in value for ch in "^$[].*+?()|\\{}")


def _regex_string_replacements(value: str):
    """Yield regex-metacharacter perturbations for a string literal."""
    candidates: list[tuple[str, str]] = []

    # anchor <-> word-boundary: ^x <-> \bx, x$ <-> x\b
    if value.startswith("^") and "\\b" not in value[:3]:
        candidates.append(("^->\\b", "\\b" + value[1:]))
    if "\\b" in value and value.startswith("\\b"):
        candidates.append(("\\b->^", "^" + value[2:]))
    if value.endswith("$"):
        candidates.append(("$->\\b", value[:-1] + "\\b"))

    # negate character classes [a-z] -> [^a-z]
    import re as _re

    def negate_class(match: _re.Match) -> str:
        inner = match.group(1)
        return "[^" + inner + "]"

    value_neg = _re.sub(r"\[(\^?[^\]]+)\]", negate_class, value, count=3)
    if value_neg != value:
        candidates.append(("negate-class", value_neg))

    # quantifier swap + <-> *
    value1 = value.replace("+", "*")
    if value1 != value:
        candidates.append(("+->*", value1))
    value2 = value.replace("*", "+", 1)
    if value2 != value:
        candidates.append(("first-*->+", value2))

    # escape/un-escape grouping parens: \\( -> ( and ( -> \\(
    value3 = value.replace("\\)", ")").replace("\\(", "(")
    if value3 != value:
        candidates.append(("unescape-parens", value3))
    escaped = value.replace("(", "\\(").replace(")", "\\)")
    if escaped != value:
        candidates.append(("escape-parens", escaped))

    # char-class shortcut swap \d <-> \w, \s <-> \S, etc.
    for esc_from, esc_to in (("\\d", "\\w"), ("\\w", "\\d"), ("\\s", "\\S"), ("\\S", "\\s")):
        swapped = value.replace(esc_from, esc_to)
        if swapped != value:
            candidates.append((f"{esc_from}->{esc_to}", swapped))

    seen = set()
    for detail, replacement in candidates:
        if replacement == value or replacement in seen:
            continue
        seen.add(replacement)
        yield str(detail), cst.SimpleString(repr(replacement))


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