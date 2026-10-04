"""AST feature extraction for mutated nodes."""

from __future__ import annotations

import libcst as cst
from libcst.metadata import MetadataWrapper

from mutation_engine.mutators import CandidateVisitor


def node_metadata(
    source_text: str,
    line: int,
    mutation_type: str,
    detail: str | None = None,
) -> dict:
    """Return ``ast_depth`` and ``parent_node_type`` for a mutation target.

    Candidates are matched by line and mutation type; ``detail`` is ignored
    because MOR mutants are identified by their replacement operator, which is
    not part of the candidate's own metadata.
    """
    wrapper = MetadataWrapper(cst.parse_module(source_text))
    visitor = CandidateVisitor(wrapper.module, wrapper)
    wrapper.module.visit(visitor)
    module = wrapper.module
    for candidate in visitor.candidates:
        if candidate.line == line and candidate.mutation_type == mutation_type:
            depth, parent = _depth_and_parent(module, candidate.node)
            return {
                "ast_depth": depth,
                "parent_node_type": type(parent).__name__ if parent is not None else None,
            }
    raise ValueError(
        f"No {mutation_type} candidate at line {line} with detail {detail!r}"
    )


def _depth_and_parent(module: cst.Module, target: cst.CSTNode):
    state = {"depth": None, "parent": None, "ancestors": []}

    class _Visitor(cst.CSTVisitor):
        def on_visit(self, node):
            if node is target:
                state["depth"] = len(state["ancestors"])
                state["parent"] = (
                    state["ancestors"][-1] if state["ancestors"] else None
                )
                return False
            state["ancestors"].append(node)
            return True

        def on_leave(self, node):
            if state["ancestors"] and state["ancestors"][-1] is node:
                state["ancestors"].pop()

    module.visit(_Visitor())
    return state["depth"], state["parent"]