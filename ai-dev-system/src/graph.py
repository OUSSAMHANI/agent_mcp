"""LangGraph StateGraph — wires all agent nodes into the full pipeline.

Graph shape
-----------

    Issue Scout ──► Spec Agent ──► Validator Agent ──► Coding Agent ──► Testing Agent ──► PR Agent ──► END
                         ▲                │                                    ▲               │
                         └── (invalid) ───┘                                    └─── (fail<3) ──┘
                                                                                               │
                                                                                            (fail≥3)
                                                                                               ▼
                                                                                             END
"""
from langgraph.graph import StateGraph, END

from src.state import GraphState
from src.agents.issue_scout.agent import issue_scout_node
from src.agents.spec_agent import spec_agent_node
from src.agents.validator_agent import validator_agent_node
from src.agents.coding_agent import coding_agent_node
from src.agents.testing_agent import testing_agent_node
from src.agents.pr_agent import pr_agent_node


def _route_issue_scout(state: GraphState) -> str:
    if not state.get("issue_number"):
        return END
    return "Spec Agent"


def _route_validator(state: GraphState) -> str:
    return "Coding Agent" if state.get("spec_feedback") == "VALID" else "Spec Agent"


def _route_testing(state: GraphState) -> str:
    if state.get("tests_passed", False):
        return "PR Agent"
    if state.get("iteration_count", 0) < 3:
        return "Coding Agent"
    return END


def build_graph():
    """Compile and return the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    workflow.add_node("Issue Scout",     issue_scout_node)
    workflow.add_node("Spec Agent",      spec_agent_node)
    workflow.add_node("Validator Agent", validator_agent_node)
    workflow.add_node("Coding Agent",    coding_agent_node)
    workflow.add_node("Testing Agent",   testing_agent_node)
    workflow.add_node("PR Agent",        pr_agent_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("Issue Scout")

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.add_conditional_edges("Issue Scout", _route_issue_scout)
    workflow.add_edge("Spec Agent",      "Validator Agent")
    workflow.add_conditional_edges("Validator Agent", _route_validator)
    workflow.add_edge("Coding Agent",    "Testing Agent")
    workflow.add_conditional_edges("Testing Agent",  _route_testing)
    workflow.add_edge("PR Agent",        END)

    return workflow.compile()


def build_graph_manual(ticket_text: str):
    """
    Build the same graph but skip Issue Scout and inject ticket_text
    directly — used by the manual POST /run endpoint.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("Spec Agent",      spec_agent_node)
    workflow.add_node("Validator Agent", validator_agent_node)
    workflow.add_node("Coding Agent",    coding_agent_node)
    workflow.add_node("Testing Agent",   testing_agent_node)
    workflow.add_node("PR Agent",        pr_agent_node)

    workflow.set_entry_point("Spec Agent")
    workflow.add_edge("Spec Agent",      "Validator Agent")
    workflow.add_conditional_edges("Validator Agent", _route_validator)
    workflow.add_edge("Coding Agent",    "Testing Agent")
    workflow.add_conditional_edges("Testing Agent",  _route_testing)
    workflow.add_edge("PR Agent",        END)

    return workflow.compile()
