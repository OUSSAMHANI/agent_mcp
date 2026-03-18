"""Validator Agent — SRP: validate the technical specification against the codebase.

Uses MCP dev tools (AST + GraphRAG) to inspect the workspace and
flag naming collisions or missing implementation details.
"""
import os
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import GraphState
from src.config.llm import get_llm
from src.mcp.client import get_mcp_tools


def validator_agent_node(state: GraphState) -> dict:
    import asyncio
    return asyncio.run(_run_async(state))


async def _run_async(state: GraphState) -> dict:
    llm = get_llm()
    spec = state.get("spec", "")
    iteration_count = state.get("spec_iteration_count", 1)

    # Prevent infinite loops
    if iteration_count >= 3:
        print(f"[ Validator Agent ] Reached {iteration_count} iterations. Forcing VALID verdict.")
        return {"spec_feedback": "VALID"}

    workspace_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    )

    async with get_mcp_tools() as tools:
        # Validator only needs AST + GraphRAG tools — subset of dev tools
        dev_tools = tools["dev"]
        analysis_tools = [
            t for t in dev_tools
            if t.name in {
                "summarise_code_graph",
                "query_code_graph",
                "analyze_file_ast",
                "list_workspace_symbols",
            }
        ]
        llm_with_tools = llm.bind_tools(analysis_tools)

        # ── Pass 1: inspect workspace ────────────────────────────────────────
        inspect_messages = [
            SystemMessage(content=(
                "You are an expert technical reviewer with access to code-analysis tools.\n"
                "Before reviewing a specification, call `summarise_code_graph` on the workspace "
                "to understand what already exists, then call `query_code_graph` for key terms "
                "from the spec to check for naming collisions or missing dependencies."
            )),
            HumanMessage(content=(
                f"Workspace path: {workspace_dir}\n"
                f"Specification to review:\n{spec}\n\n"
                "Use your tools to inspect the workspace and gather context."
            ))
        ]

        print("[ Validator Agent ] Inspecting workspace via AST + GraphRAG tools...")
        inspection_response = llm_with_tools.invoke(inspect_messages)

        # Execute inspection tool calls
        tool_results: list[str] = []
        if hasattr(inspection_response, "tool_calls") and inspection_response.tool_calls:
            for tool_call in inspection_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                matched = next((t for t in analysis_tools if t.name == tool_name), None)
                if matched:
                    result = await matched.ainvoke(tool_args)
                    tool_results.append(f"[{tool_name} result]:\n{result}")
        else:
            tool_results = ["No workspace tools were called — proceeding with spec-only review."]

        # ── Pass 2: final verdict ────────────────────────────────────────────
        verdict_prompt = (
            "Review the technical specification below against the following criteria:\n"
            "1. Does it clearly identify which files need to be modified or created?\n"
            "2. Does it provide a high-level description of the logic or changes required?\n"
            "3. Does it avoid clear naming collisions based on the workspace analysis?\n\n"
            "If the specification meets all 3 criteria, respond with EXACTLY 'VALID'.\n"
            "Do not be overly pedantic. If a developer has enough direction to write the code, approve it.\n"
            "If it fundamentally fails a criterion, provide specific actionable feedback.\n\n"
            f"Specification to review:\n{spec}\n\n"
            f"Workspace analysis results:\n" + "\n".join(tool_results)
        )

        messages = [
            SystemMessage(content=(
                "You are a pragmatic technical reviewer. "
                "Your final answer must be either EXACTLY 'VALID', or specific actionable feedback. "
                "Err on the side of approval if the core architectural direction is clear."
            )),
            HumanMessage(content=verdict_prompt)
        ]

        print("[ Validator Agent ] Evaluating spec for final verdict...")
        response = llm.invoke(messages)
        raw = response.content
        if isinstance(raw, list):
            raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
        content = str(raw).strip()
        

    if content.upper().startswith("VALID"):
        return {"spec_feedback": "VALID"}
    return {"spec_feedback": content}