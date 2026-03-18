"""Coding Agent — SRP: code generation and iterative fixing.

Uses MCP dev tools to write, read, lint, and analyse code.
Tools are fetched from the MCP dev server — not bound inline.
"""
import os
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import GraphState
from src.config.llm import get_llm
from src.mcp.client import get_mcp_tools


def coding_agent_node(state: GraphState) -> dict:
    import asyncio
    return asyncio.run(_run_async(state))


async def _run_async(state: GraphState) -> dict:
    llm = get_llm()
    spec = state.get("spec", "")
    test_output = state.get("test_output", "")
    iteration_count = state.get("iteration_count", 0)

    workspace_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    )

    async with get_mcp_tools() as tools:
        dev_tools = tools["dev"]
        llm_with_tools = llm.bind_tools(dev_tools)

        prompt = f"Implement the complete codebase based on this specification:\n\n{spec}\n"
        if test_output:
            prompt += (
                f"\nYour last implementation failed tests with this output:\n"
                f"{test_output}\nPlease fix the failing code.\n"
            )

        messages = [
            SystemMessage(content=(
                "You are a senior Software Engineer. You write clean, testable code "
                "in whatever language the repository uses.\n"
                "Identify the primary language by inspecting existing files if unsure.\n"
                "Before writing any new code, ALWAYS follow this reasoning sequence:\n"
                f"1. Call `summarise_code_graph` on '{workspace_dir}' to understand the existing codebase.\n"
                "2. Call `query_code_graph` with a keyword from the spec to find related entities.\n"
                "3. Call `list_workspace_symbols` for a detailed symbol-level view of specific files.\n"
                "4. Only THEN create or edit files using the file management tools.\n"
                "5. After writing code, call `run_linter` to check for issues before finishing."
            )),
            HumanMessage(content=prompt)
        ]

        print(f"[ Coding Agent ] Iteration {iteration_count + 1} — Asking LLM to generate or fix code...")
        response = llm_with_tools.invoke(messages)

        # Execute tool calls returned by the LLM
        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"[ Coding Agent ] Executing {len(response.tool_calls)} tool call(s)...")
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"  -> Calling tool: {tool_name}")
                matched = next((t for t in dev_tools if t.name == tool_name), None)
                if matched:
                    await matched.ainvoke(tool_args)

    
    return {"iteration_count": iteration_count + 1}

