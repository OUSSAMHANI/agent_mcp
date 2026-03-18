"""Testing Agent — SRP: test script generation and sandboxed execution.

Uses MCP dev tools to run tests inside Docker sandbox.
Tools are fetched from the MCP dev server — not bound inline.
"""
import os
from langchain_core.messages import SystemMessage, HumanMessage

from src.state import GraphState
from src.mcp.token_counter import get_tracked_llm
from src.mcp.client import get_mcp_tools


def testing_agent_node(state: GraphState) -> dict:
    import asyncio
    return asyncio.run(_run_async(state))


async def _run_async(state: GraphState) -> dict:
    llm = get_tracked_llm("Testing Agent")
    ticket_text = state.get("ticket_text", "")

    workspace_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "workspace")
    )

    # ── Step 1: Generate test script ─────────────────────────────────────────
    print("[ Testing Agent ] Generating test script (script.sh)...")
    gen_messages = [
        SystemMessage(content=(
            "You are an expert test engineer. The user has an issue they want to test. "
            "Write a bash script that will test this codebase for the issue. "
            "If it's a Python project, it might run pytest or create test files and run them. "
            "Output ONLY the raw script content, no markdown code blocks or explanations."
        )),
        HumanMessage(content=f"Issue ticket:\n{ticket_text}")
    ]
    script_content = str(llm.invoke(gen_messages).content).strip()

    # Cleanup markdown block if the LLM adds it anyway
    if script_content.startswith("```"):
        lines = script_content.splitlines()
        script_content = "\n".join(
            lines[1:-1] if lines[-1].startswith("```") else lines[1:]
        )

    script_path = os.path.join(workspace_dir, "script.sh")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    # ── Step 2: Run sandbox via MCP tool ─────────────────────────────────────
    async with get_mcp_tools() as tools:
        dev_tools = tools["dev"]

        # Scope to only the sandbox tool — no extra schemas loaded
        sandbox_tool = next(
            (t for t in dev_tools if t.name == "run_tests_in_sandbox"), None
        )

        if not sandbox_tool:
            return {
                "test_output": "Error: run_tests_in_sandbox tool not found in MCP dev server.",
                "tests_passed": False,
            }

        print("[ Testing Agent ] Spinning up Docker sandbox to execute script.sh...")
        result = await sandbox_tool.ainvoke({"workspace_path": workspace_dir})

    # ── Step 3: LLM evaluates test output ────────────────────────────────────
    eval_messages = [
        SystemMessage(content=(
            "You are an independent tester in a software company. "
            "You just executed script.sh in the sandbox. "
            "Evaluate this output and respond with 'PASS' if the tests passed, "
            "or 'FAIL' if they didn't."
        )),
        HumanMessage(content=f"Sandbox script.sh output:\n{result}")
    ]

    eval_response = llm.invoke(eval_messages).content

    eval_str = str(eval_response).strip().upper()

    tests_passed = "PASS" in eval_str and "FAIL" not in eval_str

    if tests_passed:
        print("[ Testing Agent ] Result: PASS. Code is fully verified.")
    else:
        print("[ Testing Agent ] Result: FAIL. Tests did not pass.")

    return {
        "test_output": result,
        "tests_passed": tests_passed,
    }