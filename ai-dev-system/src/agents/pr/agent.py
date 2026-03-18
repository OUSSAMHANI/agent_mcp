"""PR Agent — SRP: commit, push, and open a GitHub Pull Request."""
from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.base import BaseAgentNode
from src.mcp.token_counter import get_tracked_llm
from src.state import GraphState
from src.mcp.client import get_mcp_tools


class PRAgent(BaseAgentNode):
    """Commits and pushes the fix, then opens a GitHub Pull Request."""

    def run(self, state: GraphState) -> dict:
        import asyncio
        return asyncio.run(self._run_async(state))

    async def _run_async(self, state: GraphState) -> dict:
        llm = get_tracked_llm("PR Agent")
        ticket_text = state.get("ticket_text", "")
        issue_number = state.get("issue_number", 0)
        branch_name = state.get("branch_name", "fix/automated")

        async with get_mcp_tools() as tools:
            github_tools = tools["github"]
            llm_with_tools = llm.bind_tools(github_tools)

            # ── LLM drafts commit message and PR description ─────────────────
            draft_messages = [
                SystemMessage(content=(
                    "You are a CI/CD agent. Produce a concise git commit message and "
                    "a short GitHub PR description for the fix described below. "
                    "Format your response as:\n"
                    "COMMIT: <one-line commit message>\n"
                    "PR_BODY: <short markdown description, include 'Closes #<N>'>"
                )),
                HumanMessage(content=ticket_text),
            ]

            print("[ PR Agent ] Drafting commit message and Pull Request body...")
            raw = llm.invoke(draft_messages).content
            if isinstance(raw, list):
                raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
            draft = str(raw).strip()

            # Parse COMMIT and PR_BODY from the LLM output
            commit_msg = "fix: automated patch via AI Dev System"
            pr_body = f"Automated fix.\n\nCloses #{issue_number}"

            for line in draft.splitlines():
                if line.startswith("COMMIT:"):
                    commit_msg = line.replace("COMMIT:", "").strip()
                elif line.startswith("PR_BODY:"):
                    pr_body = line.replace("PR_BODY:", "").strip()

            # ── Push the branch ───────────────────────────────────────────────
            print(f"[ PR Agent ] Committing and pushing to branch: {branch_name} ...")
            push_tool = next(t for t in github_tools if t.name == "commit_and_push")
            await push_tool.ainvoke({
                "commit_message": commit_msg,
                "branch_name": branch_name,
            })

            # ── Open the PR ───────────────────────────────────────────────────
            pr_title = f"fix: {ticket_text.splitlines()[0][:72]}"
            print(f"[ PR Agent ] Opening GitHub Pull Request: '{pr_title}' ...")
            pr_tool = next(t for t in github_tools if t.name == "create_pull_request")
            pr_result = await pr_tool.ainvoke({
                "branch_name": branch_name,
                "title": pr_title,
                "body": pr_body,
            })

            # Extract URL from result string
            pr_url = ""
            if "http" in pr_result:
                for word in pr_result.split():
                    clean_url = word.strip("()[]{},;\"'")
                    if clean_url.startswith("http"):
                        pr_url = clean_url
                        break

        return {"pr_url": pr_url}


# LangGraph node callable
pr_agent_node = PRAgent()