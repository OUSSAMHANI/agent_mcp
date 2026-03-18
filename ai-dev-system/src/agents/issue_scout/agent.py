"""Issue Scout Agent — SRP: GitHub issue discovery, assignment, and repo setup.

This agent is the autonomous entry point of the pipeline. It:
1. Lists open, unassigned issues on the configured GitHub repo.
2. Picks the first issue (LLM selects the most actionable one).
3. Self-assigns the issue so no parallel run picks the same ticket.
4. Clones / pulls the repository into workspace/.
5. Creates a fix branch named ``fix/issue-<N>-<slug>``.
6. Populates state with ticket_text, issue_number, branch_name, repo_url.
"""
import os
import re

from github import Github
from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.base import BaseAgentNode
from src.mcp.token_counter import get_tracked_llm
from src.state import GraphState
from src.mcp.client import get_mcp_tools


def _slugify(text: str, max_len: int = 40) -> str:
    """Convert a string to a URL-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len]


class IssueScooutAgent(BaseAgentNode):
    """Autonomous GitHub issue picker and repository bootstrapper."""

    def run(self, state: GraphState) -> dict:
        import asyncio
        return asyncio.run(self._run_async(state))

    async def _run_async(self, state: GraphState) -> dict:
        llm =  get_tracked_llm("Issue Scout")

        async with get_mcp_tools() as tools:
            # Only github tools needed for this agent
            github_tools = tools["github"]
            llm_with_tools = llm.bind_tools(github_tools)

            # ── Step 1: fetch issues ─────────────────────────────────────────
            print("[ Issue Scout ] Fetching open unassigned issues from GitHub...")
            # Find the list_open_issues tool and invoke it
            list_tool = next(t for t in github_tools if t.name == "list_open_issues")
            issues_text = await list_tool.ainvoke({"max_results": 10})

            if "No open" in issues_text:
                print("[ Issue Scout ] No open unassigned issues found. Stopping workflow.")
                return {
                    "ticket_text": "",
                    "issue_number": 0,
                    "branch_name": "",
                    "repo_url": "",
                }

            # ── Step 2: LLM picks the best issue ────────────────────────────
            pick_messages = [
                SystemMessage(content=(
                    "You are an autonomous developer agent. "
                    "From the list of open GitHub issues below, pick the single most "
                    "actionable and self-contained one for a coding fix. "
                    "Respond with ONLY the issue number as a plain integer, nothing else."
                )),
                HumanMessage(content=issues_text),
            ]
            print("[ Issue Scout ] Asking LLM to pick the best issue...")
            pick_response = llm.invoke(pick_messages)
            raw = pick_response.content
            if isinstance(raw, list):
                raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
            raw = str(raw).strip()
            match = re.search(r"\d+", raw)
            if not match:
                return {"ticket_text": "", "issue_number": 0, "branch_name": "", "repo_url": ""}

            issue_number = int(match.group())

            # ── Step 3: self-assign the issue ────────────────────────────────
            print(f"[ Issue Scout ] Selected issue #{issue_number}. Assigning to self...")
            assign_tool = next(t for t in github_tools if t.name == "assign_issue")
            await assign_tool.ainvoke({"issue_number": issue_number})

            # ── Step 4: fetch full issue body via PyGithub ───────────────────
            token = os.environ.get("GITHUB_TOKEN", "")
            repo_name = os.environ.get("GITHUB_REPOSITORY", "")
            gh = Github(token)
            repo = gh.get_repo(repo_name)
            issue = repo.get_issue(issue_number)
            ticket_text = f"#{issue.number} — {issue.title}\n\n{issue.body or ''}"
            repo_url = repo.clone_url
            slug = _slugify(issue.title)
            branch_name = f"fix/issue-{issue_number}-{slug}"

            # ── Step 5: clone / pull repo ────────────────────────────────────
            print(f"[ Issue Scout ] Cloning/pulling repository: {repo_url} ...")
            clone_tool = next(t for t in github_tools if t.name == "clone_or_pull_repo")
            await clone_tool.ainvoke({"repo_url": repo_url})

            # ── Step 6: create fix branch ────────────────────────────────────
            print(f"[ Issue Scout ] Creating local fix branch: {branch_name} ...")
            branch_tool = next(t for t in github_tools if t.name == "create_branch")
            await branch_tool.ainvoke({"branch_name": branch_name})

        return {
            "ticket_text": ticket_text,
            "issue_number": issue_number,
            "branch_name": branch_name,
            "repo_url": repo_url,
            "iteration_count": 0,
        }


# LangGraph node callable (used in graph.py)
issue_scout_node = IssueScooutAgent()