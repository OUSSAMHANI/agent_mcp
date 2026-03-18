"""GitHub MCP Server — exposes GitHub + Git tools via FastMCP.

Run as a standalone process:
    python -m src.mcp.servers.github_server

All tools are imported directly from existing src/tools/github modules —
no logic is duplicated here, this is purely a re-exposure layer.
"""
from fastmcp import FastMCP
from src.tools.github.issue_tools import list_open_issues, assign_issue
from src.tools.github.git_tools import clone_or_pull_repo, create_branch, commit_and_push
from src.tools.github.pr_tools import create_pull_request

mcp = FastMCP("github-server")

# ── Register all GitHub + Git tools ─────────────────────────────────────────
mcp.add_tool(list_open_issues)
mcp.add_tool(assign_issue)
mcp.add_tool(clone_or_pull_repo)
mcp.add_tool(create_branch)
mcp.add_tool(commit_and_push)
mcp.add_tool(create_pull_request)

if __name__ == "__main__":
    mcp.run(transport="stdio")