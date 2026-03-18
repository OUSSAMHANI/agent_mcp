"""Dev Tools MCP Server — exposes coding, analysis, and search tools via FastMCP.

Run as a standalone process:
    python -m src.mcp.servers.dev_server

All tools are imported directly from existing src/tools modules —
no logic is duplicated here, this is purely a re-exposure layer.
"""
import os
from fastmcp import FastMCP
from src.tools.linter.linter import run_linter
from src.tools.ast_analysis.tools import analyze_file_ast, list_workspace_symbols
from src.tools.graph_rag.tools import query_code_graph, summarise_code_graph
from src.tools.docker.sandbox import run_tests_in_sandbox
from src.tools.search.search import get_search_tools
from src.tools.files.file_tool import get_file_tools

mcp = FastMCP("dev-server")

# ── Static tools (single @tool decorated functions) ──────────────────────────
mcp.add_tool(run_linter)
mcp.add_tool(analyze_file_ast)
mcp.add_tool(list_workspace_symbols)
mcp.add_tool(query_code_graph)
mcp.add_tool(summarise_code_graph)
mcp.add_tool(run_tests_in_sandbox)

# ── Dynamic tools (returned from factory functions) ──────────────────────────
# Search
for tool in get_search_tools():
    mcp.add_tool(tool)

# File management — rooted at workspace/
_workspace = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "workspace")
)
for tool in get_file_tools(_workspace):
    mcp.add_tool(tool)

if __name__ == "__main__":
    mcp.run(transport="stdio")