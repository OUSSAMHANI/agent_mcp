"""Shared MCP client — single place to configure both MCP servers.

Usage in agents:
    from src.mcp.client import get_mcp_tools

    async with get_mcp_tools() as tools:
        github_tools = tools["github"]
        dev_tools    = tools["dev"]
        all_tools    = tools["all"]
"""
import os
from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient


def _server_config() -> dict:
    """Build the MultiServerMCPClient config for both MCP servers."""
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    return {
        "github": {
            "command": "python",
            "args": ["-m", "src.mcp.servers.github_server"],
            "cwd": root,
            "transport": "stdio",
        },
        "dev": {
            "command": "python",
            "args": ["-m", "src.mcp.servers.dev_server"],
            "cwd": root,
            "transport": "stdio",
        },
    }


@asynccontextmanager
async def get_mcp_tools():
    """
    Async context manager that spins up both MCP servers and yields
    a dict with pre-grouped tool lists.

    Yields:
        {
            "github": [...],   ← tools from github_server
            "dev":    [...],   ← tools from dev_server
            "all":    [...],   ← all tools combined
        }
    """
    async with MultiServerMCPClient(_server_config()) as client:
        github_tools = client.get_tools(server_name="github")
        dev_tools    = client.get_tools(server_name="dev")
        yield {
            "github": github_tools,
            "dev":    dev_tools,
            "all":    github_tools + dev_tools,
        }