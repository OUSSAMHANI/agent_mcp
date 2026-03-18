"""Workflow routes — SRP: HTTP request/response handling only."""
import asyncio
import os
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.schemas.workflow import RunResponse, TicketRequest
from src.graph import build_graph, build_graph_manual
from src.mcp.token_counter import reset_token_log, get_token_report

router = APIRouter(prefix="/run", tags=["Workflow"])


def _base_state(ticket_text: str = "") -> dict:
    os.makedirs("workspace", exist_ok=True)
    return {
        "ticket_text": ticket_text,
        "issue_number": 0,
        "branch_name": "",
        "repo_url": "",
        "spec": "",
        "spec_feedback": "",
        "spec_iteration_count": 0,
        "test_output": "",
        "tests_passed": False,
        "pr_url": "",
        "iteration_count": 0,
    }


def _extract_final(outputs: list[dict]) -> RunResponse:
    final: dict = {}
    for o in outputs:
        node_name = list(o.keys())[0]
        print(f"[pipeline] Node complete: {node_name}")
        for v in o.values():
            final.update(v)
    print("[pipeline] Run finished.")
    print(get_token_report())
    return RunResponse(
        spec=final.get("spec", ""),
        spec_feedback=final.get("spec_feedback", ""),
        test_output=final.get("test_output", ""),
        tests_passed=final.get("tests_passed", False),
        pr_url=final.get("pr_url", ""),
        iteration_count=final.get("iteration_count", 0),
    )


async def _sse_stream(graph, initial_state: dict) -> AsyncGenerator[str, None]:
    for output in graph.stream(initial_state):
        node_name = list(output.keys())[0]
        print(f"[pipeline] Node stream: {node_name}")
        yield f"data: [node:{node_name}] {output[node_name]}\n\n"
        await asyncio.sleep(0)
    print(get_token_report())
    print("[pipeline] Streaming run finished.")


@router.post("", response_model=RunResponse)
async def run_manual(request: TicketRequest) -> RunResponse:
    if not request.ticket_text.strip():
        raise HTTPException(status_code=422, detail="ticket_text must not be empty.")
    reset_token_log()
    graph = build_graph_manual(request.ticket_text)
    outputs = list(graph.stream(_base_state(request.ticket_text)))
    return _extract_final(outputs)


@router.post("/auto", response_model=RunResponse)
async def run_auto() -> RunResponse:
    reset_token_log()
    graph = build_graph()
    outputs = list(graph.stream(_base_state()))
    return _extract_final(outputs)


@router.post("/stream")
async def stream_manual(request: TicketRequest) -> StreamingResponse:
    if not request.ticket_text.strip():
        raise HTTPException(status_code=422, detail="ticket_text must not be empty.")
    reset_token_log()
    graph = build_graph_manual(request.ticket_text)
    return StreamingResponse(
        _sse_stream(graph, _base_state(request.ticket_text)),
        media_type="text/event-stream",
    )


@router.post("/auto/stream")
async def stream_auto() -> StreamingResponse:
    reset_token_log()
    graph = build_graph()
    return StreamingResponse(
        _sse_stream(graph, _base_state()),
        media_type="text/event-stream",
    )
