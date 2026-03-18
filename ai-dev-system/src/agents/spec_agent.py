from src.state import GraphState
from src.mcp.token_counter import get_tracked_llm
from langchain_core.messages import SystemMessage, HumanMessage

def spec_agent_node(state: GraphState) -> dict:
    """
    Reads the development ticket text and generates a comprehensive technical specification.
    Also handles iterative feedback from the Validator Agent.
    """
    llm = get_tracked_llm("Spec Agent")
    ticket_text = state.get("ticket_text", "")
    feedback = state.get("spec_feedback", "")
    iteration_count = state.get("spec_iteration_count", 0)
    
    prompt = f"Please write a robust technical specification for this ticket:\n\n{ticket_text}\n"
    if feedback and feedback != "VALID":
        prompt += f"\nPrevious specification was rejected with this feedback. Please fix the spec:\n{feedback}"
        
    messages = [
        SystemMessage(content=(
            "You are an expert AI systems architect. Produce a clear, actionable technical specification. "
            "Examine the ticket and deduce the appropriate programming language and framework. "
            "Ensure your specification matches the conventions and tech stack of the repository."
        )),
        HumanMessage(content=prompt)
    ]
    
    print("[ Spec Agent ] Generating technical specification...")
    response = llm.invoke(messages)
    raw = response.content
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    print("[ Spec Agent ] Specification generated successfully.")
    return {
        "spec": str(raw),
        "spec_iteration_count": iteration_count + 1
    }
