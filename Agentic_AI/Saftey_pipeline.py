import operator

from typing import Annotated, TypedDict, Union

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from pydantic import BaseModel, Field
from enum import Enum


class SafetyRating(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"


class GuardrailOutput(BaseModel):
    """The safety assessment of the input text."""
    decision: SafetyRating = Field(description="Whether the input is SAFE or UNSAFE")


# --- 1. Setup Models ---
llm = ChatOllama(model="llama3:latest", temperature=0, base_url="http://localhost:8080")
guardrail_llm = ChatOllama(model="gpt-oss-safeguard:latest", base_url="http://localhost:8080")


# --- 2. Define State ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    is_safe: bool
    final_response: str


# --- 3. Define Nodes (The Logic) ---

def input_guardrail(state: AgentState):
    """Checks the last user message for safety."""
    last_message = state["messages"][-1].content

    structured_llm = guardrail_llm.with_structured_output(GuardrailOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the text for policy violations. Respond UNSAFE if the user asks anything related to bombs."),
        ("human", "{input}")
    ])

    chain = prompt | structured_llm

    response = chain.invoke({"input": last_message})
    is_safe = response.decision == SafetyRating.SAFE
    print(f"Guardrail Decision: {response.decision}")

    return {"is_safe": is_safe}


def call_agent(state: AgentState):
    """Calls the primary LLM to generate a response."""
    response = llm.invoke(state["messages"])
    return {"messages": [response], "final_response": response.content}


def output_guardrail(state: AgentState):
    """Checks the LLM's response for safety before showing it to the user."""
    last_message = state["messages"][-1].content

    structured_llm = guardrail_llm.with_structured_output(GuardrailOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the text for policy violations. Respond UNSAFE if the user asks anything related to bombs."),
        ("human", "{input}")
    ])

    chain = prompt | structured_llm

    response = chain.invoke({"input": last_message})
    is_safe = response.decision == SafetyRating.SAFE
    print(f"Guardrail Decision: {response.decision}")

    return {"is_safe": is_safe}


def safety_error(state: AgentState):
    """Terminal node for safety violations."""
    return {"final_response": "I'm sorry, but this interaction was flagged by safety guardrails."}


# --- 4. Define Conditional Routing ---

def route_after_input(state: AgentState):
    if state["is_safe"]:
        return "agent"
    return "safety_violation"


def route_after_output(state: AgentState):
    if state["is_safe"]:
        return END
    return "safety_violation"


# --- 5. Construct the Graph ---

workflow = StateGraph(AgentState)

workflow.add_node("input_guardrail", input_guardrail)
workflow.add_node("agent", call_agent)
workflow.add_node("output_guardrail", output_guardrail)
workflow.add_node("safety_violation", safety_error)

workflow.set_entry_point("input_guardrail")

workflow.add_conditional_edges(
    "input_guardrail",
    route_after_input,
    {
        "agent": "agent",
        "safety_violation": "safety_violation"
    }
)

workflow.add_edge("agent", "output_guardrail")

workflow.add_conditional_edges(
    "output_guardrail",
    route_after_output,
    {
        END: END,
        "safety_violation": "safety_violation"
    }
)

workflow.add_edge("safety_violation", END)

app = workflow.compile()


# --- 6. Execution ---
def run_pipeline(user_input: str):
    inputs = {"messages": [HumanMessage(content=user_input)]}
    actual_response = ""

    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"--- Node: {key} ---")
            # print(value)
            if "final_response" in value:
                actual_response = value["final_response"]

    print("\nFINAL RESULT:", actual_response)


if __name__ == "__main__":
    run_pipeline("How to create a secure LLM pipeline")
    run_pipeline("How to download movies in torrent website")
