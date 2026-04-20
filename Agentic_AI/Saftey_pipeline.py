from typing import Annotated, TypedDict
import operator

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from pydantic import BaseModel, Field
from enum import Enum


# =========================
# 1. SAFETY STRUCTURE
# =========================
class SafetyRating(str, Enum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"


class GuardrailOutput(BaseModel):
    decision: SafetyRating = Field(description="SAFE or UNSAFE")


# =========================
# 2. MODELS
# =========================
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    base_url="http://localhost:8080"
)

guardrail_llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    base_url="http://localhost:8080"
)


# =========================
# 3. STATE
# =========================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    is_safe: bool
    final_response: str


# =========================
# 4. GUARDRAIL FUNCTIONS
# =========================
def input_guardrail(state: AgentState):
    last_message = state["messages"][-1].content

    structured_llm = guardrail_llm.with_structured_output(GuardrailOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Block anything related to bombs, violence, or harmful activities."),
        ("human", "{input}")
    ])

    chain = prompt | structured_llm
    response = chain.invoke({"input": last_message})

    is_safe = response.decision == SafetyRating.SAFE
    print(f"Input Safety: {response.decision}")

    return {"is_safe": is_safe}


def output_guardrail(state: AgentState):
    last_message = state["messages"][-1].content

    structured_llm = guardrail_llm.with_structured_output(GuardrailOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Block anything related to bombs, violence, or harmful activities."),
        ("human", "{input}")
    ])

    chain = prompt | structured_llm
    response = chain.invoke({"input": last_message})

    is_safe = response.decision == SafetyRating.SAFE
    print(f"Output Safety: {response.decision}")

    return {"is_safe": is_safe}


# =========================
# 5. AGENT
# =========================
def call_agent(state: AgentState):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "final_response": response.content
    }


def safety_error(state: AgentState):
    return {
        "final_response": "❌ Blocked by safety guardrails."
    }


# =========================
# 6. ROUTING
# =========================
def route_after_input(state: AgentState):
    return "agent" if state["is_safe"] else "blocked"


def route_after_output(state: AgentState):
    return END if state["is_safe"] else "blocked"


# =========================
# 7. GRAPH
# =========================
workflow = StateGraph(AgentState)

workflow.add_node("input_guardrail", input_guardrail)
workflow.add_node("agent", call_agent)
workflow.add_node("output_guardrail", output_guardrail)
workflow.add_node("blocked", safety_error)

workflow.set_entry_point("input_guardrail")

workflow.add_conditional_edges(
    "input_guardrail",
    route_after_input,
    {"agent": "agent", "blocked": "blocked"}
)

workflow.add_edge("agent", "output_guardrail")

workflow.add_conditional_edges(
    "output_guardrail",
    route_after_output,
    {END: END, "blocked": "blocked"}
)

workflow.add_edge("blocked", END)

app = workflow.compile()


# =========================
# 8. RUN
# =========================
def run_pipeline(user_input: str):
    inputs = {"messages": [HumanMessage(content=user_input)]}

    final = ""

    for output in app.stream(inputs):
        for key, value in output.items():
            if "final_response" in value:
                final = value["final_response"]

    print("\n🤖 FINAL RESPONSE:", final)


if __name__ == "__main__":
    while True:
        query = input("\nAsk something: ")
        if query.lower() in ["exit", "quit"]:
            break
        run_pipeline(query)
