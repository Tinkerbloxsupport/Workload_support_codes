from typing import List, Literal, Optional, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()
import os


# =========================
# 1. SCHEMA (DATA MODEL)
# =========================
class FactualInformation(BaseModel):
    full_name: Optional[str] = Field(None)
    email: Optional[str] = Field(None)
    phone_number: Optional[str] = Field(None)
    current_salary_range: Optional[str] = Field(None)
    desired_salary_range: Optional[str] = Field(None)
    current_job_title: Optional[str] = Field(None)
    current_location: Optional[str] = Field(None)
    willing_to_relocate: Optional[bool] = Field(None)
    notice_period_days: Optional[int] = Field(None)


# =========================
# 2. STATE
# =========================
class AgentState(TypedDict):
    conversation_history: List[BaseMessage]
    factual_info: FactualInformation
    turn_count: int


# =========================
# 3. LLM SETUP
# =========================
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    base_url="http://localhost:8080"
)

llm_with_structured_output = llm.with_structured_output(FactualInformation)


# =========================
# 4. HELPER FUNCTION
# =========================
def format_chat(history):
    return "\n".join([
        f"{'USER' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
        for msg in history
    ])


# =========================
# 5. DECIDE NEXT QUESTION
# =========================
def decide_next_question(state: AgentState) -> AgentState:
    factual_info = state["factual_info"]
    history = state["conversation_history"]

    missing = []
    collected = []

    for field, value in factual_info.model_dump().items():
        if value is None:
            missing.append(field.replace("_", " "))
        else:
            collected.append(f"{field}: {value}")

    prompt = f"""
You are a friendly AI interviewer.

Chat History:
{format_chat(history)}

Collected Info:
{chr(10).join(collected)}

Missing Info:
{chr(10).join(missing)}

Ask a natural, conversational question to collect missing details.
You can ask for 1-2 fields together.
"""

    response = llm.invoke(prompt).content
    state["conversation_history"].append(AIMessage(content=response))
    return state


# =========================
# 6. USER INPUT
# =========================
def get_user_input(state: AgentState) -> AgentState:
    print("\n" + format_chat(state["conversation_history"]))

    user_input = input("\nUser: ").strip()

    # Exit condition
    if user_input.lower() in ["exit", "quit"]:
        print("\nExiting interview...")
        return state

    state["conversation_history"].append(HumanMessage(content=user_input))
    state["turn_count"] += 1
    return state


# =========================
# 7. EXTRACT INFO (MERGE FIX)
# =========================
def extract_information(state: AgentState) -> AgentState:
    history = state["conversation_history"]

    prompt = f"""
Extract structured information from this conversation.
If a field is not found, leave it as None.

Conversation:
{format_chat(history)}
"""

    response = llm_with_structured_output.invoke(prompt)

    # 🔥 Merge instead of overwrite
    current_data = state["factual_info"]

    for field, value in response.model_dump().items():
        if value is not None:
            setattr(current_data, field, value)

    state["factual_info"] = current_data
    return state


# =========================
# 8. CONTINUE OR END
# =========================
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    data = state["factual_info"]

    # Stop if all fields filled
    if all(value is not None for value in data.model_dump().values()):
        return "end"

    # Safety stop (avoid infinite loop)
    if state["turn_count"] > 20:
        print("\nReached max turns. Ending interview.")
        return "end"

    return "continue"


# =========================
# 9. BUILD WORKFLOW
# =========================
workflow = StateGraph(AgentState)

workflow.add_node("decide_next_question", decide_next_question)
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("extract_information", extract_information)

workflow.set_entry_point("decide_next_question")

workflow.add_edge("decide_next_question", "get_user_input")
workflow.add_edge("get_user_input", "extract_information")

workflow.add_conditional_edges(
    "extract_information",
    should_continue,
    {
        "continue": "decide_next_question",
        "end": END,
    },
)

app = workflow.compile()


# =========================
# 10. RUN
# =========================
if __name__ == "__main__":

    state = {
        "conversation_history": [],
        "factual_info": FactualInformation(),
        "turn_count": 0
    }

    final_state = app.invoke(state)

    print("\n\n===== FINAL RESULT =====\n")

    for key, value in final_state["factual_info"].model_dump().items():
        print(f"{key}: {value}")

    # Save to CSV (ATS usage)
    try:
        import pandas as pd
        df = pd.DataFrame([final_state["factual_info"].model_dump()])
        df.to_csv("candidates.csv", mode="a", header=not os.path.exists("candidates.csv"), index=False)
        print("\nSaved to candidates.csv ✅")
    except Exception as e:
        print("\nCSV save failed:", e)

    # Save graph + chat log
    with open("readme.txt", "w") as f:
        f.write(app.get_graph().draw_ascii())
        f.write("\n\nChat History:\n")
        f.write(format_chat(final_state["conversation_history"]))
