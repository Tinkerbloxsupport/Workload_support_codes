from typing import List, Literal, Optional, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os

load_dotenv()

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
# Ensure your Ollama server is running at this URL
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
    base_url="http://localhost:8080" # ollama port
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

    missing = [field.replace("_", " ") for field, value in factual_info.model_dump().items() if value is None]
    collected = [f"{field}: {value}" for field, value in factual_info.model_dump().items() if value is not None]

    prompt = f"""
You are a friendly AI interviewer. 
Your goal is to collect candidate details naturally.

Collected so far:
{", ".join(collected) if collected else "None"}

Missing fields:
{", ".join(missing)}

Recent Chat:
{format_chat(history[-2:])} 

Ask a polite question to collect 1 or 2 of the missing fields. Do not repeat questions already answered.
"""
    response = llm.invoke(prompt).content
    state["conversation_history"].append(AIMessage(content=response))
    return state

# =========================
# 6. USER INPUT
# =========================
def get_user_input(state: AgentState) -> AgentState:
    # 1. Get the last question asked by the AI
    last_ai_message = state["conversation_history"][-1].content
    print(f"\nAI: {last_ai_message}")

    # 2. Use the LLM to generate a fake response instead of calling input()
    simulation_prompt = f"""
    You are a candidate being interviewed for a job. 
    The interviewer just asked: "{last_ai_message}"
    
    Provide a brief, realistic answer. 
    If they asked for your name, say "John Doe". 
    If they asked for salary, say "100k".
    Keep it short.
    """
    
    # Generate the "User" response automatically
    simulated_response = llm.invoke(simulation_prompt).content
    print(f"User (Simulated): {simulated_response}")

    # 3. Append the simulated response to history
    state["conversation_history"].append(HumanMessage(content=simulated_response))
    state["turn_count"] += 1
    
    return state

# =========================
# 7. EXTRACT INFO
# =========================
def extract_information(state: AgentState) -> AgentState:
    history = state["conversation_history"]
    
    prompt = f"""
Extract personal details from the conversation. 
Current known data: {state['factual_info'].model_dump_json()}

Conversation:
{format_chat(history[-2:])}

Return the updated JSON schema.
"""
    response = llm_with_structured_output.invoke(prompt)

    # Update state by merging
    if response:
        for field, value in response.model_dump().items():
            if value is not None:
                setattr(state["factual_info"], field, value)

    return state

# =========================
# 8. CONTINUE OR END (THE FIX)
# =========================
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    # 1. Stop if user typed 'exit'
    if state["turn_count"] >= 100:
        return "end"

    # 2. Stop if all info is collected
    data = state["factual_info"]
    if all(value is not None for value in data.model_dump().values()):
        print("\n--- All information collected! ---")
        return "end"

    # 3. SET YOUR TURN LIMIT HERE
    # Change to 1 to test a single loop, or 10 for a full interview
    MAX_TURNS = 10 
    if state["turn_count"] >= MAX_TURNS:
        print(f"\n--- Reached turn limit ({MAX_TURNS}). Ending. ---")
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
    {"continue": "decide_next_question", "end": END}
)

app = workflow.compile()

# =========================
# 10. RUN
# =========================
if __name__ == "__main__":
    initial_state = {
        "conversation_history": [],
        "factual_info": FactualInformation(),
        "turn_count": 0
    }

    final_state = app.invoke(initial_state)

    print("\n" + "="*30)
    print("FINAL COLLECTED DATA")
    print("="*30)
    for key, value in final_state["factual_info"].model_dump().items():
        print(f"{key:20}: {value}")
