from typing import List, Literal, Optional, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()
import os

class FactualInformation(BaseModel):
    """Schema for factual and resume-based information."""
    full_name: Optional[str] = Field(None, description="The candidate's full name.")
    email: Optional[str] = Field(None, description="The candidate's primary email address.")
    phone_number: Optional[str] = Field(None, description="The candidate's phone number.")
    current_salary_range: Optional[str] = Field(None, description="The candidate's current salary range.")
    desired_salary_range: Optional[str] = Field(None, description="The candidate's desired salary range.")
    current_job_title: Optional[str] = Field(None, description="The candidate's current job title.")
    current_location: Optional[str] = Field(None, description="The candidate's current location.")
    willing_to_relocate: Optional[bool] = Field(None, description="True if the candidate is willing to relocate to Bangalore.")
    notice_period_days: Optional[int] = Field(None, description="Number of days required for notice period, if applicable.")

class AgentState(TypedDict):
    conversation_history: List[BaseMessage]
    factual_info: FactualInformation

llm = ChatOllama(model="llama3.1", temperature=0, base_url="http://localhost:8080") # gpt-4o for its strong function calling and reasoning

llm_with_structured_output = llm.with_structured_output(FactualInformation)


def decide_next_question(state: AgentState) -> AgentState:
    """
    Decides the next question to ask based on the current state of factual_info.
    This also handles the initial greeting and identifies missing fields.
    """
    factual_info = state["factual_info"]
    conversation_history = state["conversation_history"]

    # Identify missing fields
    missing_fields = []
    information_gathered_fields = []
    if factual_info.full_name is None: 
        missing_fields.append("full name")
    else:
        information_gathered_fields.append(f"full name: {factual_info.full_name}")
    if factual_info.email is None: 
        missing_fields.append("email address")
    else:
        information_gathered_fields.append(f"email address: {factual_info.email}")
    if factual_info.phone_number is None: 
        missing_fields.append("phone number")
    else:
        information_gathered_fields.append(f"phone number: {factual_info.phone_number}")
    if factual_info.current_salary_range is None: 
        missing_fields.append("current salary range")
    else:
        information_gathered_fields.append(f"current salary range: {factual_info.current_salary_range}")
    if factual_info.desired_salary_range is None: 
        missing_fields.append("desired salary range")
    else:
        information_gathered_fields.append(f"desired salary range: {factual_info.desired_salary_range}")
    if factual_info.current_job_title is None: 
        missing_fields.append("current job title")
    else:
        information_gathered_fields.append(f"current job title: {factual_info.current_job_title}")
    if factual_info.current_location is None: 
        missing_fields.append("current location")
    else:
        information_gathered_fields.append(f"current location: {factual_info.current_location}")
    if factual_info.willing_to_relocate is None: 
        missing_fields.append("willingness to relocate to Bangalore (Yes/No)")
    else:
        information_gathered_fields.append(f"willingness to relocate to Bangalore (Yes/No): {factual_info.willing_to_relocate}")
    if factual_info.notice_period_days is None: 
        missing_fields.append("notice period in days")
    else:
        information_gathered_fields.append(f"notice period in days: {factual_info.notice_period_days}")
    chat_history = "\n".join([f"{"USER" if isinstance(msg, HumanMessage) else "AI"}: {msg.content}" for msg in conversation_history])

    information_gathered_fields = "\n".join(information_gathered_fields)
    missing_fields = "\n".join(missing_fields)
    prompt = f"""**You are an AI assistant conducting a job application interview.**

**Context:**

* **Chat History:**
  {chat_history}
* **Information Already Collected:**
  {information_gathered_fields}
* **Missing Information:**
  {missing_fields}

**Task:**
Based on the chat history and the information already gathered, generate the next natural question to ask the candidate that helps collect one or more of the missing fields. The question should feel like a smooth continuation of the conversation.
"""
    next_question = llm.invoke(prompt).content
    state["conversation_history"].append(AIMessage(content=next_question))

    return state

def get_user_input(state: AgentState) -> AgentState:
    chat_history = "\n".join([f"{"USER" if isinstance(msg, HumanMessage) else "AI"}: {msg.content}" for msg in state["conversation_history"]])
    print(chat_history)

    user_input = ""
    factual_info = state["factual_info"]
    if factual_info.full_name is None: 
        user_input = "My name is Sivaji"
    elif factual_info.email is None: 
        user_input = "My email is kumarsiva73457@gmail.com"
    elif factual_info.phone_number is None: 
        user_input = "My phone number is 8234521389"
    elif factual_info.current_salary_range is None: 
        user_input = "current_salary_range is 20k-30k"
    elif factual_info.desired_salary_range is None: 
        user_input = "My desired_salary_range is 50k-100k"
    elif factual_info.current_job_title is None: 
        user_input = "My current_job_title is AI engineer"
    elif factual_info.current_location is None: 
        user_input = "My current_location is Chennai"
    elif factual_info.willing_to_relocate is None: 
        user_input = "My willing_to_relocate is True"
    elif factual_info.notice_period_days is None: 
        user_input = "My notice_period_days is 60"


    state["conversation_history"].append(HumanMessage(content=user_input))
    return state


def extract_information(state: AgentState) -> AgentState:
    """
    Calls the LLM to process the latest user input and potentially extract information.
    """
    conversation_history = state["conversation_history"]

    chat_history = "\n".join([f"{"USER" if isinstance(msg, HumanMessage) else "AI"}: {msg.content}" for msg in conversation_history])
    prompt = f"You are an AI assistant who is expert in extracting information from a conversation. If you cannot find any information in the given conversation, you can leave the field with None value.\nExtract relevant information from the given conversation history:\n\n{chat_history}\n"
    response = llm_with_structured_output.invoke(prompt)
    state["factual_info"] = response
    return state


def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """
    Determines whether the agent should continue asking questions or stop.
    Stops when all fields in factual_info are populated.
    """
    factual_info = state["factual_info"]


    # Check if all fields are populated
    all_fields_populated = all(
        getattr(factual_info, field) is not None
        for field in FactualInformation.model_fields.keys()
    )
    if all_fields_populated:
        return "end"
    else:
        return "continue"


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


if __name__ == "__main__":
    # if not os.getenv("OPENAI_API_KEY"):
    #     api_key = input("Please set OPENAI_API_KEY environment variable.\n")
    #     os.environ["OPENAI_API_KEY"] = api_key

    current_state = {
        "conversation_history": [],
        "factual_info": FactualInformation()
    }

    final_state = app.invoke(current_state)

    chat_history = "\n".join([f"{"USER" if isinstance(msg, HumanMessage) else "AI"}: {msg.content}" for msg in final_state["conversation_history"]])
    with open('readme.txt', 'w') as f:
        f.write(app.get_graph().draw_ascii())
        f.write("\n\nChat History:\n")
        f.write(chat_history)
        f.write("\n\nFactual Information Collected:\n")
        for field, value in final_state["factual_info"].model_dump().items():
            f.write(f"{field}: {value}\n")
