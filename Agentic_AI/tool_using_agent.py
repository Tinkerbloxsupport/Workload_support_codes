from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage


# =========================
# 1. DEFINE TOOL
# =========================
@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


# =========================
# 2. SETUP MODEL
# =========================
llm = ChatOllama(
    model="llama3.1",                     # your pulled model
    temperature=0,
    base_url="http://localhost:8080"      # your running Docker port
)

# Bind tool to model
llm_with_tools = llm.bind_tools([get_weather])


# =========================
# 3. MAIN FUNCTION
# =========================
def main():
    print("=== Tool Calling Agent Started ===\n")

    user_query = input("Ask something: ")

    # Step 1: Ask model
    response = llm_with_tools.invoke(user_query)

    # Step 2: Check if tool is called
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"\n🔧 Tool called: {tool_name}")
            print(f"📦 Args: {tool_args}")

            # Step 3: Execute tool
            if tool_name == "get_weather":
                tool_result = get_weather.invoke(tool_args)

                print(f"🛠 Tool Result: {tool_result}")

                # Step 4: Send tool result back to model
                final_response = llm_with_tools.invoke([
                    HumanMessage(content=user_query),
                    response,
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    )
                ])

                print("\n🤖 Final Answer:")
                print(final_response.content)

    else:
        # If no tool is used
        print("\n🤖 Direct Answer:")
        print(response.content)


# =========================
# 4. RUN
# =========================
if __name__ == "__main__":
    main()
