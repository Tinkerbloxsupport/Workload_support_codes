from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


model = ChatOllama(
    model="qwen3",
    temperature=0,
    base_url="http://localhost:8081"
)

model_with_tools = model.bind_tools([get_weather])


def main():
    user_query = "What's the weather like in Boston?"

    response = model_with_tools.invoke(user_query)

    # Step 1: Print tool calls (if any)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            print(f"Tool: {tool_call['name']}")
            print(f"Args: {tool_call['args']}")

        # Step 2: Execute tools
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"\nTool called: {tool_name}")
            print(f"Args: {tool_args}")

            if tool_name == "get_weather":
                tool_result = get_weather.invoke(tool_args)

                # Step 3: Send tool result back
                final_response = model_with_tools.invoke([
                    HumanMessage(content=user_query),
                    response,
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"]
                    )
                ])

                print("\nFinal Answer:")
                print(final_response.content)
    else:
        print(response.content)


if __name__ == "__main__":
    main()
