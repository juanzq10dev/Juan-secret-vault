from datetime import datetime

from anthropic import Anthropic
from anthropic.types import ContentBlock, MessageParam, ParsedMessage
from dotenv import load_dotenv


def add_user_message(messages: list[MessageParam], text: str) -> list[MessageParam]:
    return [*messages, {"role": "user", "content": [{"type": "text", "text": text}]}]


def add_assistant_message(
    messages: list[MessageParam], content: list[ContentBlock]
) -> list[MessageParam]:
    return [*messages, {"role": "assistant", "content": content}]


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "get_current_time":
        return get_current_time()
    
    raise ValueError(f"Unknown tool: {name}")


TOOLS = [
    {"type": "web_search_20250305", "name": "web_search"},
    {
        "name": "get_current_time",
        "description": "Returns the current date and time.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

SYSTEM_PROMPT = "You are a very concise agent"


def stream_turn(messages: list[MessageParam], client: Anthropic) -> ParsedMessage:
    with client.messages.stream(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=messages,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
    ) as stream:
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "tool_use":
                    print(f"------Tool usage: {event.content_block.name}------")
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    print(event.delta.text, end="", flush=True)
        print()
        return stream.get_final_message()


def chat(messages: list[MessageParam], client: Anthropic) -> list[ContentBlock]:
    final = stream_turn(messages, client)

    if final.stop_reason != "tool_use":
        return final.content

    # Execute all tools: https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works
    tool_results = [
        {"type": "tool_result", "tool_use_id": block.id, "content": execute_tool(block.name, block.input)}
        for block in final.content
        if block.type == "tool_use"
    ]

    return chat(
        [*messages, {"role": "assistant", "content": final.content}, {"role": "user", "content": tool_results}],
        client,
    )


def main() -> None:
    load_dotenv()
    client = Anthropic()
    messages: list[MessageParam] = []

    while True:
        user_input = input("> ")

        if user_input.strip() == "/exit":
            break

        messages = add_user_message(messages, user_input)
        content = chat(messages, client)
        messages = add_assistant_message(messages, content)


if __name__ == "__main__":
    main()
