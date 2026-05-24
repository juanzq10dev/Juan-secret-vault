import os
from anthropic import Anthropic
from anthropic.types import MessageParam


def add_user_message(messages: list[MessageParam], text: str) -> list[MessageParam]:
    return [*messages, {"role": "user", "content": text}]


def add_assistant_message(messages: list[MessageParam], text: str) -> list[MessageParam]:
    return [*messages, {"role": "assistant", "content": text}]


def chat(messages: list[MessageParam], client: Anthropic) -> str:
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=messages,
    )
    return message.content[0].text


def loop(messages: list[MessageParam], client: Anthropic) -> None:
    user_input = input("> ")
    print(">", user_input)

    messages = add_user_message(messages, user_input)
    answer = chat(messages, client)
    print(answer)

    loop(add_assistant_message(messages, answer), client)


def main() -> None:
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    loop([], client)


if __name__ == "__main__":
    main()
