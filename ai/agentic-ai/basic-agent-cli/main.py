from anthropic import Anthropic
from dotenv import load_dotenv
from anthropic.types import MessageParam


def add_user_message(messages: list[MessageParam], text: str) -> list[MessageParam]:
    return [*messages, {"role": "user", "content": text}]


def add_assistant_message(messages: list[MessageParam], text: str) -> list[MessageParam]:
    return [*messages, {"role": "assistant", "content": text}]


def chat(messages: list[MessageParam], client: Anthropic) -> str:
    system_prompt = """
    You are a very concise agent
    """

    params = {
        "model": "claude-haiku-4-5",
        "max_tokens": 1000,
        "messages": messages,
        "system": system_prompt,
    }

    with client.messages.stream(**params) as stream:
        for text in stream.text_stream:
            print(text, end="")
        print()
        return stream.get_final_message().content[0].text


def main() -> None:
    load_dotenv()
    client = Anthropic()
    messages: list[MessageParam] = []

    while True:
        user_input = input("> ")

        messages = add_user_message(messages, user_input)
        answer = chat(messages, client)
        messages = add_assistant_message(messages, answer)


if __name__ == "__main__":
    main()
