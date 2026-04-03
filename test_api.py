import httpx
import json

CHATBOT_API_URL = "https://hub-dev2.datacommons.cancer.gov/api/chat/question"


def ask_chatbot(
    question: str, session_id: str | None = None, conversation_history: list = None
) -> tuple[str, str]:
    """
    Hit the live chatbot API and collect the streamed response.
    Returns (answer_text, session_id)
    """
    payload = {
        "question": question,
        "sessionId": session_id,
        "conversationHistory": conversation_history or [],
    }

    answer_parts = []
    returned_session_id = session_id

    with httpx.stream("POST", CHATBOT_API_URL, json=payload, timeout=60.0) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") == "session":
                returned_session_id = event["sessionId"]
            elif event.get("type") == "response":
                answer_parts.append(event.get("output", ""))

    return "".join(answer_parts), returned_session_id


if __name__ == "__main__":
    answer, session_id = ask_chatbot(
        "What is the purpose of the program_external_url property in GC?"
    )
    print(f"Session: {session_id}")
    print(f"Answer: {answer}")
