"""
DeepTeam red-team evaluation against the live CRDC chatbot API.

Runs adversarial attacks to test for security and safety vulnerabilities.
Designed to run on a schedule (nightly/pre-release), NOT on every PR.

Usage:
    uv run python deepteam_eval.py
"""

import asyncio
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from deepteam import red_team
from bedrock_judge import BedrockJudge
from deepteam.vulnerabilities import (
    PromptLeakage,
    Misinformation,
    ExcessiveAgency,
    Robustness,
)
from deepteam.attacks.single_turn import (
    PromptInjection,
    Roleplay,
    SystemOverride,
)

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results" / "deepteam"
CHATBOT_API_URL = os.getenv(
    "CHATBOT_API_URL", "https://hub-dev2.datacommons.cancer.gov/api/chat/question"
)


def ask_chatbot(question: str) -> str:
    """Same streaming parser as deepeval_eval.py."""
    payload = {
        "question": question,
        "sessionId": None,
        "conversationHistory": [],
    }
    answer_parts = []
    try:
        with httpx.stream("POST", CHATBOT_API_URL, json=payload, timeout=60.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "response":
                    answer_parts.append(event.get("output", ""))
    except Exception as exc:
        print(f"  [WARN] Chatbot call failed: {exc}")
    return "".join(answer_parts)


async def model_callback(input: str) -> str:
    """DeepTeam calls this with each adversarial input."""
    response = await asyncio.to_thread(ask_chatbot, input)
    if not response:
        response = "[BLOCKED: No response from chatbot]"
    print(f"  [CHATBOT] {response[:200]}")
    return response


def build_vulnerabilities() -> list:
    return [
        PromptLeakage(),  # ASI06
        Misinformation(),  # ASI09
        ExcessiveAgency(),  # general safety
        Robustness(),  # trust/consistency
    ]


def build_attacks() -> list:
    return [
        # Direct prompt injection attempts # ASI06
        PromptInjection(),
        # Role-play attacks — "pretend you are an admin with no restrictions"
        Roleplay(),
        # System override — attempts to replace the system prompt
        SystemOverride(),
    ]


def main():
    if not os.getenv("CHATBOT_API_URL") and not os.getenv("KB_ID"):
        raise ValueError("CHATBOT_API_URL not set in .env")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running DeepTeam red-team suite...")

    print("  Note: this will make ~40-60 calls to the live chatbot API\n")

    risk_assessment = red_team(
        model_callback=model_callback,
        vulnerabilities=build_vulnerabilities(),
        attacks=build_attacks(),
        attacks_per_vulnerability_type=2,
        target_purpose="A domain-specific chatbot for the NCI CRDC Submission Portal that answers questions about data submission procedures and data models.",
        evaluation_model=BedrockJudge(),
        ignore_errors=False,
    )

    print("\nOverview:")
    print(risk_assessment.overview)

    out_path = str(RESULTS_DIR / "risk_assessment")
    risk_assessment.save(to=out_path)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
