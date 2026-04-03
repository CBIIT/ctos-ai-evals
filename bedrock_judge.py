import asyncio
import os
import boto3
from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")

# Claude Sonnet 4.6 — different vendor from Nova Pro, eliminates same-LLM judge bias
JUDGE_MODEL_ID = "us.anthropic.claude-sonnet-4-6"


class BedrockJudge(DeepEvalBaseLLM):
    """DeepEval-compatible wrapper around AWS Bedrock converse() API."""

    def __init__(self):
        self.model_id = JUDGE_MODEL_ID
        self.client = self._build_client()

    def _build_client(self):
        # Exact same credential pattern as build_aws_clients() in generate_and_eval.py
        if os.environ.get("AWS_ACCESS_KEY_ID"):
            session = boto3.Session(region_name=AWS_REGION)
        else:
            session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        return session.client("bedrock-runtime", region_name=AWS_REGION)

    def load_model(self):
        # DeepEval calls this to get the "model object" — we just return our boto3 client
        return self.client

    def get_model_name(self) -> str:
        # Used by DeepEval in logs and reports
        return self.model_id

    def generate(self, prompt: str) -> str:
        # DeepEval calls this synchronously to get a judge response
        resp = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0},  # deterministic judging
        )
        return resp["output"]["message"]["content"][0]["text"]

    async def a_generate(self, prompt: str) -> str:
        # DeepEval calls this for async evaluation — we wrap the sync call
        return await asyncio.to_thread(self.generate, prompt)


if __name__ == "__main__":
    judge = BedrockJudge()
    print(f"Judge model: {judge.get_model_name()}")
    response = judge.generate("Say hello in one sentence.")
    print(f"Response: {response}")
