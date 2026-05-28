from __future__ import annotations

import os

from dotenv import load_dotenv

from utils.cloudflare_gateway_eval_model import CloudflareGatewayEvalModel


load_dotenv(".env.local")
load_dotenv(".env")


def get_eval_model():
    provider = os.getenv("DEEPEVAL_EVALUATOR_PROVIDER", "openai").strip().lower()

    if provider == "cloudflare_gateway":
        return CloudflareGatewayEvalModel()

    return os.getenv("DEEPEVAL_EVALUATOR_MODEL", "gpt-4.1-mini")