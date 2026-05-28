from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv(".env.local")
load_dotenv(".env")


def get_eval_model() -> str:
    return os.getenv("DEEPEVAL_EVALUATOR_MODEL", "gpt-4.1-mini")