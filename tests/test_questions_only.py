from __future__ import annotations

import pytest

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from utils.api_client import ask_smart_api
from utils.eval_model import get_eval_model
from utils.questions_loader import load_questions


EVAL_MODEL = get_eval_model()

QUESTIONS = load_questions()


answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=EVAL_MODEL,
    include_reason=True,
)


smart_quality = GEval(
    name="SMART Response Quality",
    criteria=(
        "Avalie se a resposta responde corretamente à pergunta enviada pelo usuário, "
        "é objetiva, útil, clara, não foge do tema, não inventa dados específicos "
        "e, quando aplicável, considera corretamente o contexto da fazenda Faz_NSAAB."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
    threshold=0.7,
    model=EVAL_MODEL,
)


@pytest.mark.parametrize("question", QUESTIONS)
def test_smart_response_quality(question: str) -> None:
    actual_output, _raw_json = ask_smart_api(question)

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
    )

    assert_test(
        test_case,
        [
            answer_relevancy,
            smart_quality,
        ],
    )