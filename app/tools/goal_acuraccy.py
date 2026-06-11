from __future__ import annotations

import inspect

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

from app.api import JudgeConfig


GOAL_ACCURACY_NAME = "Goal Accuracy"

GOAL_ACCURACY_STEPS = [
    "Identifique o objetivo central do usuario a partir do campo 'input'.",
    "Verifique no 'actual output' se a resposta responde de forma clara e direta a esse objetivo, dando uma conclusao explicita quando a pergunta exige (por exemplo sim/nao, ha risco ou nao, abre janela de manejo ou nao).",
    "Compare o 'actual output' com o 'expected output' quanto aos criterios tecnicos esperados, mas nao penalize diferencas numericas quando o 'actual output' estiver claramente baseado em dados atuais coletados pela API SMART. Penalize apenas quando houver erro de unidade, conclusao tecnica incoerente, recomendacao inadequada ou ausencia de dados essenciais para responder a pergunta.",
    "Avalie se a resposta esta fundamentada nos dados apresentados e se a recomendacao final e acionavel e tecnicamente justificada.",
    "Atribua nota mais alta quando o objetivo do usuario for atingido de forma correta, fundamentada e util, mesmo que os valores numericos atuais sejam diferentes da referencia. Atribua nota baixa quando a conclusao estiver ausente, vaga, tecnicamente incoerente ou quando a resposta ignorar criterios agronomicos essenciais do 'expected output'.",
]


def build_goal_accuracy_metric(config: JudgeConfig) -> GEval:
    metric_kwargs = {
        "name": GOAL_ACCURACY_NAME,
        "evaluation_steps": GOAL_ACCURACY_STEPS,
        "evaluation_params": [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        "threshold": config.threshold,
        "model": config.model,
    }

    geval_params = inspect.signature(GEval).parameters
    if "async_mode" in geval_params:
        metric_kwargs["async_mode"] = False

    return GEval(**metric_kwargs)


def build_goal_accuracy_case(
    question: str,
    answer: str,
    expected_output: str,
    tools_called: list[ToolCall],
) -> LLMTestCase:
    case_kwargs = {
        "input": question,
        "actual_output": answer,
        "expected_output": expected_output,
    }
    if tools_called:
        case_kwargs["tools_called"] = tools_called

    return LLMTestCase(**case_kwargs)