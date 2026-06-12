from __future__ import annotations

import inspect

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

from app.api import JudgeConfig


G_EVAL_NAME = "G-Eval"

G_EVAL_STEPS = [
    "Identifique o objetivo central do usuario a partir do campo 'input'.",
    "Verifique no 'actual output' se a resposta responde de forma clara e direta a esse objetivo, dando uma conclusao explicita quando a pergunta exige (por exemplo sim/nao, ha risco ou nao, abre janela de manejo ou nao).",
    "Use o 'expected output' como lista de criterios de avaliacao, nao como resposta fixa. Nao exija que o 'actual output' repita valores, horarios ou conclusoes de uma referencia historica.",
    "Nao penalize conclusao diferente da referencia quando o 'actual output' apresenta dados atuais coletados pela API SMART que justificam essa conclusao. Exemplo: se a referencia dizia calor alto, mas os dados atuais mostram temperatura moderada, umidade aceitavel e radiacao baixa, uma conclusao de baixo risco pode estar correta.",
    "Penalize apenas quando houver erro de unidade, conclusao tecnica incoerente com os dados apresentados, recomendacao inadequada, ausencia de conclusao explicita ou ausencia de dados essenciais para responder a pergunta.",
    "Avalie se a resposta esta fundamentada nos dados apresentados e se a recomendacao final e acionavel e tecnicamente justificada.",
    "Atribua nota mais alta quando o objetivo do usuario for atingido de forma correta, fundamentada e util, mesmo que os valores numericos atuais e a conclusao operacional sejam diferentes da referencia. Atribua nota baixa quando a resposta estiver vaga, tecnicamente incoerente, inventar limiares sem base ou ignorar criterios agronomicos essenciais do 'expected output'.",
]


def build_g_eval_metric(config: JudgeConfig) -> GEval:
    metric_kwargs = {
        "name": G_EVAL_NAME,
        "evaluation_steps": G_EVAL_STEPS,
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


def build_g_eval_case(
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
