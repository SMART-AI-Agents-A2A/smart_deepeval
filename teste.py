import json
from dotenv import load_dotenv
import os
import requests
from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase, Turn, ToolCall
from deepeval.metrics import GoalAccuracyMetric, TaskCompletionMetric
from deepeval.tracing import observe, update_current_span
from deepeval.dataset import EvaluationDataset, Golden

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONFIDENT_API_KEY = os.getenv("CONFIDENT_API_KEY")
BASE_URL = os.getenv("HONO_BASE_URL")
TOKEN_API = os.getenv("HONO_TOKEN")

os.environ["OPENAI_API_KEY"]    = OPENAI_API_KEY
os.environ["CONFIDENT_API_KEY"] = CONFIDENT_API_KEY

with open("perguntas.json", encoding="utf-8") as f:
    PERGUNTAS: list[str] = json.load(f)


def extrair_tools_usadas(dados: dict) -> list[ToolCall]:
    tools_set: set[str] = set()
    for agente in dados.get("agentResults", []):
        for nome in agente.get("evidence", {}).get("mcpTools", []):
            tools_set.add(nome)
    return [ToolCall(name=nome) for nome in tools_set]


def extrair_agentes_acionados(dados: dict) -> list[str]:
    vistos: set[str] = set()
    resultado = []
    for agente in dados.get("agentResults", []):
        nome = agente.get("agentName")
        if nome and nome not in vistos:
            vistos.add(nome)
            resultado.append(nome)
    return resultado


def chamar_chat(pergunta: str, conversation_id: str) -> tuple[str, dict]:
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/ai/chat",
            json={
                "conversationId": conversation_id,
                "messages": [{"role": "user", "content": pergunta}],
            },
            headers={"Authorization": f"Bearer {TOKEN_API}"},
            timeout=60,
        )
        resp.raise_for_status()
        dados = resp.json()
        return dados.get("response", "Sem resposta"), dados
    except Exception as e:
        print(f"[AVISO] Falha ao chamar chat: {e}")
        return "Erro", {}


@observe(type="agent", metrics=[TaskCompletionMetric(threshold=0.7, model="gpt-4o", include_reason=True)])
def agente_hono(pergunta: str, conversation_id: str) -> str:
    resposta, dados = chamar_chat(pergunta, conversation_id)
    tools = extrair_tools_usadas(dados)
    if tools:
        update_current_span(tools_called=tools, output=resposta)
    return resposta


print(f"Coletando respostas para {len(PERGUNTAS)} cenários...\n")

casos_goal: list[ConversationalTestCase] = []

for idx, pergunta in enumerate(PERGUNTAS):
    conversation_id = f"deepeval-session-{idx + 1}"
    print(f"  [{idx + 1}/{len(PERGUNTAS)}] {pergunta[:75]}")

    resposta, dados = chamar_chat(pergunta, conversation_id)

    tools_usadas = extrair_tools_usadas(dados)
    agentes      = extrair_agentes_acionados(dados)
    route        = dados.get("trace", {}).get("route", "desconhecido")

    print(f"         route={route} | agentes={agentes} | tools={[t.name for t in tools_usadas]}")

    casos_goal.append(
        ConversationalTestCase(
            scenario=(
                "Usuário consulta dados ambientais e meteorológicos da Fazenda NSAAB "
                "via orquestrador multi-agente com roteamento A2A/MCP."
            ),
            expected_outcome=(
                "O orquestrador deve acionar os agentes corretos, consultar as ferramentas "
                "MCP adequadas e retornar uma resposta precisa e fundamentada nos dados reais."
            ),
            turns=[
                Turn(role="user", content=pergunta),
                Turn(
                    role="assistant",
                    content=resposta,
                    tools_called=tools_usadas if tools_usadas else None,
                ),
            ],
        )
    )

dataset_task = EvaluationDataset(
    goldens=[Golden(input=p) for p in PERGUNTAS]
)

metric_goal = GoalAccuracyMetric(threshold=0.7, model="gpt-4o", include_reason=True)
metric_task = TaskCompletionMetric(threshold=0.7, model="gpt-4o", include_reason=True)

print("\n[1/2] Avaliando Goal Accuracy...")
try:
    evaluate(test_cases=casos_goal, metrics=[metric_goal])
except Exception as e:
    print(f"[ERRO] GoalAccuracy falhou: {e}")

print("\n[2/2] Avaliando Task Completion via tracing...")
try:
    for idx, golden in enumerate(dataset_task.evals_iterator(metrics=[metric_task])):
        agente_hono(golden.input, f"deepeval-task-{idx + 1}")
except Exception as e:
    print(f"[ERRO] TaskCompletion falhou: {e}")

print("\nAvaliação concluída.")