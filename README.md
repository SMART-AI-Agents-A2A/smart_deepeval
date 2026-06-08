# Avaliacao da API SMART com DeepEval

## Objetivo

Este projeto avalia a API SMART, implementada em Hono, usando o framework Python DeepEval.

A avaliacao tem como foco verificar se o orquestrador multiagente da SMART:

- entende a pergunta agricola do usuario;
- aciona os agentes e ferramentas MCP corretos;
- usa os dados retornados pelas fontes disponiveis;
- entrega uma resposta util, fundamentada e adequada para tomada de decisao no cafezal.

O dataset usado como base fica em `app/db/db.json` e contem perguntas, respostas esperadas e justificativas tecnicas de referencia.

## Fluxo Avaliado

O fluxo executado pela avaliacao e:

1. O script Python carrega perguntas do dataset.
2. Cada pergunta e enviada para a rota de chat da API SMART.
3. A API SMART responde usando o orquestrador, agentes e ferramentas MCP do proprio projeto SMART.
4. O DeepEval avalia a resposta usando tres metricas agenticas:
   - Goal Accuracy;
   - Tool Correctness;
   - Task Completion.
5. Opcionalmente, os resultados coletados podem ser exportados em CSV para analise posterior em Google Colab.

## Metricas Usadas

### Goal Accuracy

**Finalidade:** avaliar se o agente conseguiu atingir o objetivo do usuario.

No DeepEval, `GoalAccuracyMetric` e uma metrica agentica para conversas. Ela avalia a capacidade do agente de identificar o objetivo, planejar e executar uma resposta que alcance esse objetivo.

Neste projeto, a metrica recebe:

- a pergunta do usuario;
- a resposta produzida pela API SMART;
- as ferramentas chamadas pelo agente;
- uma resposta esperada de referencia.

**O que ela mede aqui:** se a resposta final da SMART atende ao objetivo agricola da pergunta, considerando o contexto esperado.

**Limitacao:** a avaliacao atual usa uma conversa curta, com um turno de usuario e um turno de assistente. Isso e valido para avaliar o comportamento final, mas nao representa uma conversa longa com multiplas interacoes.

Documentacao: https://deepeval.com/docs/metrics-goal-accuracy

### Tool Correctness

**Finalidade:** avaliar se o agente chamou as ferramentas corretas.

No DeepEval, `ToolCorrectnessMetric` compara:

- `tools_called`: ferramentas realmente chamadas pelo agente;
- `expected_tools`: ferramentas esperadas para aquela pergunta.

Neste projeto, as ferramentas chamadas sao extraidas do payload da API SMART ou inferidas pelos agentes acionados. As ferramentas esperadas podem vir do dataset ou ser inferidas por palavras-chave da pergunta e da resposta esperada.

**O que ela mede aqui:** se a API SMART acionou ferramentas coerentes com a pergunta, por exemplo:

- temperatura do ar;
- umidade do ar;
- previsao de chuva;
- velocidade do vento;
- direcao do vento;
- rajadas.

**Limitacao:** a avaliacao atual valida principalmente o nome das ferramentas. Ela ainda nao valida rigorosamente os parametros usados em cada chamada, nem o conteudo retornado por cada ferramenta.

Documentacao: https://deepeval.com/docs/metrics-tool-correctness

### Task Completion

**Finalidade:** avaliar se o agente concluiu a tarefa solicitada.

No DeepEval, `TaskCompletionMetric` usa tracing para avaliar se o agente completou a tarefa. Neste projeto, a chamada para a API SMART foi encapsulada com `@observe`, e o span recebe:

- input;
- output;
- expected_output;
- ferramentas chamadas.

**O que ela mede aqui:** se a resposta da API SMART resolveu a pergunta do usuario de ponta a ponta.

**Limitacao:** como a API SMART roda fora do processo Python, o DeepEval nao enxerga cada subetapa interna como spans separados. Ele avalia a chamada observada como um todo, com base no input, output e metadados coletados.

Documentacao: https://deepeval.com/docs/metrics-task-completion

## Resultado Inicial

Foi executado um teste com uma pergunta do dataset:

```text
Considerando que os pes de cafe da Fazenda NSAAB estao em periodo de florada ou pre-florada, as condicoes climaticas de hoje favorecem a polinizacao e o pegamento das flores, levando em conta velocidade do vento, direcao do vento, rajadas, temperatura do ar, umidade relativa do ar e possibilidade de chuva nas proximas horas?
```

Resultado observado:

| Metrica | Score | Threshold | Status |
| --- | ---: | ---: | --- |
| Goal Accuracy | 0.75 | 0.70 | Passou |
| Tool Correctness | 0.86 | 0.70 | Passou |
| Task Completion | 0.90 | 0.70 | Passou |

## Interpretacao dos Resultados

O resultado inicial foi positivo.

A API SMART conseguiu:

- identificar que a pergunta exigia avaliacao multiagente;
- acionar agentes relacionados a ar, chuva e vento;
- consultar ferramentas coerentes com os dados solicitados;
- produzir uma resposta estruturada com resposta, risco, recomendacao e dados coletados.

O score de `Goal Accuracy` passou, mas ficou proximo do limite. Isso indica que a resposta atendeu ao objetivo, mas ainda pode melhorar em alinhamento com a resposta esperada de referencia.

O score de `Tool Correctness` foi bom, indicando que a selecao de ferramentas foi coerente.

O score de `Task Completion` foi forte, indicando que a tarefa foi considerada concluida pelo juiz.

## Ponto de Atencao Sobre o Dataset

As respostas esperadas do dataset possuem valores fixos, como temperatura, umidade, horario de medicao e velocidade do vento.

A API SMART, por outro lado, consulta dados atuais vindos de fontes como InfluxDB e OpenWeather. Por isso, a avaliacao nao deve exigir igualdade literal entre valores numericos atuais e valores do dataset.

A avaliacao deve priorizar criterios como:

- a resposta usou os tipos de dados corretos?
- as ferramentas chamadas foram adequadas?
- a recomendacao foi cautelosa e tecnicamente coerente?
- a resposta informou fontes e horarios?
- a resposta evitou inventar dados?
- a resposta concluiu a tarefa do usuario?

## Exportacao CSV

O script suporta exportacao dos dados coletados para CSV.

Exemplo com poucas perguntas:

```powershell
python -m app.main --dataset .\app\db\db.json --limit 5 --export-csv .\outputs\smart_deepeval_5.csv
```

Exemplo com o dataset completo:

```powershell
python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\smart_deepeval_90.csv
```

O CSV contem:

- numero da pergunta;
- conversation_id;
- rota usada pela API;
- agentes chamados;
- ferramentas chamadas;
- ferramentas esperadas;
- score simples de cobertura de ferramentas;
- pergunta;
- resposta esperada;
- resposta real;
- trace em JSON;
- metadados de RAG em JSON.

Esse CSV pode ser usado depois em Google Colab para gerar graficos e analises.

Exemplo de leitura no Colab:

```python
import pandas as pd

df = pd.read_csv("/content/smart_deepeval_90.csv")
df.head()
```

Exemplo de analise:

```python
df["simple_tool_score"].describe()
df["route"].value_counts()
df["simple_tool_score"].hist()
```

## Comandos Principais

Rodar apenas uma pergunta:

```powershell
python -m app.main --dataset .\app\db\db.json --limit 1
```

Rodar cinco perguntas e exportar CSV:

```powershell
python -m app.main --dataset .\app\db\db.json --limit 5 --export-csv .\outputs\smart_deepeval_5.csv
```

Rodar todas as perguntas:

```powershell
python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\smart_deepeval_90.csv
```

Quando o DeepEval perguntar se deve abrir o inspect TUI:

```text
Open run in deepeval inspect TUI? [Y/n]:
```

Responder:

```text
n
```

## Limitacoes Atuais

- A avaliacao de Tool Correctness valida principalmente nomes de ferramentas, nao parametros.
- Task Completion avalia a chamada externa da API SMART como um todo, nao cada span interno do orquestrador Hono.
- Os valores esperados no dataset podem divergir dos dados atuais retornados pela API.
- O modelo juiz usado via Cloudflare Gateway pode variar em estabilidade de JSON, principalmente usando modelos OSS.
- O CSV exportado registra a coleta e metadados, mas os scores oficiais do DeepEval continuam sendo apresentados no terminal e/ou Confident AI.

## Proximos Passos Recomendados

1. Rodar as 90 perguntas e analisar distribuicao de rotas, agentes e ferramentas.
2. Revisar perguntas com score baixo ou ferramentas ausentes.
3. Adicionar avaliacao de parametros das ferramentas com `Argument Correctness`.
4. Criar uma metrica customizada com `GEval` para qualidade agronomica da recomendacao.
5. Ajustar o dataset para separar:
   - resposta esperada textual;
   - criterios tecnicos esperados;
   - ferramentas esperadas;
   - dados numericos de referencia quando forem estaticos.

## Conclusao

A avaliacao inicial indica que a API SMART esta conseguindo executar o fluxo multiagente esperado para perguntas agricolas sobre cafezal.

O primeiro caso testado passou nas tres metricas principais. Isso sugere que a arquitetura de avaliacao esta funcional e que o comportamento da API esta alinhado ao objetivo geral.

Ainda assim, para uma avaliacao mais robusta, o proximo passo e rodar o dataset completo e complementar as metricas atuais com validacao de parametros e qualidade tecnica agronomica.
