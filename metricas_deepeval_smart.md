# Metricas do DeepEval no Projeto SMART

Este projeto usa tres metricas principais para avaliar a API SMART:

1. **G-Eval**
2. **Tool Correctness**
3. **Task Completion**

Cada metrica gera linhas no CSV de metricas com campos como:

| Campo | Significado |
|---|---|
| `numero` | Numero da pergunta no dataset |
| `metric_name` | Nome da metrica avaliada |
| `score` | Nota de 0 a 1 |
| `threshold` | Nota minima para passar |
| `passed` | Indica se passou ou falhou |
| `reason` | Justificativa textual do juiz |
| `question` | Pergunta enviada para a API SMART |
| `actual_output` | Resposta real da API SMART |
| `expected_output` | Resposta esperada do `db.json` |
| `route` | Rota usada pela API SMART |
| `agents_called` | Agentes acionados |
| `tools_called` | Tools chamadas |
| `expected_tools` | Tools esperadas |
| `tools_correct` | Tools chamadas corretamente |
| `tools_missing` | Tools esperadas que faltaram |
| `tools_extra` | Tools chamadas sem necessidade |

## 1. G-Eval

**O que mede:**  
Avalia a qualidade tecnica da resposta final da API SMART.

**Pergunta que responde:**  
> A resposta foi boa, correta, clara, fundamentada e util para o usuario?

**Como funciona:**  
O G-Eval compara:

| Entrada | Uso |
|---|---|
| `input` | Pergunta do usuario |
| `actual_output` | Resposta real da API SMART |
| `expected_output` | Resposta esperada do `db.json`, usada como referencia tecnica |

No projeto SMART, o G-Eval foi configurado para nao punir automaticamente diferencas numericas quando a API estiver usando dados atuais coletados por sensores ou provedores externos. O foco e avaliar se a conclusao, recomendacao, justificativa tecnica e uso dos dados fazem sentido.

**Campos principais:**

| Campo | Interpretacao |
|---|---|
| `score` | Nota de 0 a 1 |
| `threshold` | Meta minima, normalmente `0.7` |
| `passed` | `true` se `score >= threshold` |
| `reason` | Explicacao do juiz sobre a nota |

**Interpretacao:**

| Resultado | Leitura |
|---|---|
| Score alto | Resposta tecnicamente boa, direta, fundamentada e util |
| Score baixo | Resposta vaga, incompleta, incoerente ou sem conclusao clara |

## 2. Tool Correctness

**O que mede:**  
Avalia se a API SMART chamou as ferramentas corretas para responder a pergunta.

**Pergunta que responde:**  
> A API chamou as tools certas?

**Como funciona:**  
Compara:

| Campo | Uso |
|---|---|
| `tools_called` | Tools realmente chamadas pela API SMART |
| `expected_tools` | Tools esperadas para aquele tipo de pergunta |

**Campos principais:**

| Campo | Interpretacao |
|---|---|
| `score` | Nota de 0 a 1 sobre acerto das tools |
| `passed` | Indica se passou do threshold |
| `reason` | Explica o que faltou ou sobrou |
| `tools_correct` | Tools esperadas que foram chamadas corretamente |
| `tools_missing` | Tools que deveriam ter sido chamadas e nao foram |
| `tools_extra` | Tools chamadas sem necessidade |

**Interpretacao:**

| Resultado | Leitura |
|---|---|
| `score = 1.00` | Chamadas de tools corretas |
| Score intermediario | Parte das tools certas foi chamada, mas faltou algo ou houve excesso |
| Score baixo | A API escolheu tools inadequadas ou deixou de chamar tools essenciais |

Exemplo: se a pergunta pede temperatura, umidade e vento, mas a API chama apenas vento e solo, o Tool Correctness deve cair porque faltaram ferramentas de ar.

## 3. Task Completion

**O que mede:**  
Avalia se a tarefa foi concluida de ponta a ponta.

**Pergunta que responde:**  
> A API resolveu a tarefa do usuario de forma completa?

**Como funciona:**  
Avalia a execucao observada no fluxo:

1. Pergunta do usuario
2. Chamada da API SMART
3. Resposta final
4. Resultado esperado

No projeto SMART, essa metrica roda via tracing/observed agent, entao pode ser mais pesada e mais lenta que as outras.

**Campos principais:**

| Campo | Interpretacao |
|---|---|
| `score` | Nota de 0 a 1 sobre conclusao da tarefa |
| `passed` | Indica se passou do threshold |
| `reason` | Explica o que foi ou nao concluido |
| `actual_output` | Resposta observada |
| `expected_output` | Referencia esperada |

**Interpretacao:**

| Resultado | Leitura |
|---|---|
| Score alto | Tarefa resolvida de forma completa e util |
| Score baixo | Tarefa incompleta, resposta parcial, falta de decisao, falta de dado essencial ou recomendacao insuficiente |

## Como Interpretar as Tres Juntas

| Combinacao | Interpretacao |
|---|---|
| G-Eval alto + Tool Correctness alto + Task Completion alto | A API SMART respondeu bem, usou as tools corretas e concluiu a tarefa |
| G-Eval alto + Tool Correctness baixo | A resposta pode parecer boa, mas o caminho de ferramentas foi incompleto ou inadequado |
| Tool Correctness alto + G-Eval baixo | A API chamou as tools certas, mas interpretou mal os dados ou respondeu de forma ruim |
| Task Completion baixo | Mesmo com alguma resposta ou tool correta, a tarefa final nao foi resolvida de forma satisfatoria |

## Resumo

| Metrica | Mede |
|---|---|
| **G-Eval** | Qualidade tecnica da resposta |
| **Tool Correctness** | Uso correto das ferramentas |
| **Task Completion** | Conclusao da tarefa de ponta a ponta |

Essas metricas se complementam. Uma resposta pode passar em uma e falhar em outra, e isso ajuda a identificar se o problema esta na resposta final, na selecao de ferramentas ou na execucao completa da tarefa.
