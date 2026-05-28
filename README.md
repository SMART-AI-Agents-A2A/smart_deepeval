# SMART DeepEval

Projeto de avaliação de qualidade das respostas do **SMART**, usando **DeepEval** para testar automaticamente respostas geradas pela API do projeto.

O objetivo é validar se as respostas dos agentes e do orquestrador são claras, úteis, relevantes para a pergunta e coerentes com o contexto da fazenda monitorada.

## Objetivo

Este repositório contém a estrutura de testes com DeepEval para avaliar respostas do projeto SMART, que utiliza API com Hono em Cloudflare Workers.

O DeepEval dispara perguntas contra a API do SMART, coleta a resposta retornada e usa um modelo LLM online como juiz avaliador.

## Como funciona

Fluxo geral:

```txt
Perguntas em datasets/questions.txt
        ↓
DeepEval executa os testes
        ↓
API SMART/Hono recebe cada pergunta
        ↓
API retorna a resposta gerada
        ↓
DeepEval avalia a qualidade da resposta
```

## Estrutura do projeto

```txt
DeepEval/
├── datasets/
│   └── questions.txt
├── tests/
│   ├── __init__.py
│   └── test_questions_only.py
├── utils/
│   ├── __init__.py
│   ├── api_client.py
│   ├── eval_model.py
│   └── questions_loader.py
├── .env.example
├── .gitignore
└── requirements.txt
```

## Requisitos

* Python 3.10+
* DeepEval 4.0.4
* API SMART rodando localmente ou em ambiente acessível
* Chave de um provedor LLM online para atuar como juiz avaliador

## Instalação

Clone o repositório:

```bash
git clone https://github.com/SMART-AI-Agents-A2A/smart_deepeval.git
cd smart_deepeval
```

Crie e ative o ambiente virtual.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Configuração

Copie o arquivo de exemplo:

```bash
cp .env.example .env.local
```

No Windows PowerShell:

```powershell
Copy-Item .env.example .env.local
```

Configure o `.env.local`:

```env
SMART_API_BASE_URL=http://127.0.0.1:8787
SMART_API_ENDPOINT=/v1/orchestrator
SMART_API_METHOD=POST
SMART_API_TIMEOUT=120
SMART_REQUEST_FIELD=message
SMART_RESPONSE_JSON_PATH=data.answer
SMART_AUTH_TOKEN=

OPENAI_API_KEY=sua_chave_openai_aqui
DEEPEVAL_EVALUATOR_MODEL=gpt-4.1-mini
```

## Variáveis de ambiente

| Variável                   | Descrição                                       |
| -------------------------- | ----------------------------------------------- |
| `SMART_API_BASE_URL`       | URL base da API SMART                           |
| `SMART_API_ENDPOINT`       | Endpoint que receberá as perguntas              |
| `SMART_API_METHOD`         | Método HTTP usado na chamada                    |
| `SMART_API_TIMEOUT`        | Tempo máximo da requisição                      |
| `SMART_REQUEST_FIELD`      | Campo enviado no body da requisição             |
| `SMART_RESPONSE_JSON_PATH` | Caminho da resposta dentro do JSON retornado    |
| `SMART_AUTH_TOKEN`         | Token Bearer, caso a API exija autenticação     |
| `OPENAI_API_KEY`           | Chave do provedor usado pelo DeepEval como juiz |
| `DEEPEVAL_EVALUATOR_MODEL` | Modelo online usado para avaliar as respostas   |

## Perguntas de teste

As perguntas ficam em:

```txt
datasets/questions.txt
```

Exemplo:

```txt
Analise a condição atual do solo da fazenda Faz_NSAAB.
Existe risco de chuva para a fazenda Faz_NSAAB?
A radiação atual exige algum alerta operacional?
Como está a condição do vento na fazenda Faz_NSAAB?
Existe algum alerta importante para a operação agrícola agora?
```

Cada linha representa uma pergunta enviada para a API SMART.

## Executando os testes

Antes de rodar o DeepEval, inicie a API SMART em outro terminal.

Exemplo:

```bash
pnpm wrangler dev
```

Depois, execute os testes:

```bash
deepeval test run tests/test_questions_only.py
```

## O que é avaliado

Os testes atuais avaliam as respostas usando dois critérios principais:

1. **Relevância da resposta**

   Verifica se a resposta retornada está relacionada com a pergunta enviada.

2. **Qualidade da resposta SMART**

   Avalia se a resposta é clara, objetiva, útil, coerente com o contexto da fazenda Faz_NSAAB e se evita inventar dados específicos.

## Observação

Nesta versão, os testes não dependem de respostas esperadas fixas.

O DeepEval avalia:

```txt
pergunta enviada + resposta recebida
```

Futuramente, o projeto pode ser expandido para usar datasets com:

```txt
pergunta + resposta esperada + contexto + resposta recebida
```

Isso permitirá avaliações mais rígidas de precisão, aderência ao contexto e qualidade técnica.

## Segurança

O arquivo `.env.local` não deve ser versionado.

Ele pode conter chaves de API e tokens sensíveis.

Use o `.env.example` apenas como modelo público de configuração.

## Licença

Este projeto faz parte do ecossistema SMART AI Agents.
