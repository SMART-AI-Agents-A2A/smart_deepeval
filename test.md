python -m app.main --dataset .\app\db\db.json --quiet --export-csv .\outputs\deepseek4pro_response.csv --export-metrics-csv .\outputs\deepseek4pro_metrics.csv --debug-json .\outputs\deepseek4pro_debug.json

python -m app.main --dataset .\app\db\db.json --quiet --export-csv .\outputs\gpt54mini_response.csv --export-metrics-csv .\outputs\gpt54mini_metrics.csv --debug-json .\outputs\gpt54mini_debug.json

python -m app.main --dataset .\app\db\db.json --quiet --export-csv .\outputs\minimax3_response.csv --export-metrics-csv .\outputs\minimax3_metrics.csv --debug-json .\outputs\minimax3_debug.json

python -m app.main --dataset .\app\db\db.json --quiet --export-csv .\outputs\kimi-k2.6_response.csv --export-metrics-csv .\outputs\kimi-k2.6_metrics.csv --debug-json .\outputs\kimi-k2.6_debug.json

python -m app.main --dataset .\app\db\db.json --quiet --export-csv .\outputs\gptoss120b_response.csv --export-metrics-csv .\outputs\gptoss120b_metrics.csv --debug-json .\outputs\gptoss120b_debug.json

# PENSAR EM USAR DEPOIS
python -m app.main --dataset .\app\db\db.json --quiet --export-csv .\outputs\gemini31pro_preview_response.csv --export-metrics-csv .\outputs\gemini31pro_preview_metrics.csv --debug-json .\outputs\gemini31pro_preview_debug.json

python -m app.main --dataset .\app\db\db.json --quiet --export-csv .\outputs\claudeOpus47_response.csv --export-metrics-csv .\outputs\claudeOpus47_preview_metrics.csv --debug-json .\outputs\claudeOpus47_preview_debug.json


# GRÁFICOS
python .\analises\deepeval.py `
  --csv .\outputs\gpt54mini_metrics.csv .\outputs\deepseek4pro_metrics.csv .\outputs\minimax3_metrics.csv `
  --labels GPT-5.4-mini DeepSeek-V4-Pro MiniMax-M3 `
  --out-dir .\analises\graficos_comparativo `
  --pdf-name comparativo_3modelos.pdf


python .\analises\graphs.py `
  --csv .\outputs\gpt54mini_metrics.csv .\outputs\deepseek4pro_metrics.csv .\outputs\minimax3_metrics.csv `
  --labels GPT-5.4-mini DeepSeek-V4-Pro MiniMax-M3 `
  --out-dir .\analises\graficos_comparativo `
  --pdf-name comparativo_3modelos.pdf `
  --language both