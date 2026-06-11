python -m app.main --dataset .\app\db\db.json --quiet --export-csv .\outputs\deepseek4pro_response.csv --export-metrics-csv .\outputs\deepseek4pro_metrics.csv --debug-json .\outputs\deepseek4pro_debug.json

python -m app.main --dataset .\app\db\db.json --quiet --export-csv .\outputs\gpt54mini_response.csv --export-metrics-csv .\outputs\gpt54mini_metrics.csv --debug-json .\outputs\gpt54mini_debug.json

python -m app.main --dataset .\app\db\db.json --quiet --export-csv .\outputs\minimax3_response.csv --export-metrics-csv .\outputs\minimax3_metrics.csv --debug-json .\outputs\minimax3_debug.json

python -m app.main --dataset .\app\db\db.json --quiet --export-csv .\outputs\gptoss120b_response.csv --export-metrics-csv .\outputs\gptoss120b_metrics.csv --debug-json .\outputs\gptoss120b_debug.json



gráficos:
python .\analises\deepeval.py --csv .\outputs\smart_deepeval_metrics_20260608-175635_gpt_oss_120b.csv --out-dir .\analises\graficos_gpt_oss_120b