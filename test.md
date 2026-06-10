python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\smart_deepeval_responses_gpt_oss_120b.csv --export-metrics-csv .\outputs\smart_deepeval_metrics_gpt_oss_120b.csv

python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\smart_deepeval_responses_gpt_5_4_mini.csv --export-metrics-csv .\outputs\smart_deepeval_metrics_gpt_5_4_mini.csv

python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\smart_deepeval_responses_kimi_k_2_6.csv --export-metrics-csv .\outputs\smart_deepeval_metrics_kimi_k_2_6.csv

python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\smart_deepeval_responses_o4_mini_deep_research.csv --export-metrics-csv .\outputs\smart_deepeval_metrics_o4_mini_deep_research.csv


gráficos:
python .\analises\deepeval.py --csv .\outputs\smart_deepeval_metrics_20260608-175635_gpt_oss_120b.csv --out-dir .\analises\graficos_gpt_oss_120b