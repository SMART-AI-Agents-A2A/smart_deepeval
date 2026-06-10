python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\smart_deepeval_gpt_oss_120b.csv

python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\smart_deepeval_30_gpt_5_4_mini.csv

python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\smart_deepeval_30_kimi_k_2_6.csv

python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\smart_deepeval_o4_mini_deep_research.csv


gráficos:
python .\analises\deepeval.py --csv .\outputs\smart_deepeval_metrics_20260608-175635_gpt_oss_120b.csv --out-dir .\analises\graficos_gpt_oss_120b