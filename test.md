python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\gpt_oss_120b_smart_deepeval_responses.csv --export-metrics-csv .\outputs\gpt_oss_120b_pro_smart_deepeval_metrics.csv


python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\gpt_5_4_mini_smart_deepeval_responses.csv --export-metrics-csv .\outputs\gpt_5_4_mini_pro_smart_deepeval_metrics.csv


python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\minimax_m3_smart_deepeval_responses.csv --export-metrics-csv .\outputs\minimax_m3_smart_deepeval_metrics.csv


python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\deepseek_v4_pro_smart_deepeval_responses.csv --export-metrics-csv .\outputs\deepseek_v4_pro_smart_deepeval_metrics.csv


python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\anthropic-claude-opus-4.8_smart_deepeval_responses.csv --export-metrics-csv .\outputs\anthropic-claude-opus-4.8_smart_deepeval_metrics.csv


#============================================================


python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\kimi_k_2_6_smart_deepeval_responses.csv --export-metrics-csv .\outputs\kimi_k_2_6_pro_smart_deepeval_metrics.csv


python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\o4_mini_deep_research_smart_deepeval_responses.csv --export-metrics-csv .\outputs\o4_mini_deep_research_smart_deepeval_metrics.csv


gráficos:
python .\analises\deepeval.py --csv .\outputs\smart_deepeval_metrics_20260608-175635_gpt_oss_120b.csv --out-dir .\analises\graficos_gpt_oss_120b