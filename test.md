testar 5 perguntas:
python -m app.main --dataset .\app\db\db.json --limit 5 --export-csv .\outputs\smart_top30_after_orchestrate_5_collected.csv --export-metrics-csv .\outputs\smart_top30_after_orchestrate_5_metrics.csv

testar tudo:
python -m app.main --dataset .\app\db\db.json --export-csv .\outputs\smart_top30_after_orchestrate_collected.csv --export-metrics-csv .\outputs\smart_top30_after_orchestrate_metrics.csv