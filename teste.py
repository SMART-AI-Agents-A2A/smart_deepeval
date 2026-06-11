from dotenv import load_dotenv
load_dotenv(".env", override=True)

from app.api.juiz import configure_judge_environment

cfg = configure_judge_environment()
print("modelo:", cfg.model_name)
print("base_url:", cfg.base_url)
resp = cfg.model.generate("Responda apenas com a palavra: OK")
print("RESPOSTA:", repr(resp))