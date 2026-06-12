from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TOOL_LABELS = {
    "smart_soil_data": "umidade/temperatura/condutividade do solo quando a pergunta depender de solo, irrigacao, trafegabilidade ou gotejamento",
    "smart_rain_accumulated": "chuva acumulada quando chuva recente alterar risco operacional, compactacao, lavagem ou solo umido",
    "smart_rain_forecast": "previsao de chuva quando a pergunta envolver janela futura, manejo posterior, pulverizacao, adubacao ou risco de chuva",
    "smart_radiation_solar": "radiacao solar quando calor, estresse, evaporacao, pulverizacao ou demanda hidrica forem relevantes",
    "smart_lightning_incidence": "incidencia de raios quando houver risco eletrico ou tempestade",
    "smart_lightning_strikes": "descargas atmosfericas quando houver risco eletrico ou tempestade",
    "smart_lightning_risk": "risco de raios quando houver decisao de seguranca operacional",
    "smart_wind_speed": "velocidade do vento quando houver deriva, pulverizacao, manejo operacional ou risco por vento",
    "smart_wind_direction": "direcao do vento quando houver deriva, pulverizacao ou deslocamento de frente/vento",
    "smart_wind_gust": "rajadas quando houver risco operacional, pulverizacao, florada ou seguranca em campo",
    "smart_wind_current_weather": "vento atual de fonte meteorologica quando for necessario comparar sensor/local e previsao atual",
    "smart_wind_forecast": "previsao de vento quando a decisao for para janela futura",
    "smart_air_temperature": "temperatura do ar quando calor, frio, geada, estresse ou janela operacional forem relevantes",
    "smart_air_humidity": "umidade relativa quando houver calor, evaporacao, pulverizacao, estresse ou doenca",
    "smart_air_pressure": "pressao atmosferica quando a pergunta envolver estabilidade do tempo ou tendencia meteorologica",
    "smart_air_conditions": "condicoes gerais do ar quando a pergunta for ampla sobre clima atual",
    "smart_air_current_weather": "clima atual OpenWeather quando a pergunta exigir dado atual externo ou comparacao meteorologica",
}


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split("|") if item.strip()]
    return []


def _criteria_for_item(item: dict[str, Any]) -> list[str]:
    expected_tools = _as_list(item.get("expected_tools"))
    tool_criteria = [TOOL_LABELS[tool] for tool in expected_tools if tool in TOOL_LABELS]

    criteria = [
        "Responder diretamente ao objetivo da pergunta com conclusao explicita.",
        "Usar dados atuais coletados pela API SMART; valores numericos da resposta de referencia sao exemplos, nao gabarito fixo.",
        "Manter coerencia agronomica mesmo quando os valores atuais diferirem da referencia historica.",
    ]
    if tool_criteria:
        criteria.append("Considerar como evidencias essenciais: " + "; ".join(tool_criteria) + ".")
    criteria.extend(
        [
            "Informar fonte dos dados, unidade e data/hora quando houver leitura medida ou prevista.",
            "Incluir motivo tecnico da recomendacao e sinalizar limitacao quando faltar dado essencial.",
        ]
    )
    return criteria


def build_dataset(input_path: Path, output_path: Path) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("O arquivo de entrada deve conter uma lista JSON.")

    output: list[dict[str, Any]] = []
    for raw_item in data:
        if not isinstance(raw_item, dict):
            raise ValueError("Todos os itens do dataset devem ser objetos JSON.")
        item = dict(raw_item)
        item["expected_criteria"] = _criteria_for_item(item)
        item.setdefault("optional_tools", [])
        output.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Dataset criterial criado em: {output_path}")
    print(f"Itens: {len(output)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera uma versao do db.json com expected_criteria e optional_tools."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("app/db/db.json"),
        help="Dataset original. Padrao: app/db/db.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("app/db/db_criteria.json"),
        help="Dataset de saida. Padrao: app/db/db_criteria.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
