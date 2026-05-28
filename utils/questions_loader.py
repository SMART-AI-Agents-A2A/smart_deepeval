from __future__ import annotations

from pathlib import Path


def load_questions(path: str = "datasets/questions.txt") -> list[str]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo de perguntas não encontrado: {file_path}")

    questions = [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not questions:
        raise ValueError(f"Nenhuma pergunta encontrada em: {file_path}")

    return questions