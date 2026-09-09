"""Escrita das tabelas em CSV, com registro de volume no log.

O formato (separador, marcador decimal, formato dos reais e codificacao) vem
inteiramente do `settings.json`, para que o mesmo codigo produza tanto arquivos
portaveis para pandas/R quanto arquivos que abrem direto no Excel em
portugues-BR.

Sobre a precisao dos reais: com `float_format` nulo, o pandas grava a menor
representacao decimal que reconstroi exatamente o `float64` original
(*round-trip*). E o padrao aqui, porque os coeficientes dos cortes sao duais de
um problema de otimizacao e arredondar altera o resultado de quem reconstruir a
funcao de custo futuro a partir do CSV.

Tabela vazia: arquivo sempre gravado
------------------------------------
Uma tabela sem linhas e gravada de todo modo, com a linha de cabecalho. A razao
e de contrato: o conjunto de arquivos de saida nao deve depender do caso, para
que os scripts a jusante possam ler todos eles sem tratar ausencia, e para que a
propria existencia do arquivo comprove que a etapa rodou. Em troca, a ausencia de
dados nao pode passar em silencio - por isso as tabelas vazias sao nomeadas em um
WARNING e contadas no resumo final.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ExportSettings


@dataclass(frozen=True)
class ExportedFile:
    """Registro do que foi gravado, usado no resumo final da execucao."""

    name: str
    path: Path
    rows: int
    columns: int
    size_bytes: int


def write_tables(
    tables: dict[str, pd.DataFrame],
    output_directory: Path,
    settings: ExportSettings,
    logger: logging.Logger,
) -> list[ExportedFile]:
    """Grava cada tabela como um CSV no diretorio de saida.

    Args:
        tables: dicionario `nome_do_csv -> DataFrame` (sem a extensao).
        output_directory: diretorio de destino, criado se necessario.
        settings: formato dos CSVs, lido do settings.json.
        logger: logger da aplicacao.

    Returns:
        Um registro por arquivo gravado, na ordem em que foram gravados.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    exported: list[ExportedFile] = []
    for name, table in tables.items():
        path = output_directory / f"{name}.csv"
        table.to_csv(
            path,
            sep=settings.separator,
            decimal=settings.decimal,
            float_format=settings.float_format,
            encoding=settings.encoding,
            index=False,
        )
        record = ExportedFile(
            name=name,
            path=path,
            rows=len(table),
            columns=table.shape[1],
            size_bytes=path.stat().st_size,
        )
        exported.append(record)
        logger.info(
            "gravado %s: %d linhas x %d colunas (%.2f MiB)",
            path.name,
            record.rows,
            record.columns,
            record.size_bytes / (1024 * 1024),
        )

    vazias = [record.name for record in exported if record.rows == 0]
    if vazias:
        logger.warning(
            "%d tabela(s) gravada(s) apenas com o cabecalho, sem nenhuma linha: "
            "%s. Confirme que a ausencia de dados e esperada para este caso - as "
            "tabelas de tempo de viagem, por exemplo, ficam legitimamente vazias "
            "quando o caso nao tem UHE com tempo de viagem da agua.",
            len(vazias),
            ", ".join(vazias),
        )
    return exported


def log_export_summary(
    exported: list[ExportedFile],
    output_directory: Path,
    logger: logging.Logger,
) -> None:
    """Registra o resumo consolidado da exportacao."""
    total_rows = sum(record.rows for record in exported)
    total_bytes = sum(record.size_bytes for record in exported)
    empty_count = sum(1 for record in exported if record.rows == 0)
    logger.info(
        "exportacao concluida: %d arquivos (%d apenas com cabecalho), %d linhas, "
        "%.2f MiB em %s",
        len(exported),
        empty_count,
        total_rows,
        total_bytes / (1024 * 1024),
        output_directory,
    )
