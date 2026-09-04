"""Orquestracao da execucao: ler os binarios, conferir e exportar os CSVs.

A ordem das etapas nao e arbitraria. O `mapcut` e sempre lido primeiro, mesmo
quando so se quer o `cortdeco`, porque e dele que sai a geometria dos registros
de corte; e a geometria e conferida contra o arquivo *antes* de qualquer leitura
de cortes, porque uma geometria errada nao faz a leitura falhar - ela devolve
numeros plausiveis e errados.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ConfigError, Settings
from .cortdeco_reader import build_cortdeco_tables, read_cortdeco, validate_cuts
from .csv_export import ExportedFile, log_export_summary, write_tables
from .geometry import CutGeometry, build_cut_geometry, validate_geometry
from .logging_setup import log_file_summary, log_step
from .mapcut_reader import build_mapcut_tables, read_mapcut


@dataclass(frozen=True)
class RunReport:
    """Resultado de uma execucao completa."""

    geometry: CutGeometry
    exported: list[ExportedFile]
    output_directory: Path


def _require_input_file(path: Path, label: str) -> None:
    """Interrompe a execucao se um binario de entrada nao existir ou estiver vazio."""
    if not path.is_file():
        raise ConfigError(f"{label} nao encontrado: {path}")
    if path.stat().st_size == 0:
        raise ConfigError(f"{label} esta vazio: {path}")


def run(
    settings: Settings,
    logger: logging.Logger,
    read_cortdeco_file: bool = True,
) -> RunReport:
    """Executa a leitura dos binarios e a exportacao dos CSVs.

    Args:
        settings: configuracao validada.
        logger: logger da aplicacao.
        read_cortdeco_file: quando falso, le apenas o mapcut e exporta suas
            tabelas. A geometria continua sendo derivada e conferida, porque ela
            e informacao do proprio mapcut.

    Returns:
        O relatorio da execucao, com a geometria usada e os arquivos gravados.

    Raises:
        ConfigError: binario de entrada ausente ou vazio.
        GeometryError: geometria inconsistente entre o mapcut e o cortdeco.
    """
    logger.info("configuracao carregada de %s", settings.settings_path)
    _require_input_file(settings.paths.mapcut_path, "mapcut")
    _require_input_file(settings.paths.cortdeco_path, "cortdeco")
    log_file_summary(logger, "entrada mapcut", settings.paths.mapcut_path)
    log_file_summary(logger, "entrada cortdeco", settings.paths.cortdeco_path)

    with log_step(logger, "leitura do mapcut"):
        mapcut = read_mapcut(settings.paths.mapcut_path, logger)

    with log_step(logger, "derivacao e conferencia da geometria dos cortes"):
        geometry = build_cut_geometry(
            mapcut, settings.paths.cortdeco_path, settings.cortdeco, logger
        )
        validate_geometry(geometry, settings.paths.cortdeco_path, logger)

    tables: dict[str, pd.DataFrame] = {}

    if settings.export.write_mapcut:
        with log_step(logger, "montagem das tabelas do mapcut"):
            tables.update(build_mapcut_tables(mapcut, geometry, logger))
    else:
        logger.info("tabelas do mapcut desabilitadas em export.write_mapcut")

    if read_cortdeco_file and settings.export.write_cortdeco:
        with log_step(logger, "leitura do cortdeco"):
            cortdeco = read_cortdeco(settings.paths.cortdeco_path, geometry, logger)
            cuts = cortdeco.cortes
            if cuts is not None:
                validate_cuts(cuts, geometry, logger)
        with log_step(logger, "montagem das tabelas do cortdeco"):
            tables.update(
                build_cortdeco_tables(
                    cortdeco,
                    geometry,
                    write_wide=settings.export.write_cortdeco_wide,
                    write_tidy=settings.export.write_cortdeco_tidy,
                    logger=logger,
                )
            )
    elif not read_cortdeco_file:
        logger.info("leitura do cortdeco omitida por opcao da linha de comando")
    else:
        logger.info(
            "tabelas do cortdeco desabilitadas em export.write_cortdeco_wide "
            "e export.write_cortdeco_tidy"
        )

    if not tables:
        logger.warning(
            "nenhuma tabela selecionada para exportacao; revise a secao 'export' "
            "do settings.json"
        )

    with log_step(logger, "escrita dos CSVs"):
        exported = write_tables(
            tables, settings.paths.output_directory, settings.export, logger
        )
    log_export_summary(exported, settings.paths.output_directory, logger)

    return RunReport(
        geometry=geometry,
        exported=exported,
        output_directory=settings.paths.output_directory,
    )
