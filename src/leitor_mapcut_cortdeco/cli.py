"""Interface de linha de comando.

Os argumentos cobrem apenas o que muda de uma execucao para outra (qual
configuracao usar, onde gravar, nivel de log). Tudo o mais - nomes de arquivos,
formato dos CSVs, geometria - fica no `settings.json`, para que uma execucao
oficial seja reproduzivel a partir de um arquivo versionado, e nao de uma linha
de comando digitada na hora.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from .config import ConfigError, Settings, load_settings
from .geometry import GeometryError
from .logging_setup import setup_logging
from .pipeline import run

EPILOG = """\
Exemplos:
  leitor-mapcut-cortdeco
  leitor-mapcut-cortdeco --settings /caminho/settings.json
  leitor-mapcut-cortdeco --output-directory ./saida_rv0 --log-level DEBUG
  leitor-mapcut-cortdeco --somente-mapcut
"""


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="leitor-mapcut-cortdeco",
        description=(
            "Le os binarios mapcut e cortdeco do DECOMP (via idecomp) e exporta "
            "seu conteudo em CSV."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        metavar="ARQUIVO",
        help=(
            "caminho do JSON de configuracao (padrao: settings.json na raiz do projeto)"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        metavar="DIRETORIO",
        help="sobrescreve paths.output_directory do settings.json",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="sobrescreve logging.level do settings.json",
    )
    parser.add_argument(
        "--somente-mapcut",
        action="store_true",
        help=(
            "le e exporta apenas o mapcut; a geometria dos cortes ainda e "
            "derivada e conferida contra o cortdeco"
        ),
    )
    return parser


def _apply_overrides(settings: Settings, output_directory: Path | None) -> Settings:
    """Aplica ao objeto de configuracao as sobreposicoes vindas da linha de comando."""
    if output_directory is None:
        return settings
    paths = replace(
        settings.paths, output_directory=output_directory.expanduser().resolve()
    )
    return replace(settings, paths=paths)


def main(argv: list[str] | None = None) -> int:
    """Ponto de entrada da aplicacao.

    Args:
        argv: argumentos de linha de comando (padrao: `sys.argv[1:]`).

    Returns:
        0 em caso de sucesso; 1 em erro de configuracao ou de geometria;
        2 em qualquer falha inesperada.
    """
    args = build_parser().parse_args(argv)

    # A configuracao e carregada antes do log porque e ela que define o log.
    # Por isso este bloco reporta erro em stderr, e nao pelo logger.
    try:
        settings = _apply_overrides(load_settings(args.settings), args.output_directory)
    except ConfigError as exc:
        print(f"erro de configuracao: {exc}", file=sys.stderr)
        return 1

    logger = setup_logging(settings.logging, level_override=args.log_level)

    try:
        report = run(settings, logger, read_cortdeco_file=not args.somente_mapcut)
    except (ConfigError, GeometryError) as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.exception("falha inesperada durante a execucao")
        return 2

    logger.info(
        "execucao concluida: %d CSVs em %s",
        len(report.exported),
        report.output_directory,
    )
    return 0
