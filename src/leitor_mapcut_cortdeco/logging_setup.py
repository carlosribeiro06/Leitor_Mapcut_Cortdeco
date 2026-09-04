"""Configuracao do log de auditoria (console + arquivo rotativo).

O objetivo aqui e permitir reconstruir depois, a partir do arquivo de log, o que
uma execucao oficial de fato fez: quais binarios foram lidos, com que geometria,
quantas linhas cada CSV recebeu e quanto tempo cada etapa levou.

Sao instalados dois destinos:

* **console** - legivel para quem acompanha a execucao. Usa `RichHandler`
  (colorido, com nivel e horario alinhados) quando a biblioteca `rich` esta
  disponivel e habilitada no `settings.json`; caso contrario, cai num
  `StreamHandler` com formato limpo.
* **arquivo** - `RotatingFileHandler` com formato detalhado e fixo (sempre o
  mesmo, independente do console), que e a trilha de auditoria propriamente dita.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

from .config import LoggingSettings

LOGGER_NAME: Final[str] = "leitor_mapcut_cortdeco"

# Formato do arquivo de log: completo e estavel, para leitura posterior.
FILE_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
FILE_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

# Formato do console quando o rich nao esta disponivel.
CONSOLE_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(message)s"
CONSOLE_DATE_FORMAT: Final[str] = "%H:%M:%S"


def _build_console_handler(settings: LoggingSettings) -> logging.Handler:
    """Cria o handler de console, preferindo o `rich` quando possivel.

    O `rich` e opcional de proposito: se ele nao estiver instalado no ambiente,
    a aplicacao continua funcionando com um formato de texto simples em vez de
    falhar por causa de uma dependencia de apresentacao.
    """
    if settings.use_rich:
        try:
            from rich.console import Console
            from rich.logging import RichHandler
        except ImportError:
            pass
        else:
            handler: logging.Handler = RichHandler(
                console=Console(stderr=True),
                show_path=False,
                rich_tracebacks=True,
                log_time_format=f"[{CONSOLE_DATE_FORMAT}]",
            )
            # O RichHandler ja imprime horario e nivel; resta apenas a mensagem.
            handler.setFormatter(logging.Formatter("%(message)s"))
            return handler

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(CONSOLE_FORMAT, datefmt=CONSOLE_DATE_FORMAT)
    )
    return stream_handler


def _build_file_handler(settings: LoggingSettings) -> logging.Handler:
    """Cria o handler de arquivo rotativo, garantindo a existencia do diretorio."""
    settings.directory.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        settings.log_path,
        maxBytes=settings.max_bytes,
        backupCount=settings.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, datefmt=FILE_DATE_FORMAT))
    return file_handler


def setup_logging(
    settings: LoggingSettings,
    level_override: str | None = None,
) -> logging.Logger:
    """Instala os handlers de console e arquivo e devolve o logger da aplicacao.

    Args:
        settings: politica de log lida do `settings.json`.
        level_override: nivel vindo da linha de comando, que tem precedencia
            sobre o do arquivo de configuracao.

    Returns:
        O logger raiz da aplicacao, ja configurado.
    """
    level = (level_override or settings.level).upper()
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    # Chamadas repetidas (por exemplo, em testes) nao devem duplicar mensagens.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    for handler in (_build_console_handler(settings), _build_file_handler(settings)):
        handler.setLevel(level)
        logger.addHandler(handler)

    # O log da aplicacao e autossuficiente; nao repassa para o logger raiz.
    logger.propagate = False

    logger.debug("log configurado em nivel %s; arquivo: %s", level, settings.log_path)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Devolve o logger da aplicacao (ou um filho seu, nomeado por modulo)."""
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


@contextmanager
def log_step(logger: logging.Logger, description: str) -> Iterator[None]:
    """Registra inicio, fim e tempo decorrido de uma etapa significativa.

    Em caso de excecao, registra a falha com o tempo gasto antes de propagar,
    para que o log mostre onde e quando a execucao parou.

    Args:
        logger: logger que recebera as mensagens.
        description: descricao da etapa, usada nas duas mensagens.
    """
    logger.info("[inicio] %s", description)
    started_at = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started_at
        logger.exception("[falha]  %s (%.3f s)", description, elapsed)
        raise
    elapsed = time.perf_counter() - started_at
    logger.info("[fim]    %s (%.3f s)", description, elapsed)


def log_file_summary(logger: logging.Logger, label: str, path: Path) -> None:
    """Registra o tamanho de um arquivo de entrada, para rastreabilidade do volume."""
    size_bytes = path.stat().st_size
    logger.info(
        "%s: %s (%d bytes; %.2f MiB)",
        label,
        path,
        size_bytes,
        size_bytes / (1024 * 1024),
    )
