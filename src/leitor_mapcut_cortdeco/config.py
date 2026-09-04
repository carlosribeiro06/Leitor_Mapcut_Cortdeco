"""Carga e validacao do arquivo de configuracao (`settings.json`).

Toda a parametrizacao do leitor vive nesse arquivo externo: caminhos dos
binarios, destino dos CSVs, politica de log e geometria de leitura do cortdeco.
Nenhum caminho ou parametro fica embutido no codigo, de modo que o projeto roda
sem alteracao em WSL, em servidor Linux ou na maquina de outro pesquisador.

A validacao e *fail fast*: qualquer chave ausente, com tipo errado ou com valor
fora do dominio aceito interrompe a execucao com uma mensagem que aponta
exatamente a chave problematica.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

DEFAULT_SETTINGS_FILENAME: Final[str] = "settings.json"

# Fontes aceitas para o numero de cortes por no (ver CortdecoSettings).
CUTS_PER_NODE_SOURCES: Final[frozenset[str]] = frozenset(
    {"numero_iteracoes", "file_geometry"}
)

VALID_LOG_LEVELS: Final[frozenset[str]] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


class ConfigError(Exception):
    """Erro de configuracao: chave ausente, tipo invalido ou valor fora do dominio."""


@dataclass(frozen=True)
class PathsSettings:
    """Caminhos de entrada e saida, ja resolvidos para caminhos absolutos."""

    input_directory: Path
    mapcut_file: str
    cortdeco_file: str
    output_directory: Path

    @property
    def mapcut_path(self) -> Path:
        """Caminho completo do binario mapcut."""
        return self.input_directory / self.mapcut_file

    @property
    def cortdeco_path(self) -> Path:
        """Caminho completo do binario cortdeco."""
        return self.input_directory / self.cortdeco_file


@dataclass(frozen=True)
class LoggingSettings:
    """Politica de log de auditoria (console + arquivo rotativo)."""

    level: str
    directory: Path
    filename: str
    max_bytes: int
    backup_count: int
    use_rich: bool

    @property
    def log_path(self) -> Path:
        """Caminho completo do arquivo de log."""
        return self.directory / self.filename


@dataclass(frozen=True)
class ExportSettings:
    """Formato dos CSVs e selecao de quais conjuntos de tabelas gerar."""

    separator: str
    decimal: str
    float_format: str | None
    encoding: str
    write_mapcut: bool
    write_cortdeco_wide: bool
    write_cortdeco_tidy: bool

    @property
    def write_cortdeco(self) -> bool:
        """Indica se alguma tabela derivada do cortdeco sera gerada."""
        return self.write_cortdeco_wide or self.write_cortdeco_tidy


@dataclass(frozen=True)
class CortdecoSettings:
    """Como determinar o numero de cortes por no na leitura do cortdeco.

    O cortdeco guarda os cortes como uma lista encadeada de registros de tamanho
    fixo, e o numero de cortes de cada no *nao* esta gravado explicitamente no
    mapcut: ele equivale ao numero de iteracoes da politica. Como esse valor
    dimensiona os vetores de leitura, ele pode ser obtido por duas vias
    independentes (que o leitor sempre compara entre si) ou fixado na mao.
    """

    cuts_per_node_source: str
    cuts_per_node_override: int | None


@dataclass(frozen=True)
class Settings:
    """Configuracao completa da aplicacao."""

    paths: PathsSettings
    logging: LoggingSettings
    export: ExportSettings
    cortdeco: CortdecoSettings
    settings_path: Path


def _type_names(types: tuple[type, ...]) -> str:
    """Descreve uma tupla de tipos de forma legivel na mensagem de erro."""
    return " ou ".join(t.__name__ for t in types)


def _require_section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    """Extrai uma secao obrigatoria do JSON, garantindo que seja um objeto."""
    if name not in raw:
        raise ConfigError(f"secao obrigatoria ausente em settings.json: {name!r}")
    section = raw[name]
    if not isinstance(section, dict):
        raise ConfigError(
            f"a secao {name!r} deve ser um objeto JSON, e nao {type(section).__name__}"
        )
    return section


def _require(
    section: dict[str, Any],
    path: str,
    key: str,
    expected: type | tuple[type, ...],
) -> Any:
    """Extrai uma chave obrigatoria e valida seu tipo.

    O caso `bool` e tratado explicitamente porque em Python `isinstance(True, int)`
    e verdadeiro, o que deixaria passar `true` onde se espera um inteiro.
    """
    if key not in section:
        raise ConfigError(f"chave obrigatoria ausente em settings.json: {path}.{key}")
    value = section[key]
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    if bool not in expected_types and isinstance(value, bool):
        raise ConfigError(
            f"{path}.{key} deve ser {_type_names(expected_types)}, e nao booleano"
        )
    if not isinstance(value, expected_types):
        raise ConfigError(
            f"{path}.{key} deve ser {_type_names(expected_types)}, "
            f"e nao {type(value).__name__} (valor: {value!r})"
        )
    return value


def _resolve(base: Path, value: str, key: str) -> Path:
    """Resolve um caminho da configuracao em relacao ao diretorio do settings.json.

    Caminhos absolutos sao respeitados como estao; caminhos relativos passam a ser
    ancorados no proprio arquivo de configuracao, e nao no diretorio de trabalho
    de quem chamou a aplicacao.
    """
    if not value.strip():
        raise ConfigError(f"{key} nao pode ser uma string vazia")
    candidate = Path(value).expanduser()
    resolved = candidate if candidate.is_absolute() else (base / candidate)
    return resolved.resolve()


def _require_filename(section: dict[str, Any], path: str, key: str) -> str:
    """Valida um nome de arquivo simples (sem componente de diretorio)."""
    value: str = _require(section, path, key, str)
    if not value.strip():
        raise ConfigError(f"{path}.{key} nao pode ser uma string vazia")
    if Path(value).name != value:
        raise ConfigError(
            f"{path}.{key} deve ser apenas o nome do arquivo, sem diretorio "
            f"(use paths.input_directory para o diretorio); valor: {value!r}"
        )
    return value


def _parse_paths(raw: dict[str, Any], base: Path) -> PathsSettings:
    """Valida a secao 'paths'."""
    section = _require_section(raw, "paths")
    return PathsSettings(
        input_directory=_resolve(
            base,
            _require(section, "paths", "input_directory", str),
            "paths.input_directory",
        ),
        mapcut_file=_require_filename(section, "paths", "mapcut_file"),
        cortdeco_file=_require_filename(section, "paths", "cortdeco_file"),
        output_directory=_resolve(
            base,
            _require(section, "paths", "output_directory", str),
            "paths.output_directory",
        ),
    )


def _parse_logging(raw: dict[str, Any], base: Path) -> LoggingSettings:
    """Valida a secao 'logging'."""
    section = _require_section(raw, "logging")
    level = _require(section, "logging", "level", str).upper()
    if level not in VALID_LOG_LEVELS:
        raise ConfigError(
            f"logging.level invalido: {level!r}; use um de {sorted(VALID_LOG_LEVELS)}"
        )
    max_bytes = _require(section, "logging", "max_bytes", int)
    if max_bytes <= 0:
        raise ConfigError(f"logging.max_bytes deve ser positivo (valor: {max_bytes})")
    backup_count = _require(section, "logging", "backup_count", int)
    if backup_count < 0:
        raise ConfigError(
            f"logging.backup_count nao pode ser negativo (valor: {backup_count})"
        )
    return LoggingSettings(
        level=level,
        directory=_resolve(
            base,
            _require(section, "logging", "directory", str),
            "logging.directory",
        ),
        filename=_require_filename(section, "logging", "filename"),
        max_bytes=max_bytes,
        backup_count=backup_count,
        use_rich=_require(section, "logging", "use_rich", bool),
    )


def _parse_export(raw: dict[str, Any]) -> ExportSettings:
    """Valida a secao 'export'."""
    section = _require_section(raw, "export")
    separator = _require(section, "export", "separator", str)
    if len(separator) != 1:
        raise ConfigError(
            f"export.separator deve ter exatamente um caractere (valor: {separator!r})"
        )
    decimal = _require(section, "export", "decimal", str)
    if len(decimal) != 1:
        raise ConfigError(
            f"export.decimal deve ter exatamente um caractere (valor: {decimal!r})"
        )
    if separator == decimal:
        raise ConfigError(
            "export.separator e export.decimal nao podem ser o mesmo caractere "
            f"({separator!r}), pois isso tornaria o CSV ambiguo"
        )
    # float_format aceita null (precisao total do double) ou uma mascara de formato.
    float_format = section.get("float_format")
    if float_format is not None and not isinstance(float_format, str):
        raise ConfigError(
            f"export.float_format deve ser string ou null, "
            f"e nao {type(float_format).__name__}"
        )
    encoding = _require(section, "export", "encoding", str)
    if not encoding.strip():
        raise ConfigError("export.encoding nao pode ser uma string vazia")
    return ExportSettings(
        separator=separator,
        decimal=decimal,
        float_format=float_format,
        encoding=encoding,
        write_mapcut=_require(section, "export", "write_mapcut", bool),
        write_cortdeco_wide=_require(section, "export", "write_cortdeco_wide", bool),
        write_cortdeco_tidy=_require(section, "export", "write_cortdeco_tidy", bool),
    )


def _parse_cortdeco(raw: dict[str, Any]) -> CortdecoSettings:
    """Valida a secao 'cortdeco'."""
    section = _require_section(raw, "cortdeco")
    source = _require(section, "cortdeco", "cuts_per_node_source", str)
    if source not in CUTS_PER_NODE_SOURCES:
        raise ConfigError(
            f"cortdeco.cuts_per_node_source invalido: {source!r}; "
            f"use um de {sorted(CUTS_PER_NODE_SOURCES)}"
        )
    override = section.get("cuts_per_node_override")
    if override is not None:
        if isinstance(override, bool) or not isinstance(override, int):
            raise ConfigError(
                "cortdeco.cuts_per_node_override deve ser inteiro ou null, "
                f"e nao {type(override).__name__}"
            )
        if override <= 0:
            raise ConfigError(
                f"cortdeco.cuts_per_node_override deve ser positivo (valor: {override})"
            )
    return CortdecoSettings(
        cuts_per_node_source=source,
        cuts_per_node_override=override,
    )


def _default_settings_path() -> Path:
    """Localiza o `settings.json` que acompanha o projeto.

    O modulo vive em `<raiz>/src/leitor_mapcut_cortdeco/config.py`, logo a raiz
    esta dois diretorios acima.
    """
    return (Path(__file__).resolve().parents[2] / DEFAULT_SETTINGS_FILENAME).resolve()


def load_settings(settings_path: str | Path | None = None) -> Settings:
    """Le e valida o arquivo de configuracao.

    Args:
        settings_path: caminho do JSON de configuracao. Quando omitido, procura
            `settings.json` na raiz do projeto.

    Returns:
        A configuracao validada, com todos os caminhos resolvidos.

    Raises:
        ConfigError: arquivo inexistente, JSON malformado, ou qualquer chave
            ausente / com tipo ou valor invalido.
    """
    path = (
        Path(settings_path).expanduser().resolve()
        if settings_path is not None
        else _default_settings_path()
    )
    if not path.is_file():
        raise ConfigError(f"arquivo de configuracao nao encontrado: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"JSON malformado em {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"o conteudo de {path} deve ser um objeto JSON")

    # O diretorio do proprio settings.json e a ancora dos caminhos relativos.
    base = path.parent
    return Settings(
        paths=_parse_paths(raw, base),
        logging=_parse_logging(raw, base),
        export=_parse_export(raw),
        cortdeco=_parse_cortdeco(raw),
        settings_path=path,
    )
