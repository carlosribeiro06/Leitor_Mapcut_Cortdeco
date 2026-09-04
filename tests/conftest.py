"""Fixtures compartilhadas pelos testes.

Duas estrategias convivem aqui:

* **caso sintetico** - um mapcut falso e um cortdeco minusculo gravado byte a
  byte, com valores escolhidos para que cada coeficiente lido possa ser
  conferido individualmente. E o que permite testar a geometria e a leitura da
  lista encadeada sem depender de nenhum arquivo grande.
* **caso real** - os binarios `mapcut.rv0` / `cortdeco.rv0` da raiz do projeto,
  usados nos testes de integracao. Quando nao estao presentes (uma copia limpa
  do repositorio, por exemplo), esses testes sao ignorados em vez de falhar.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dimensoes do caso sintetico. Sao pequenas de proposito, para que o arquivo
# gerado caiba em algumas centenas de bytes e possa ser conferido na mao.
SYNTHETIC_STAGES = 3
SYNTHETIC_PLANT_CODES = (11, 22)
SYNTHETIC_TRAVEL_TIME_PLANT_CODES = (22,)
SYNTHETIC_GNL_SUBMARKET_CODES = (1,)
SYNTHETIC_MAX_TRAVEL_TIME_LAG = 2
SYNTHETIC_LOAD_BLOCKS = 2
SYNTHETIC_CUTS_PER_NODE = 3
SYNTHETIC_RECORD_SIZE = 128

# 1 rhs + 2 volume armazenado + 1 UHE x 2 lags + 1 submercado x 3 estagios x 2 patamares
SYNTHETIC_COEFFICIENT_COUNT = 11

# Encadeamento dos registros (1-based) de cada no, na ordem em que sao lidos.
# O no 1 (estagio 1) tem 3 cortes; o no 2 (estagio 2) tem 4, porque e o ultimo
# estagio que constroi cortes e o DECOMP grava um registro extra.
SYNTHETIC_CHAINS: dict[int, tuple[int, ...]] = {
    1: (6, 4, 2),
    2: (7, 5, 3, 1),
}
SYNTHETIC_TOTAL_RECORDS = 7


@dataclass
class FakeMapcut:
    """Dublê do objeto `Mapcut` com apenas os atributos que a geometria consome."""

    tamanho_corte: int
    numero_iteracoes: int
    numero_cortes: int
    numero_estagios: int
    patamares_por_estagio: list[int]
    maximo_lag_tempo_viagem: int
    codigos_uhes: list[int]
    codigos_uhes_tempo_viagem: list[int]
    codigos_submercados_gnl: list[int]
    registro_ultimo_corte_no: pd.DataFrame
    codigos_uhes_jusante: list[Any]


def synthetic_last_cut_table() -> pd.DataFrame:
    """Tabela de entrada da lista encadeada de cada no do caso sintetico.

    O ultimo estagio nao constroi cortes, por isso seu no tem indice 0 - valor
    que tambem sinaliza ao leitor do idecomp o fim da varredura.
    """
    return pd.DataFrame(
        {
            "no": [1, 2, 3],
            "estagio": [1, 2, 3],
            # O no do estagio 2 aponta para 5; o leitor soma o passo (2 nos) para
            # alcancar o registro extra 7.
            "indice_ultimo_corte": [6, 5, 0],
        }
    )


@pytest.fixture
def fake_mapcut() -> FakeMapcut:
    """Mapcut sintetico coerente com o cortdeco de `synthetic_cortdeco`."""
    return FakeMapcut(
        tamanho_corte=SYNTHETIC_RECORD_SIZE,
        numero_iteracoes=SYNTHETIC_CUTS_PER_NODE,
        numero_cortes=SYNTHETIC_CUTS_PER_NODE * 2,
        numero_estagios=SYNTHETIC_STAGES,
        patamares_por_estagio=[SYNTHETIC_LOAD_BLOCKS] * SYNTHETIC_STAGES,
        maximo_lag_tempo_viagem=SYNTHETIC_MAX_TRAVEL_TIME_LAG,
        codigos_uhes=list(SYNTHETIC_PLANT_CODES),
        codigos_uhes_tempo_viagem=list(SYNTHETIC_TRAVEL_TIME_PLANT_CODES),
        codigos_submercados_gnl=list(SYNTHETIC_GNL_SUBMARKET_CODES),
        registro_ultimo_corte_no=synthetic_last_cut_table(),
        # Topologia gravada como int32 e lida como float32, tal como o idecomp faz:
        # a usina 1 deflui para a posicao 2 e a usina 2 nao tem jusante.
        codigos_uhes_jusante=list(np.array([2, 0], dtype=np.int32).view(np.float32)),
    )


def synthetic_coefficient(record_index: int, coefficient_index: int) -> float:
    """Valor deterministico de um coeficiente, unico por (registro, coeficiente).

    Codificar a posicao no proprio valor permite afirmar, no teste, que cada
    coeficiente veio do registro certo - o que nao seria possivel com valores
    aleatorios ou repetidos.
    """
    return record_index * 100.0 + coefficient_index


def write_synthetic_cortdeco(path: Path, padding_byte: int = 0) -> None:
    """Grava um cortdeco sintetico de 7 registros.

    Cada registro contem o indice do proximo corte do mesmo no (`int32`), os
    coeficientes (`float64`) e o preenchimento nulo ate completar o registro.

    Args:
        path: destino do arquivo.
        padding_byte: valor usado no preenchimento. Diferente de zero, serve para
            testar a deteccao de coeficientes truncados.
    """
    next_record = {0: 0}  # registro 0 nao existe; mantem o dicionario homogeneo
    for chain in SYNTHETIC_CHAINS.values():
        for position, record in enumerate(chain):
            has_next = position + 1 < len(chain)
            next_record[record] = chain[position + 1] if has_next else 0

    used = 4 + 8 * SYNTHETIC_COEFFICIENT_COUNT
    padding = bytes([padding_byte]) * (SYNTHETIC_RECORD_SIZE - used)
    with path.open("wb") as handle:
        for record in range(1, SYNTHETIC_TOTAL_RECORDS + 1):
            handle.write(np.int32(next_record[record]).tobytes())
            handle.write(
                np.array(
                    [
                        synthetic_coefficient(record, index)
                        for index in range(SYNTHETIC_COEFFICIENT_COUNT)
                    ],
                    dtype=np.float64,
                ).tobytes()
            )
            handle.write(padding)


@pytest.fixture
def synthetic_cortdeco(tmp_path: Path) -> Path:
    """Cortdeco sintetico valido, com preenchimento nulo."""
    path = tmp_path / "cortdeco.sintetico"
    write_synthetic_cortdeco(path)
    return path


@pytest.fixture
def quiet_logger() -> logging.Logger:
    """Logger que nao imprime nada, para nao poluir a saida dos testes."""
    logger = logging.getLogger("teste_leitor_mapcut_cortdeco")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


@pytest.fixture
def caplog_logger(caplog: pytest.LogCaptureFixture) -> logging.Logger:
    """Logger cujas mensagens ficam disponiveis em `caplog`."""
    logger = logging.getLogger("teste_leitor_capturado")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    caplog.set_level(logging.DEBUG, logger=logger.name)
    return logger


def valid_settings_dict() -> dict[str, Any]:
    """Configuracao minima e valida.

    Todos os caminhos sao relativos, portanto ela se ancora sozinha no
    diretorio onde o settings.json for gravado.
    """
    return {
        "paths": {
            "input_directory": ".",
            "mapcut_file": "mapcut.rv0",
            "cortdeco_file": "cortdeco.rv0",
            "output_directory": "output",
        },
        "logging": {
            "level": "INFO",
            "directory": "logs",
            "filename": "leitor.log",
            "max_bytes": 1024,
            "backup_count": 1,
            "use_rich": False,
        },
        "export": {
            "separator": ",",
            "decimal": ".",
            "float_format": None,
            "encoding": "utf-8",
            "write_mapcut": True,
            "write_cortdeco_wide": True,
            "write_cortdeco_tidy": True,
        },
        "cortdeco": {
            "cuts_per_node_source": "numero_iteracoes",
            "cuts_per_node_override": None,
        },
    }


@pytest.fixture
def settings_file(tmp_path: Path) -> Path:
    """Grava um settings.json valido em um diretorio temporario."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(valid_settings_dict(), indent=2), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def real_mapcut_path() -> Path:
    """Caminho do mapcut real; ignora o teste quando ele nao esta no repositorio."""
    path = PROJECT_ROOT / "mapcut.rv0"
    if not path.is_file():
        pytest.skip(f"binario real ausente: {path}")
    return path


@pytest.fixture(scope="session")
def real_cortdeco_path() -> Path:
    """Caminho do cortdeco real; ignora o teste quando ele nao esta no repositorio."""
    path = PROJECT_ROOT / "cortdeco.rv0"
    if not path.is_file():
        pytest.skip(f"binario real ausente: {path}")
    return path
