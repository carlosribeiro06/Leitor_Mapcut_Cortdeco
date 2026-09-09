"""Testes das tabelas dos blocos opcionais, com dubles.

Por que este modulo existe
--------------------------
Os dois blocos opcionais do corte - tempo de viagem e GNL - produzem tabelas
vazias quando o caso nao os tem. Vazias, porem, **com as colunas e os dtypes
declarados**: um `pd.DataFrame([])` sai tambem sem colunas, e o CSV
correspondente ficaria sem cabecalho, que e a mesma patologia do `idecomp` que
este projeto corrige.

Esse cenario nao pode ser coberto por `test_mapcut_reader.py`, que roda sobre o
deck presente no repositorio: nenhum dos decks disponiveis tem bloco ausente, e a
cobertura ficaria a merce de qual caso esta em disco. Aqui os casos sao montados
com dubles, e portanto valem sempre.

Os construtores de tabela sao privados de proposito - eles nao fazem parte da
interface do modulo -, e sao acessados aqui porque e o nivel em que o contrato
"vazia mas com cabecalho" existe.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import cast

import pandas as pd
from idecomp.decomp.mapcut import Mapcut

from conftest import FakeMapcut
from leitor_mapcut_cortdeco.mapcut_reader import (
    GNL_DTYPES,
    GNL_VALUE_DTYPES,
    TRAVEL_TIME_DTYPES,
    TRAVEL_TIME_LAGS_DTYPES,
    _gnl_tables,
    _travel_time_lags_table,
    _travel_time_table,
)


def _mapcut(fake: FakeMapcut) -> Mapcut:
    """Trata o duble como um `Mapcut` para o verificador de tipos."""
    return cast("Mapcut", fake)


def _assert_vazia_com_contrato(
    table: pd.DataFrame, dtypes: Mapping[str, str], rotulo: str
) -> None:
    """Afirma que a tabela esta vazia, mas conserva colunas e dtypes."""
    assert table.empty, rotulo
    assert list(table.columns) == list(dtypes), rotulo
    assert [str(dtype) for dtype in table.dtypes] == list(dtypes.values()), rotulo


def test_tabelas_de_tempo_viagem_sem_o_bloco(
    fake_mapcut_sem_tempo_viagem: FakeMapcut, quiet_logger: logging.Logger
) -> None:
    mapcut = _mapcut(fake_mapcut_sem_tempo_viagem)

    _assert_vazia_com_contrato(
        _travel_time_table(mapcut, quiet_logger),
        TRAVEL_TIME_DTYPES,
        "mapcut_tempo_viagem",
    )
    _assert_vazia_com_contrato(
        _travel_time_lags_table(mapcut, quiet_logger),
        TRAVEL_TIME_LAGS_DTYPES,
        "mapcut_tempo_viagem_lags",
    )


def test_tabelas_de_gnl_sem_o_bloco(
    fake_mapcut_sem_gnl: FakeMapcut, quiet_logger: logging.Logger
) -> None:
    gnl_table, value_table = _gnl_tables(_mapcut(fake_mapcut_sem_gnl), quiet_logger)

    _assert_vazia_com_contrato(gnl_table, GNL_DTYPES, "mapcut_gnl")
    _assert_vazia_com_contrato(
        value_table, GNL_VALUE_DTYPES, "mapcut_gnl_bloco_valores"
    )


def test_tabelas_de_gnl_com_o_bloco_conservam_os_mesmos_dtypes(
    fake_mapcut: FakeMapcut, quiet_logger: logging.Logger
) -> None:
    """O contrato tem de ser o mesmo com e sem dados - senao nao e contrato."""
    gnl_table, value_table = _gnl_tables(_mapcut(fake_mapcut), quiet_logger)

    assert not gnl_table.empty
    assert list(gnl_table.columns) == list(GNL_DTYPES)
    assert [str(d) for d in gnl_table.dtypes] == list(GNL_DTYPES.values())
    assert list(value_table.columns) == list(GNL_VALUE_DTYPES)
    assert [str(d) for d in value_table.dtypes] == list(GNL_VALUE_DTYPES.values())


def test_tabelas_de_tempo_viagem_com_o_bloco_conservam_os_mesmos_dtypes(
    fake_mapcut: FakeMapcut, quiet_logger: logging.Logger
) -> None:
    travel_time = _travel_time_table(_mapcut(fake_mapcut), quiet_logger)
    lags = _travel_time_lags_table(_mapcut(fake_mapcut), quiet_logger)

    assert not travel_time.empty
    assert list(travel_time.columns) == list(TRAVEL_TIME_DTYPES)
    assert [str(d) for d in travel_time.dtypes] == list(TRAVEL_TIME_DTYPES.values())
    assert not lags.empty
    assert list(lags.columns) == list(TRAVEL_TIME_LAGS_DTYPES)
    assert [str(d) for d in lags.dtypes] == list(TRAVEL_TIME_LAGS_DTYPES.values())
