"""Testes da leitura do cortdeco e das tabelas derivadas.

O caso sintetico permite a afirmacao mais forte possivel: como cada coeficiente
gravado carrega no proprio valor o registro e a posicao de onde veio
(`registro * 100 + posicao`), o teste confere que **cada** coeficiente lido veio
do registro certo - e nao apenas que a tabela tem o tamanho esperado.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from conftest import (
    SYNTHETIC_CHAINS,
    SYNTHETIC_COEFFICIENT_COUNT,
    FakeMapcut,
    synthetic_coefficient,
)
from leitor_mapcut_cortdeco.config import CortdecoSettings
from leitor_mapcut_cortdeco.cortdeco_reader import (
    CUT_KEY_COLUMNS,
    GNL_COLUMN_PATTERN,
    build_cortdeco_tables,
    read_cortdeco,
    validate_cuts,
)
from leitor_mapcut_cortdeco.geometry import build_cut_geometry

DEFAULT_CORTDECO_SETTINGS = CortdecoSettings(
    cuts_per_node_source="numero_iteracoes", cuts_per_node_override=None
)

# O idecomp inverte a numeracao dos cortes: o registro alcancado primeiro (o
# ultimo corte gravado pelo no) recebe o maior `indice_corte`. Este mapa traduz
# (no, indice_corte) -> registro esperado no arquivo sintetico.
EXPECTED_RECORD_BY_CUT = {
    (no, len(chain) - position): record
    for no, chain in SYNTHETIC_CHAINS.items()
    for position, record in enumerate(chain)
}


@pytest.fixture
def read_synthetic(
    fake_mapcut: FakeMapcut, synthetic_cortdeco: Path, quiet_logger: logging.Logger
) -> tuple[pd.DataFrame, object]:
    """Le o cortdeco sintetico e devolve a tabela de cortes junto com a geometria."""
    geometry = build_cut_geometry(
        fake_mapcut, synthetic_cortdeco, DEFAULT_CORTDECO_SETTINGS, quiet_logger
    )
    cortdeco = read_cortdeco(synthetic_cortdeco, geometry, quiet_logger)
    return cortdeco.cortes, geometry


def test_le_todos_os_cortes_do_caso_sintetico(
    read_synthetic: tuple[pd.DataFrame, object],
) -> None:
    cuts, _ = read_synthetic

    # 3 cortes no no 1 + 4 no no 2 (o ultimo estagio que constroi cortes tem o
    # registro extra gravado pelo DECOMP).
    assert len(cuts) == 7
    assert cuts.shape[1] == 3 + SYNTHETIC_COEFFICIENT_COUNT
    assert cuts.groupby("no").size().to_dict() == {1: 3, 2: 4}


def test_ordem_das_colunas_de_coeficientes(
    read_synthetic: tuple[pd.DataFrame, object],
) -> None:
    """Os tres blocos aparecem na ordem em que estao gravados no registro."""
    cuts, _ = read_synthetic

    assert list(cuts.columns) == [
        "indice_corte",
        "no",
        "estagio",
        "rhs",
        "pi_varm_uhe11",
        "pi_varm_uhe22",
        "pi_qdefp_uhe22_lag0",
        "pi_qdefp_uhe22_lag1",
        "pi_gnl_sbm1_pat1_lag1",
        "pi_gnl_sbm1_pat2_lag1",
        "pi_gnl_sbm1_pat1_lag2",
        "pi_gnl_sbm1_pat2_lag2",
        "pi_gnl_sbm1_pat1_lag3",
        "pi_gnl_sbm1_pat2_lag3",
    ]


def test_cada_coeficiente_veio_do_registro_certo(
    read_synthetic: tuple[pd.DataFrame, object],
) -> None:
    """Confere valor a valor, seguindo a lista encadeada de cada no."""
    cuts, _ = read_synthetic
    coefficient_columns = [
        column for column in cuts.columns if column not in CUT_KEY_COLUMNS
    ]

    for row in cuts.itertuples(index=False):
        expected_record = EXPECTED_RECORD_BY_CUT[(row.no, row.indice_corte)]
        for position, column in enumerate(coefficient_columns):
            assert getattr(row, column) == synthetic_coefficient(
                expected_record, position
            ), f"no {row.no}, corte {row.indice_corte}, coluna {column}"


def test_validacao_nao_encontra_linhas_zeradas(
    read_synthetic: tuple[pd.DataFrame, object],
    caplog_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cuts, geometry = read_synthetic

    validate_cuts(cuts, geometry, caplog_logger)

    assert "nenhuma das 7 linhas esta inteiramente zerada" in caplog.text
    assert "WARNING" not in {record.levelname for record in caplog.records}


def test_validacao_avisa_sobre_linhas_zeradas(
    read_synthetic: tuple[pd.DataFrame, object],
    caplog_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Linhas zeradas indicam cortes por no superestimado - o defeito mais sutil."""
    cuts, geometry = read_synthetic
    padded = pd.concat(
        [
            cuts,
            cuts.tail(1).assign(
                **{
                    column: 0.0
                    for column in cuts.columns
                    if column not in ("indice_corte", "no", "estagio")
                }
            ),
        ]
    )

    validate_cuts(padded, geometry, caplog_logger)

    assert "todos os" in caplog.text
    assert "iguais a zero" in caplog.text


def test_tabela_gnl_separa_estagio_e_patamar(
    read_synthetic: tuple[pd.DataFrame, object], quiet_logger: logging.Logger
) -> None:
    """Correcao do defeito em que o rotulo `lag` saia do token `pat`."""
    cuts, geometry = read_synthetic

    tables = build_cortdeco_tables(
        _StubCortdeco(cuts),
        geometry,
        write_wide=False,
        write_tidy=True,
        logger=quiet_logger,
    )
    gnl = tables["cortdeco_coef_geracao_gnl"]

    assert len(gnl) == len(cuts) * geometry.gnl_coefficient_count
    assert sorted(gnl["estagio_coeficiente"].unique()) == [1, 2, 3]
    assert sorted(gnl["patamar"].unique()) == [1, 2]
    assert sorted(gnl["codigo_submercado"].unique()) == [1]

    # Os valores tem de casar com as colunas do formato wide, celula a celula.
    wide_columns = [c for c in cuts.columns if GNL_COLUMN_PATTERN.match(c)]
    assert pytest.approx(gnl["valor"].sum()) == cuts[wide_columns].to_numpy().sum()


def test_tabela_rhs_isola_o_termo_constante(
    read_synthetic: tuple[pd.DataFrame, object], quiet_logger: logging.Logger
) -> None:
    cuts, geometry = read_synthetic

    tables = build_cortdeco_tables(
        _StubCortdeco(cuts),
        geometry,
        write_wide=False,
        write_tidy=True,
        logger=quiet_logger,
    )

    assert list(tables["cortdeco_rhs"].columns) == [
        "indice_corte",
        "no",
        "estagio",
        "rhs",
    ]
    assert tables["cortdeco_rhs"]["rhs"].tolist() == cuts["rhs"].tolist()


def test_formato_wide_pode_ser_desligado(
    read_synthetic: tuple[pd.DataFrame, object], quiet_logger: logging.Logger
) -> None:
    cuts, geometry = read_synthetic

    tables = build_cortdeco_tables(
        _StubCortdeco(cuts),
        geometry,
        write_wide=False,
        write_tidy=False,
        logger=quiet_logger,
    )

    assert tables == {}


class _StubCortdeco:
    """Expoe a tabela de cortes com a interface que `build_cortdeco_tables` usa.

    As duas propriedades corretas do idecomp (volume armazenado e defluencia por
    tempo de viagem) sao reimplementadas aqui de forma minima, para que o teste
    das tabelas tidy nao dependa de reconstruir um objeto `Cortdeco` completo.
    """

    def __init__(self, cuts: pd.DataFrame) -> None:
        self.cortes = cuts

    def _melt(self, prefix: str) -> pd.DataFrame:
        columns = [c for c in self.cortes.columns if c.startswith(prefix)]
        return self.cortes.melt(
            id_vars=["indice_corte", "no", "estagio"],
            value_vars=columns,
            value_name="valor",
        )

    @property
    def coeficientes_volume_armazenado(self) -> pd.DataFrame:
        return self._melt("pi_varm_")

    @property
    def coeficientes_defluencia_tempo_viagem(self) -> pd.DataFrame:
        return self._melt("pi_qdefp_")

    @property
    def coeficientes_geracao_gnl(self) -> pd.DataFrame:
        return self._melt("pi_gnl_")
