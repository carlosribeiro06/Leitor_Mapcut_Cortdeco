"""Testes da derivacao e conferencia da geometria dos cortes.

Uma geometria errada nao faz a leitura do cortdeco falhar - ela devolve numeros
plausiveis e errados. Por isso os testes cobrem, um a um, cada mecanismo de
deteccao: tamanho de arquivo incompativel, contagem de registros divergente,
preenchimento nao nulo e divergencia entre as duas vias de obtencao do numero de
cortes por no.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import (
    SYNTHETIC_COEFFICIENT_COUNT,
    SYNTHETIC_CUTS_PER_NODE,
    SYNTHETIC_RECORD_SIZE,
    SYNTHETIC_TOTAL_RECORDS,
    FakeMapcut,
    write_synthetic_cortdeco,
)
from leitor_mapcut_cortdeco.config import CortdecoSettings
from leitor_mapcut_cortdeco.geometry import (
    GeometryError,
    build_cut_geometry,
    downstream_plant_indices,
    validate_geometry,
)

DEFAULT_CORTDECO_SETTINGS = CortdecoSettings(
    cuts_per_node_source="numero_iteracoes", cuts_per_node_override=None
)


def _build(
    mapcut: FakeMapcut,
    path: Path,
    logger: logging.Logger,
    settings: CortdecoSettings = DEFAULT_CORTDECO_SETTINGS,
):
    """Atalho para montar a geometria nos testes."""
    return build_cut_geometry(mapcut, path, settings, logger)


def test_geometria_do_caso_sintetico(
    fake_mapcut: FakeMapcut, synthetic_cortdeco: Path, quiet_logger: logging.Logger
) -> None:
    geometry = _build(fake_mapcut, synthetic_cortdeco, quiet_logger)

    assert geometry.record_size_bytes == SYNTHETIC_RECORD_SIZE
    assert geometry.coefficient_count == SYNTHETIC_COEFFICIENT_COUNT
    assert geometry.storage_coefficient_count == 2
    assert geometry.travel_time_coefficient_count == 2
    assert geometry.gnl_coefficient_count == 6
    assert geometry.used_bytes_per_record == 4 + 8 * SYNTHETIC_COEFFICIENT_COUNT
    assert geometry.padding_bytes_per_record == SYNTHETIC_RECORD_SIZE - 92
    assert geometry.cut_building_node_count == 2
    assert geometry.total_cut_records == SYNTHETIC_TOTAL_RECORDS
    assert geometry.cuts_per_node == SYNTHETIC_CUTS_PER_NODE
    assert geometry.expected_total_records == SYNTHETIC_TOTAL_RECORDS


def test_as_duas_vias_concordam_no_caso_sintetico(
    fake_mapcut: FakeMapcut, synthetic_cortdeco: Path, quiet_logger: logging.Logger
) -> None:
    """`numero_iteracoes` e a geometria do arquivo devem dar o mesmo valor."""
    geometry = _build(fake_mapcut, synthetic_cortdeco, quiet_logger)

    assert geometry.cuts_per_node_from_iterations == SYNTHETIC_CUTS_PER_NODE
    assert geometry.cuts_per_node_from_file == SYNTHETIC_CUTS_PER_NODE
    assert geometry.cuts_per_node_origin == "numero_iteracoes"


def test_divergencia_entre_as_vias_gera_aviso(
    fake_mapcut: FakeMapcut,
    synthetic_cortdeco: Path,
    caplog_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Se as duas vias discordam, a execucao segue mas o log tem de registrar."""
    fake_mapcut.numero_iteracoes = SYNTHETIC_CUTS_PER_NODE + 5

    geometry = _build(fake_mapcut, synthetic_cortdeco, caplog_logger)

    assert "DIVERGENCIA" in caplog.text
    assert geometry.cuts_per_node == SYNTHETIC_CUTS_PER_NODE + 5


def test_fonte_file_geometry_usa_o_tamanho_do_arquivo(
    fake_mapcut: FakeMapcut, synthetic_cortdeco: Path, quiet_logger: logging.Logger
) -> None:
    fake_mapcut.numero_iteracoes = 999
    settings = replace(DEFAULT_CORTDECO_SETTINGS, cuts_per_node_source="file_geometry")

    geometry = _build(fake_mapcut, synthetic_cortdeco, quiet_logger, settings)

    assert geometry.cuts_per_node == SYNTHETIC_CUTS_PER_NODE
    assert geometry.cuts_per_node_origin == "file_geometry"


def test_override_tem_precedencia_e_gera_aviso(
    fake_mapcut: FakeMapcut,
    synthetic_cortdeco: Path,
    caplog_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = replace(DEFAULT_CORTDECO_SETTINGS, cuts_per_node_override=7)

    geometry = _build(fake_mapcut, synthetic_cortdeco, caplog_logger, settings)

    assert geometry.cuts_per_node == 7
    assert geometry.cuts_per_node_origin == "cuts_per_node_override"
    assert "fixado manualmente" in caplog.text


def test_tamanho_de_arquivo_nao_multiplo_do_registro(
    fake_mapcut: FakeMapcut, synthetic_cortdeco: Path, quiet_logger: logging.Logger
) -> None:
    """Sinaliza par mapcut/cortdeco de casos diferentes."""
    with synthetic_cortdeco.open("ab") as handle:
        handle.write(b"\x00" * 3)

    with pytest.raises(GeometryError, match="nao e multiplo"):
        _build(fake_mapcut, synthetic_cortdeco, quiet_logger)


def test_patamares_nao_uniformes_interrompem_a_execucao(
    fake_mapcut: FakeMapcut, synthetic_cortdeco: Path, quiet_logger: logging.Logger
) -> None:
    """A largura do bloco GNL nao pode ser deduzida; parar e melhor que adivinhar."""
    fake_mapcut.patamares_por_estagio = [2, 3, 2]

    with pytest.raises(GeometryError, match="patamares de carga diferente"):
        _build(fake_mapcut, synthetic_cortdeco, quiet_logger)


def test_tamanho_de_registro_invalido(
    fake_mapcut: FakeMapcut, synthetic_cortdeco: Path, quiet_logger: logging.Logger
) -> None:
    fake_mapcut.tamanho_corte = 0

    with pytest.raises(GeometryError, match="tamanho de registro invalido"):
        _build(fake_mapcut, synthetic_cortdeco, quiet_logger)


def test_validacao_aceita_o_caso_sintetico(
    fake_mapcut: FakeMapcut, synthetic_cortdeco: Path, quiet_logger: logging.Logger
) -> None:
    geometry = _build(fake_mapcut, synthetic_cortdeco, quiet_logger)

    validate_geometry(geometry, synthetic_cortdeco, quiet_logger)


def test_preenchimento_nao_nulo_e_erro(
    fake_mapcut: FakeMapcut, tmp_path: Path, quiet_logger: logging.Logger
) -> None:
    """Preenchimento sujo significa que ha coeficientes alem dos contabilizados."""
    path = tmp_path / "cortdeco_sujo"
    write_synthetic_cortdeco(path, padding_byte=0x7F)

    geometry = _build(fake_mapcut, path, quiet_logger)

    with pytest.raises(GeometryError, match="bytes nao nulos"):
        validate_geometry(geometry, path, quiet_logger)


def test_coeficientes_maiores_que_o_registro(
    fake_mapcut: FakeMapcut, synthetic_cortdeco: Path, quiet_logger: logging.Logger
) -> None:
    """Uma contagem de coeficientes exagerada invadiria o registro seguinte."""
    fake_mapcut.codigos_uhes = list(range(1, 200))

    geometry = _build(fake_mapcut, synthetic_cortdeco, quiet_logger)

    with pytest.raises(GeometryError, match="mais do que o registro"):
        validate_geometry(geometry, synthetic_cortdeco, quiet_logger)


def test_contagem_de_registros_divergente_gera_aviso(
    fake_mapcut: FakeMapcut,
    synthetic_cortdeco: Path,
    caplog_logger: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A leitura ainda funciona (a lista encadeada tem terminador), mas fica no log."""
    settings = replace(DEFAULT_CORTDECO_SETTINGS, cuts_per_node_override=5)
    geometry = _build(fake_mapcut, synthetic_cortdeco, caplog_logger, settings)

    validate_geometry(geometry, synthetic_cortdeco, caplog_logger)

    assert "contagem de registros inesperada" in caplog.text


def test_topologia_de_jusante_reinterpreta_os_bits(fake_mapcut: FakeMapcut) -> None:
    """Correcao do defeito float32/int32 do idecomp no registro 4 do mapcut."""
    indices = downstream_plant_indices(fake_mapcut)

    assert indices.tolist() == [2, 0]
    assert indices.dtype == np.int64
