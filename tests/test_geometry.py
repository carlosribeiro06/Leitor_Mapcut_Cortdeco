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
from typing import cast

import numpy as np
import pytest
from idecomp.decomp.mapcut import Mapcut

from conftest import (
    SYNTHETIC_COEFFICIENT_COUNT,
    SYNTHETIC_COEFFICIENT_COUNT_GNL_MAIOR,
    SYNTHETIC_COEFFICIENT_COUNT_WITHOUT_GNL,
    SYNTHETIC_COEFFICIENT_COUNT_WITHOUT_TRAVEL_TIME,
    SYNTHETIC_CUTS_PER_NODE,
    SYNTHETIC_GNL_SUBMARKET_CODES,
    SYNTHETIC_GNL_SUBMARKET_CODES_MAIOR,
    SYNTHETIC_LOAD_BLOCKS,
    SYNTHETIC_LOAD_BLOCKS_MAIOR,
    SYNTHETIC_MAX_TRAVEL_TIME_LAG,
    SYNTHETIC_RECORD_SIZE,
    SYNTHETIC_STAGES,
    SYNTHETIC_TOTAL_RECORDS,
    SYNTHETIC_TRAVEL_TIME_HOURS,
    SYNTHETIC_TRAVEL_TIME_PLANT_CODES,
    SYNTHETIC_TRAVEL_TIME_STAGE_LAGS,
    FakeMapcut,
    synthetic_gnl_payload,
    write_synthetic_cortdeco,
)
from leitor_mapcut_cortdeco.config import CortdecoSettings
from leitor_mapcut_cortdeco.geometry import (
    CutGeometry,
    GeometryError,
    build_cut_geometry,
    downstream_plant_indices,
    gnl_stage_blocks,
    gnl_submarket_codes,
    travel_time_blocks,
    travel_time_lags,
    travel_time_plant_codes,
    travel_time_register_counts,
    validate_geometry,
)
from leitor_mapcut_cortdeco.mapcut_reader import (
    # Helper privado, importado de proposito: e ele que reproduz a aritmetica
    # defeituosa do idecomp, e o teste de regressao precisa provar que a condicao
    # de falha esta de fato montada, e nao apenas que o resultado saiu certo.
    _idecomp_gnl_frame_is_evaluable,
)

DEFAULT_CORTDECO_SETTINGS = CortdecoSettings(
    cuts_per_node_source="numero_iteracoes", cuts_per_node_override=None
)


def _mapcut(fake: FakeMapcut) -> Mapcut:
    """Trata o duble como um `Mapcut` para o verificador de tipos.

    `FakeMapcut` implementa de proposito apenas o subconjunto do `Mapcut` que a
    geometria consome, e nao herda dele. Concentrar a conversao aqui mantem o
    mypy util no resto do arquivo, em vez de espalhar um `type: ignore` por
    chamada.
    """
    return cast("Mapcut", fake)


def _build(
    mapcut: FakeMapcut,
    path: Path,
    logger: logging.Logger,
    settings: CortdecoSettings = DEFAULT_CORTDECO_SETTINGS,
) -> CutGeometry:
    """Atalho para montar a geometria nos testes."""
    return build_cut_geometry(_mapcut(mapcut), path, settings, logger)


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
    indices = downstream_plant_indices(_mapcut(fake_mapcut))

    assert indices.tolist() == [2, 0]
    assert indices.dtype == np.int64


# ---------------------------------------------------------------------------
# Tempo de viagem da agua
# ---------------------------------------------------------------------------


def test_blocos_de_tempo_viagem_vem_do_payload_bruto(fake_mapcut: FakeMapcut) -> None:
    """Os codigos das UHEs sao decodificados do payload, nao da propriedade."""
    blocks = travel_time_blocks(_mapcut(fake_mapcut))

    assert len(blocks) == 1
    block = blocks[0]
    assert block.plant_code == SYNTHETIC_TRAVEL_TIME_PLANT_CODES[0]
    assert block.travel_hours == SYNTHETIC_TRAVEL_TIME_HOURS
    assert block.stage_lags == SYNTHETIC_TRAVEL_TIME_STAGE_LAGS
    # Um par (indice_lag, coeficiente) por lag de 0 a lag_maximo, em cada estagio.
    assert block.pair_count == sum(lag + 1 for lag in SYNTHETIC_TRAVEL_TIME_STAGE_LAGS)
    assert (
        travel_time_plant_codes(_mapcut(fake_mapcut))
        == SYNTHETIC_TRAVEL_TIME_PLANT_CODES
    )


def test_caso_sem_tempo_viagem_produz_bloco_de_largura_zero(
    fake_mapcut_sem_tempo_viagem: FakeMapcut,
    synthetic_cortdeco_sem_tempo_viagem: Path,
    quiet_logger: logging.Logger,
) -> None:
    """Zero UHEs com tempo de viagem e caso legitimo, nao arquivo corrompido."""
    geometry = _build(
        fake_mapcut_sem_tempo_viagem,
        synthetic_cortdeco_sem_tempo_viagem,
        quiet_logger,
    )

    assert geometry.travel_time_plant_codes == ()
    assert geometry.travel_time_coefficient_count == 0
    assert geometry.coefficient_count == SYNTHETIC_COEFFICIENT_COUNT_WITHOUT_TRAVEL_TIME
    # O corte vai direto do volume armazenado para a geracao GNL.
    assert geometry.storage_coefficient_count == 2
    assert geometry.gnl_coefficient_count == 6

    validate_geometry(geometry, synthetic_cortdeco_sem_tempo_viagem, quiet_logger)


def test_caso_sem_tempo_viagem_ignora_os_lags_espurios_do_idecomp(
    fake_mapcut_sem_tempo_viagem: FakeMapcut,
) -> None:
    """Com nutv = 0 o idecomp devolve valores de outro campo; eles nao entram."""
    assert fake_mapcut_sem_tempo_viagem.lag_tempo_viagem_por_uhe

    assert travel_time_lags(_mapcut(fake_mapcut_sem_tempo_viagem)) == []
    assert travel_time_blocks(_mapcut(fake_mapcut_sem_tempo_viagem)) == []
    assert travel_time_plant_codes(_mapcut(fake_mapcut_sem_tempo_viagem)) == ()


def test_propriedade_defeituosa_do_idecomp_nao_e_consultada(
    fake_mapcut: FakeMapcut, synthetic_cortdeco: Path, quiet_logger: logging.Logger
) -> None:
    """O duble levanta em `codigos_uhes_tempo_viagem`; a geometria tem de passar."""
    with pytest.raises(KeyError):
        _ = fake_mapcut.codigos_uhes_tempo_viagem

    geometry = _build(fake_mapcut, synthetic_cortdeco, quiet_logger)

    assert geometry.travel_time_plant_codes == SYNTHETIC_TRAVEL_TIME_PLANT_CODES
    assert geometry.travel_time_coefficient_count == (
        len(SYNTHETIC_TRAVEL_TIME_PLANT_CODES) * SYNTHETIC_MAX_TRAVEL_TIME_LAG
    )


def test_vetor_de_lags_com_tamanho_divergente(fake_mapcut: FakeMapcut) -> None:
    """O vetor tem de ter numero_uhes_tempo_viagem x numero_estagios valores."""
    fake_mapcut.lag_tempo_viagem_por_uhe = [SYNTHETIC_MAX_TRAVEL_TIME_LAG]

    with pytest.raises(GeometryError, match="eram esperados 3"):
        travel_time_lags(_mapcut(fake_mapcut))


def test_payload_de_tempo_viagem_com_sobra(fake_mapcut: FakeMapcut) -> None:
    """Sobra no payload significa que o layout assumido nao corresponde."""
    fake_mapcut.data.section.data["dados_tempo_viagem"] += [0, 0.0]

    with pytest.raises(GeometryError, match="nao corresponde"):
        travel_time_blocks(_mapcut(fake_mapcut))


def test_payload_de_tempo_viagem_truncado(fake_mapcut: FakeMapcut) -> None:
    """Payload curto tambem denuncia layout ou contagem errados."""
    fake_mapcut.data.section.data["dados_tempo_viagem"] = [
        SYNTHETIC_TRAVEL_TIME_PLANT_CODES[0],
        SYNTHETIC_TRAVEL_TIME_HOURS,
    ]

    with pytest.raises(GeometryError, match="exigiria"):
        travel_time_blocks(_mapcut(fake_mapcut))


def test_uhe_com_tempo_viagem_fora_da_lista_de_uhes(fake_mapcut: FakeMapcut) -> None:
    """Um codigo que nao e UHE do caso denuncia layout errado."""
    fake_mapcut.data.section.data["dados_tempo_viagem"][0] = 999

    with pytest.raises(GeometryError, match="nao estao entre as UHEs"):
        travel_time_plant_codes(_mapcut(fake_mapcut))


def test_numero_de_uhes_com_tempo_viagem_negativo(fake_mapcut: FakeMapcut) -> None:
    fake_mapcut.numero_uhes_tempo_viagem = -1

    with pytest.raises(GeometryError, match="negativo"):
        travel_time_plant_codes(_mapcut(fake_mapcut))


def test_leitura_desalinhada_da_seccao_de_tempo_viagem_interrompe(
    fake_mapcut: FakeMapcut, synthetic_cortdeco: Path, quiet_logger: logging.Logger
) -> None:
    """Lags nao uniformes desalinham TODOS os registros seguintes do mapcut.

    O idecomp indexa o vetor de lags por (usina * estagio), e nao por
    (usina - 1) * numero_estagios + estagio. Com dois lags diferentes ele le um
    numero errado de registros de 48 KB, e os registros de GNL e de custos passam
    a ser lidos de outro trecho do arquivo - sem excecao e sem sintoma. A
    geometria tem de parar antes de derivar qualquer coisa desses payloads.
    """
    fake_mapcut.numero_uhes_tempo_viagem = 2
    fake_mapcut.lag_tempo_viagem_por_uhe = [1, 1, 1, 2, 2, 2]

    actual, consumed = travel_time_register_counts(_mapcut(fake_mapcut))
    assert (actual, consumed) == (15, 14)

    with pytest.raises(GeometryError, match="registros de tempo de viagem"):
        _build(fake_mapcut, synthetic_cortdeco, quiet_logger)


def test_leitura_alinhada_quando_os_lags_sao_uniformes(
    fake_mapcut: FakeMapcut,
) -> None:
    """O caso comum: lags iguais fazem as duas indexacoes coincidirem."""
    actual, consumed = travel_time_register_counts(_mapcut(fake_mapcut))

    assert actual == consumed
    assert actual == sum(lag + 1 for lag in SYNTHETIC_TRAVEL_TIME_STAGE_LAGS)


# ---------------------------------------------------------------------------
# Geracao GNL antecipada
# ---------------------------------------------------------------------------


def test_submercados_de_gnl_vem_do_payload_bruto(fake_mapcut: FakeMapcut) -> None:
    """Os codigos sao decodificados do registro 9, nao da propriedade."""
    blocks = gnl_stage_blocks(_mapcut(fake_mapcut))

    assert len(blocks) == SYNTHETIC_STAGES
    assert [block.stage for block in blocks] == list(range(1, SYNTHETIC_STAGES + 1))
    for block in blocks:
        assert block.plant_count == len(SYNTHETIC_GNL_SUBMARKET_CODES)
        assert block.submarket_codes == SYNTHETIC_GNL_SUBMARKET_CODES
        assert block.load_block_counts == (SYNTHETIC_LOAD_BLOCKS,)
    # Os codigos se repetem a cada estagio e sao reduzidos preservando a ordem.
    assert gnl_submarket_codes(_mapcut(fake_mapcut)) == SYNTHETIC_GNL_SUBMARKET_CODES


def test_propriedade_de_gnl_do_idecomp_nao_e_consultada(
    fake_mapcut: FakeMapcut, synthetic_cortdeco: Path, quiet_logger: logging.Logger
) -> None:
    """O duble levanta em `codigos_submercados_gnl`; a geometria tem de passar."""
    with pytest.raises(IndexError):
        _ = fake_mapcut.codigos_submercados_gnl

    geometry = _build(fake_mapcut, synthetic_cortdeco, quiet_logger)

    assert geometry.gnl_submarket_codes == SYNTHETIC_GNL_SUBMARKET_CODES


def test_caso_sem_gnl_produz_bloco_de_largura_zero(
    fake_mapcut_sem_gnl: FakeMapcut,
    synthetic_cortdeco_sem_gnl: Path,
    quiet_logger: logging.Logger,
) -> None:
    """Nenhuma UTE a GNL e caso legitimo: o corte vai do tempo de viagem ao fim."""
    geometry = _build(fake_mapcut_sem_gnl, synthetic_cortdeco_sem_gnl, quiet_logger)

    assert geometry.gnl_submarket_codes == ()
    assert geometry.gnl_coefficient_count == 0
    assert geometry.coefficient_count == SYNTHETIC_COEFFICIENT_COUNT_WITHOUT_GNL
    assert geometry.travel_time_coefficient_count == 2

    validate_geometry(geometry, synthetic_cortdeco_sem_gnl, quiet_logger)


def test_payload_de_gnl_inteiramente_vazio_tambem_e_aceito(
    fake_mapcut_sem_gnl: FakeMapcut,
) -> None:
    """A outra forma possivel do caso sem GNL: nenhum registro 9 no payload."""
    fake_mapcut_sem_gnl.data.section.data["dados_gnl"] = []

    assert gnl_stage_blocks(_mapcut(fake_mapcut_sem_gnl)) == []
    assert gnl_submarket_codes(_mapcut(fake_mapcut_sem_gnl)) == ()


def test_payload_de_gnl_com_sobra(fake_mapcut: FakeMapcut) -> None:
    """Sobra no payload significa que o layout assumido nao corresponde."""
    fake_mapcut.data.section.data["dados_gnl"] += [0]

    with pytest.raises(GeometryError, match="nao corresponde"):
        gnl_stage_blocks(_mapcut(fake_mapcut))


def test_payload_de_gnl_acaba_antes_do_ultimo_estagio(
    fake_mapcut: FakeMapcut,
) -> None:
    """Payload que termina em fronteira de registro, faltando estagios."""
    completo = synthetic_gnl_payload()
    por_estagio = len(completo) // SYNTHETIC_STAGES
    fake_mapcut.data.section.data["dados_gnl"] = completo[: 2 * por_estagio]

    with pytest.raises(GeometryError, match="acabou no estagio 3 de 3"):
        gnl_stage_blocks(_mapcut(fake_mapcut))


def test_registro_de_gnl_nao_cabe_no_payload(fake_mapcut: FakeMapcut) -> None:
    """Payload que termina no meio de um registro."""
    fake_mapcut.data.section.data["dados_gnl"] = synthetic_gnl_payload(stage_count=1)

    with pytest.raises(GeometryError, match="exigiria 7 valores"):
        gnl_stage_blocks(_mapcut(fake_mapcut))


def test_mais_submercados_com_gnl_que_submercados_do_caso(
    fake_mapcut: FakeMapcut,
) -> None:
    """Mais submercados distintos que o caso tem denuncia layout errado."""
    fake_mapcut.numero_submercados = 1
    fake_mapcut.data.section.data["dados_gnl"] = synthetic_gnl_payload(
        submarket_codes=(1, 3)
    )

    with pytest.raises(GeometryError, match="mais do que os 1 submercados"):
        gnl_submarket_codes(_mapcut(fake_mapcut))


def test_poucos_estagios_derrubam_o_idecomp_mas_nao_a_leitura(
    fake_mapcut_gnl_maior: FakeMapcut,
    synthetic_cortdeco_gnl_maior: Path,
    quiet_logger: logging.Logger,
) -> None:
    """Regressao do deck rv4: passo errado do idecomp saindo do payload.

    Com poucos estagios e mais de um submercado, o passo errado da biblioteca cai
    no meio do registro seguinte e depois estoura o payload - a propriedade
    levanta IndexError. A leitura nao pode depender dela, e a geometria tem de
    sair com o numero certo de submercados.
    """
    mapcut = _mapcut(fake_mapcut_gnl_maior)

    # A condicao esta de fato reproduzida: a propriedade nao e avaliavel.
    assert not _idecomp_gnl_frame_is_evaluable(mapcut)
    with pytest.raises(IndexError):
        _ = fake_mapcut_gnl_maior.codigos_submercados_gnl

    # E, apesar disso, a derivacao do payload bruto entrega o resultado certo.
    assert gnl_submarket_codes(mapcut) == SYNTHETIC_GNL_SUBMARKET_CODES_MAIOR

    geometry = _build(fake_mapcut_gnl_maior, synthetic_cortdeco_gnl_maior, quiet_logger)

    assert geometry.gnl_submarket_codes == SYNTHETIC_GNL_SUBMARKET_CODES_MAIOR
    assert geometry.gnl_coefficient_count == (
        len(SYNTHETIC_GNL_SUBMARKET_CODES_MAIOR)
        * SYNTHETIC_STAGES
        * SYNTHETIC_LOAD_BLOCKS_MAIOR
    )
    assert geometry.coefficient_count == SYNTHETIC_COEFFICIENT_COUNT_GNL_MAIOR

    validate_geometry(geometry, synthetic_cortdeco_gnl_maior, quiet_logger)


def test_passo_do_idecomp_coincide_no_caso_sintetico_padrao(
    fake_mapcut: FakeMapcut,
) -> None:
    """Contraprova: com nper == patamares + 1 os dois passos coincidem.

    Serve para provar que o teste acima mede a condicao certa, e nao um efeito
    colateral do duble.
    """
    assert SYNTHETIC_STAGES == SYNTHETIC_LOAD_BLOCKS + 1
    assert _idecomp_gnl_frame_is_evaluable(_mapcut(fake_mapcut))


def test_submercado_repetido_no_mesmo_estagio(fake_mapcut: FakeMapcut) -> None:
    fake_mapcut.data.section.data["dados_gnl"] = synthetic_gnl_payload(
        submarket_codes=(1, 1)
    )

    with pytest.raises(GeometryError, match="repete submercados"):
        gnl_submarket_codes(_mapcut(fake_mapcut))
