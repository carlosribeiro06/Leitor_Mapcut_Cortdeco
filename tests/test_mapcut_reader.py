"""Testes das correcoes de decodificacao do mapcut, contra o binario real.

Estes testes usam o `mapcut.rv0` do repositorio, e nao um arquivo sintetico: as
tres correcoes deste modulo existem justamente porque o `idecomp` decodifica
esse layout de forma incorreta, e um dublê construido por nos so provaria que a
nossa premissa e coerente consigo mesma. Contra o arquivo real, as afirmacoes
tem conteudo - em especial a de que a reconstrucao consome exatamente todos os
valores do payload, que e o que ancora o layout assumido.

Quando os binarios nao estao presentes, os testes sao ignorados (ver conftest).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from leitor_mapcut_cortdeco.config import CortdecoSettings
from leitor_mapcut_cortdeco.geometry import build_cut_geometry
from leitor_mapcut_cortdeco.mapcut_reader import (
    TRAVEL_TIME_DTYPES,
    TRAVEL_TIME_LAGS_DTYPES,
    build_mapcut_tables,
    read_mapcut,
)

DEFAULT_CORTDECO_SETTINGS = CortdecoSettings(
    cuts_per_node_source="numero_iteracoes", cuts_per_node_override=None
)

# Tabelas que saem legitimamente vazias quando o caso nao tem UHE com tempo de
# viagem. Vazias, mas com cabecalho: veja o teste de colunas e dtypes.
TRAVEL_TIME_TABLES = (
    "mapcut_tempo_viagem",
    "mapcut_tempo_viagem_lags",
    "mapcut_tempo_viagem_idecomp_bruto",
)


def _travel_time_plant_count(real_tables: dict) -> int:
    """Numero de UHEs com tempo de viagem do deck presente no repositorio."""
    return int(real_tables["mapcut"].numero_uhes_tempo_viagem)


@pytest.fixture(scope="module")
def real_tables(request: pytest.FixtureRequest) -> dict:
    """Le o mapcut real uma unica vez e monta todas as tabelas."""
    mapcut_path = request.getfixturevalue("real_mapcut_path")
    cortdeco_path = request.getfixturevalue("real_cortdeco_path")
    logger = logging.getLogger("teste_mapcut_real")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False

    mapcut = read_mapcut(mapcut_path, logger)
    geometry = build_cut_geometry(
        mapcut, cortdeco_path, DEFAULT_CORTDECO_SETTINGS, logger
    )
    return {
        "mapcut": mapcut,
        "geometry": geometry,
        "tables": build_mapcut_tables(mapcut, geometry, logger),
    }


def test_todas_as_tabelas_esperadas_sao_geradas(real_tables: dict) -> None:
    assert set(real_tables["tables"]) == {
        "mapcut_metadados",
        "mapcut_usinas_hidraulicas",
        "mapcut_arvore_cenarios",
        "mapcut_estagios",
        "mapcut_ultimo_corte_por_no",
        "mapcut_tempo_viagem_lags",
        "mapcut_tempo_viagem",
        "mapcut_gnl",
        "mapcut_gnl_bloco_valores",
        "mapcut_submercados_gnl",
        "mapcut_custos",
        "mapcut_usinas_jusante_idecomp_bruto",
        "mapcut_tempo_viagem_idecomp_bruto",
        "mapcut_gnl_idecomp_bruto",
    }


def test_so_estao_vazias_as_tabelas_cuja_ausencia_e_legitima(
    real_tables: dict,
) -> None:
    """Uma tabela vazia so e aceitavel quando o dado nao existe no caso.

    A unica ausencia legitima e a do tempo de viagem, e so quando o caso nao tem
    nenhuma UHE com tempo de viagem. Qualquer outra tabela vazia e defeito de
    decodificacao, e por isso a comparacao e por igualdade de conjuntos - nao por
    inclusao, que deixaria passar tabela vazia inesperada.
    """
    tables = real_tables["tables"]
    esperadas_vazias = (
        set(TRAVEL_TIME_TABLES) if _travel_time_plant_count(real_tables) == 0 else set()
    )

    vazias = {name for name, table in tables.items() if table.empty}

    assert vazias == esperadas_vazias
    # Vazia nao pode significar "sem colunas": o CSV tem de sair com cabecalho.
    for name in sorted(vazias):
        assert list(tables[name].columns), f"{name} saiu sem colunas"


def test_tabelas_de_tempo_viagem_vazias_conservam_colunas_e_dtypes(
    real_tables: dict,
) -> None:
    """Sem tempo de viagem, as tabelas saem vazias mas com o contrato preservado."""
    if _travel_time_plant_count(real_tables) > 0:
        pytest.skip("o deck presente tem UHE com tempo de viagem")
    tables = real_tables["tables"]

    for name, dtypes in (
        ("mapcut_tempo_viagem", TRAVEL_TIME_DTYPES),
        ("mapcut_tempo_viagem_lags", TRAVEL_TIME_LAGS_DTYPES),
        ("mapcut_tempo_viagem_idecomp_bruto", TRAVEL_TIME_DTYPES),
    ):
        table = tables[name]
        assert list(table.columns) == list(dtypes), name
        assert [str(dtype) for dtype in table.dtypes] == list(dtypes.values()), name


def test_topologia_de_jusante_e_um_indice_posicional(real_tables: dict) -> None:
    """O registro 4 guarda a posicao na lista de UHEs, nao o codigo da usina."""
    plants = real_tables["tables"]["mapcut_usinas_hidraulicas"]
    plant_count = len(plants)

    assert plants["indice_usina_jusante"].between(0, plant_count).all()
    # O indice 0 significa "sem jusante"; nesses casos o codigo tambem e 0.
    sem_jusante = plants["indice_usina_jusante"] == 0
    assert (plants.loc[sem_jusante, "codigo_usina_jusante"] == 0).all()
    # Nos demais, o codigo tem de ser um codigo de usina existente no caso.
    codigos = set(plants["codigo_usina"])
    assert set(plants.loc[~sem_jusante, "codigo_usina_jusante"]) <= codigos


def test_cascata_conhecida_do_rio_grande(real_tables: dict) -> None:
    """Camargos (1) -> Itutinga (2) -> Funil-Grande (4), topologia conhecida do SIN."""
    plants = real_tables["tables"]["mapcut_usinas_hidraulicas"].set_index(
        "codigo_usina"
    )

    assert plants.loc[1, "codigo_usina_jusante"] == 2
    assert plants.loc[2, "codigo_usina_jusante"] == 4


def test_tempo_viagem_corrigido_tem_uma_linha_por_usina_estagio_e_lag(
    real_tables: dict,
) -> None:
    """O idecomp devolve 84 linhas por nao reiniciar os acumuladores; o certo e 56.

    So se aplica a deck com tempo de viagem: sem nenhuma UHE com tempo de viagem
    nao ha excesso do idecomp para comparar, e a tabela corrigida esta vazia por
    construcao. Esse caso e coberto pelos testes de tabela vazia acima e, no deck
    com tempo de viagem, por `test_deck_com_tempo_viagem.py`.
    """
    mapcut = real_tables["mapcut"]
    travel_time = real_tables["tables"]["mapcut_tempo_viagem"]
    raw = real_tables["tables"]["mapcut_tempo_viagem_idecomp_bruto"]

    plant_count = _travel_time_plant_count(real_tables)
    if plant_count == 0:
        pytest.skip("o deck presente nao tem UHE com tempo de viagem")
    stage_count = int(mapcut.numero_estagios)
    lag_count = int(mapcut.maximo_lag_tempo_viagem) + 1  # os lags vao de 0 a max

    assert len(travel_time) == plant_count * stage_count * lag_count
    assert len(raw) > len(travel_time), "a saida bruta do idecomp deveria ter excesso"

    # Cada usina recebe exatamente a mesma quantidade de linhas...
    por_usina = travel_time.groupby("codigo_usina").size()
    assert por_usina.nunique() == 1
    assert len(por_usina) == plant_count
    # ...e cada par usina/estagio cobre os lags 0..max, sem repeticao.
    por_usina_estagio = travel_time.groupby(["codigo_usina", "estagio"])[
        "indice_lag"
    ].agg(list)
    assert all(lags == list(range(lag_count)) for lags in por_usina_estagio)


def test_tempo_viagem_bruto_do_idecomp_repete_a_primeira_usina(
    real_tables: dict,
) -> None:
    """Documenta o defeito: o bloco da 2a usina comeca com os dados da 1a."""
    corrigido = real_tables["tables"]["mapcut_tempo_viagem"]
    bruto = real_tables["tables"]["mapcut_tempo_viagem_idecomp_bruto"]
    codigos = sorted(corrigido["codigo_usina"].unique())
    if len(codigos) < 2:
        pytest.skip("o caso tem menos de duas UHEs com tempo de viagem")

    primeira = corrigido[corrigido["codigo_usina"] == codigos[0]]
    bloco_segunda = bruto[bruto["codigo_usina"] == codigos[1]]
    colunas = ["indice_lag", "coeficiente_amortecimento"]

    # A comparacao e por valor: o idecomp devolve tipos do numpy e a tabela
    # corrigida usa tipos nativos, o que nao e o ponto do teste.
    assert np.array_equal(
        bloco_segunda.head(len(primeira))[colunas].to_numpy(dtype=float),
        primeira[colunas].to_numpy(dtype=float),
    )


def test_gnl_corrigido_cobre_todos_os_estagios(real_tables: dict) -> None:
    """O passo errado do idecomp decodifica so o primeiro estagio."""
    mapcut = real_tables["mapcut"]
    gnl = real_tables["tables"]["mapcut_gnl"]
    bruto = real_tables["tables"]["mapcut_gnl_idecomp_bruto"]
    stage_count = int(mapcut.numero_estagios)

    assert sorted(gnl["estagio"].unique()) == list(range(1, stage_count + 1))
    assert len(gnl) == stage_count * int(gnl["numero_utes_gnl"].iloc[0])
    assert sorted(bruto["estagio"].unique()) == [1], (
        "a saida bruta do idecomp deveria cobrir apenas o primeiro estagio"
    )


def test_submercados_gnl_batem_com_a_tabela_de_gnl(real_tables: dict) -> None:
    gnl = real_tables["tables"]["mapcut_gnl"]
    submercados = real_tables["tables"]["mapcut_submercados_gnl"]

    assert set(submercados["codigo_submercado"]) == set(gnl["codigo_submercado"])


def test_bloco_de_valores_gnl_tem_o_tamanho_declarado_pelo_decomp(
    real_tables: dict,
) -> None:
    """O bloco e dimensionado como numero_estagios x numero_utes_gnl."""
    mapcut = real_tables["mapcut"]
    valores = real_tables["tables"]["mapcut_gnl_bloco_valores"]
    gnl = real_tables["tables"]["mapcut_gnl"]
    stage_count = int(mapcut.numero_estagios)
    plant_count = int(gnl["numero_utes_gnl"].iloc[0])

    assert len(valores) == stage_count * (stage_count * plant_count)
    assert valores.groupby("estagio").size().nunique() == 1


def test_arvore_de_cenarios_cobre_todos_os_nos(real_tables: dict) -> None:
    mapcut = real_tables["mapcut"]
    tree = real_tables["tables"]["mapcut_arvore_cenarios"]

    assert len(tree) == int(mapcut.numero_cenarios)
    assert tree["no"].tolist() == list(range(1, len(tree) + 1))
    # Todo no pai e um no valido, e o estagio vem anotado para todos.
    assert set(tree["no_pai"]) <= set(tree["no"])
    assert tree["estagio"].notna().all()


def test_metadados_expoem_os_escalares_com_descricao(real_tables: dict) -> None:
    metadata = real_tables["tables"]["mapcut_metadados"].set_index("parametro")

    assert list(metadata.columns) == ["valor", "descricao"]
    assert metadata.loc["numero_estagios", "valor"] == int(
        real_tables["mapcut"].numero_estagios
    )
    assert metadata["descricao"].str.len().gt(0).all()


def test_estagios_batem_com_a_geometria(real_tables: dict) -> None:
    stages = real_tables["tables"]["mapcut_estagios"]
    geometry = real_tables["geometry"]

    assert len(stages) == geometry.number_of_stages
    assert (stages["patamares_carga"] == geometry.number_of_load_blocks).all()
