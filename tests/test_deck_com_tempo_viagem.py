"""Testes do caminho de tempo de viagem, contra o deck que o exercita.

Por que este modulo existe
--------------------------
O deck presente na raiz do projeto muda a cada caso, e o caso atual nao tem
nenhuma UHE com tempo de viagem (`numero_uhes_tempo_viagem = 0`). Sem este
modulo, toda a trilha de tempo de viagem - que e onde estao as correcoes mais
delicadas sobre o `idecomp` - ficaria sem cobertura, e uma regressao nela so
apareceria quando chegasse um caso com tempo de viagem, possivelmente em estudo
oficial.

O deck com tempo de viagem usado no desenvolvimento original (2 UHEs, lag maximo
3) nao esta mais no diretorio de trabalho, porque os binarios deixaram de ser
versionados, mas continua recuperavel do historico do git. Este modulo o extrai
para um diretorio temporario e roda a leitura completa contra ele.

Quando o objeto do git nao esta disponivel - uma copia do repositorio sem
historico, por exemplo -, os testes sao ignorados.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import pytest

from leitor_mapcut_cortdeco.config import CortdecoSettings
from leitor_mapcut_cortdeco.geometry import build_cut_geometry, travel_time_blocks
from leitor_mapcut_cortdeco.mapcut_reader import build_mapcut_tables, read_mapcut

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Commit que ainda versionava os binarios de entrada (o primeiro do repositorio).
DECK_COMMIT = "353ebf8"

# Dimensoes conhecidas desse deck, conferidas nos bytes crus do mapcut:
# cabecalho do registro 6 = [1, 7, 6, 2, 3] -> 7 estagios, 2 UHEs com tempo de
# viagem, lag maximo 3.
EXPECTED_PLANT_COUNT = 2
EXPECTED_MAX_LAG = 3
EXPECTED_STAGES = 7

# Uma linha por usina, estagio e lag, com os lags indo de 0 a lag_maximo:
# 2 x 7 x 4 = 56. O idecomp devolve 84 linhas por nao reiniciar os acumuladores.
EXPECTED_TRAVEL_TIME_ROWS = 56
EXPECTED_IDECOMP_ROWS = 84

DEFAULT_CORTDECO_SETTINGS = CortdecoSettings(
    cuts_per_node_source="numero_iteracoes", cuts_per_node_override=None
)


def _extract(revision: str, name: str, destination: Path) -> Path:
    """Extrai um binario do historico do git para um diretorio temporario."""
    path = destination / name
    try:
        result = subprocess.run(
            ["git", "show", f"{revision}:{name}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
    except OSError as err:  # git ausente no PATH
        pytest.skip(f"git indisponivel para recuperar {name}: {err}")
    if result.returncode != 0:
        pytest.skip(
            f"{revision}:{name} nao esta no historico "
            f"({result.stderr.decode(errors='replace').strip()})"
        )
    path.write_bytes(result.stdout)
    return path


@pytest.fixture(scope="module")
def deck_com_tempo_viagem(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Le o deck com tempo de viagem uma unica vez e monta todas as tabelas."""
    workspace = tmp_path_factory.mktemp("deck_com_tempo_viagem")
    mapcut_path = _extract(DECK_COMMIT, "mapcut.rv0", workspace)
    cortdeco_path = _extract(DECK_COMMIT, "cortdeco.rv0", workspace)

    logger = logging.getLogger("teste_deck_com_tempo_viagem")
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


def test_o_deck_recuperado_tem_tempo_de_viagem(
    deck_com_tempo_viagem: dict[str, Any],
) -> None:
    """Guarda contra extrair o deck errado: sem isso os demais testes nao valem."""
    mapcut = deck_com_tempo_viagem["mapcut"]

    assert int(mapcut.numero_uhes_tempo_viagem) == EXPECTED_PLANT_COUNT
    assert int(mapcut.maximo_lag_tempo_viagem) == EXPECTED_MAX_LAG
    assert int(mapcut.numero_estagios) == EXPECTED_STAGES


def test_codigos_vem_do_payload_e_sao_uhes_do_caso(
    deck_com_tempo_viagem: dict[str, Any],
) -> None:
    """Os codigos decodificados do payload tem de ser UHEs existentes no caso."""
    mapcut = deck_com_tempo_viagem["mapcut"]
    geometry = deck_com_tempo_viagem["geometry"]
    codigos = geometry.travel_time_plant_codes

    assert len(codigos) == EXPECTED_PLANT_COUNT
    assert len(set(codigos)) == EXPECTED_PLANT_COUNT
    assert set(codigos) <= {int(c) for c in mapcut.codigos_uhes}


def test_blocos_consomem_o_payload_inteiro(
    deck_com_tempo_viagem: dict[str, Any],
) -> None:
    """A conferencia que ancora o layout assumido dos registros 7 e 8."""
    mapcut = deck_com_tempo_viagem["mapcut"]
    blocks = travel_time_blocks(mapcut)

    assert len(blocks) == EXPECTED_PLANT_COUNT
    for block in blocks:
        assert len(block.stage_lags) == EXPECTED_STAGES
        assert max(block.stage_lags) <= EXPECTED_MAX_LAG
    # Os blocos sao consecutivos e sem lacuna entre um e o proximo.
    assert blocks[0].pair_start == 2
    for anterior, seguinte in zip(blocks, blocks[1:], strict=False):
        assert seguinte.pair_start == anterior.pair_stop + 2


def test_largura_do_bloco_de_coeficientes_no_corte(
    deck_com_tempo_viagem: dict[str, Any],
) -> None:
    """Dentro do corte ha um coeficiente por UHE e por lag de 1 a lag_maximo."""
    geometry = deck_com_tempo_viagem["geometry"]

    assert geometry.max_travel_time_lag == EXPECTED_MAX_LAG
    assert geometry.travel_time_coefficient_count == (
        EXPECTED_PLANT_COUNT * EXPECTED_MAX_LAG
    )


def test_tempo_viagem_corrigido_tem_56_linhas_contra_84_do_idecomp(
    deck_com_tempo_viagem: dict[str, Any],
) -> None:
    """A correcao do defeito de acumulador, medida no deck que o expoe."""
    tables = deck_com_tempo_viagem["tables"]
    corrigido = tables["mapcut_tempo_viagem"]
    bruto = tables["mapcut_tempo_viagem_idecomp_bruto"]

    assert len(corrigido) == EXPECTED_TRAVEL_TIME_ROWS
    assert len(bruto) == EXPECTED_IDECOMP_ROWS

    # Cada par usina/estagio cobre os lags 0..lag_maximo, sem repeticao.
    por_usina_estagio = corrigido.groupby(["codigo_usina", "estagio"])[
        "indice_lag"
    ].agg(list)
    assert len(por_usina_estagio) == EXPECTED_PLANT_COUNT * EXPECTED_STAGES
    assert all(lags == list(range(EXPECTED_MAX_LAG + 1)) for lags in por_usina_estagio)


def test_tabela_de_lags_cobre_cada_par_usina_estagio(
    deck_com_tempo_viagem: dict[str, Any],
) -> None:
    lags = deck_com_tempo_viagem["tables"]["mapcut_tempo_viagem_lags"]

    assert len(lags) == EXPECTED_PLANT_COUNT * EXPECTED_STAGES
    assert sorted(lags["estagio"].unique()) == list(range(1, EXPECTED_STAGES + 1))
    assert lags["lag_maximo"].between(0, EXPECTED_MAX_LAG).all()


def test_nenhuma_tabela_do_deck_com_tempo_viagem_esta_vazia(
    deck_com_tempo_viagem: dict[str, Any],
) -> None:
    """Com tempo de viagem, nenhuma ausencia e legitima."""
    vazias = [
        name for name, table in deck_com_tempo_viagem["tables"].items() if table.empty
    ]

    assert vazias == []
