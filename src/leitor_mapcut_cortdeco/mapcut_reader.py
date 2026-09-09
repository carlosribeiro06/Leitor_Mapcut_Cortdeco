"""Leitura do binario `mapcut` e montagem das tabelas exportaveis.

O `mapcut` e o cabecalho da funcao de custo futuro do DECOMP: ele descreve a
arvore de cenarios, as usinas, os estagios, os dados de tempo de viagem da agua,
os dados de GNL e - o que importa para a leitura dos cortes - a geometria dos
registros do `cortdeco`.

Correcoes aplicadas sobre o idecomp 1.14.2
------------------------------------------
Tres propriedades do `idecomp` decodificam o mapcut de forma incorreta. Nos tres
casos este modulo exporta **duas** tabelas: a corrigida (usada para analise) e a
saida literal da biblioteca, com sufixo `_idecomp_bruto`, para que se possa
auditar exatamente o que foi corrigido.

1. `codigos_uhes_jusante` - o registro 4 guarda `int32`, mas a biblioteca le
   `float32`; a correcao (em `geometry.downstream_plant_indices`) reinterpreta os
   bits. O valor e um indice posicional na lista de UHEs, nao um codigo.
2. `dados_tempo_viagem` - os acumuladores de lag e coeficiente nao sao
   reiniciados a cada usina, e o offset do bloco seguinte e recalculado em vez de
   acumulado. Resultado: linhas repetidas, atribuidas a usina errada e com
   rotulos de estagio deslocados. A correcao percorre o payload bruto com offset
   cumulativo.
3. `dados_gnl` - o passo entre os registros de cada estagio e calculado como
   `1 + 4*n + soma(patamares)`, mas o registro realmente ocupa
   `1 + 3*n` inteiros mais `numero_estagios * n` reais. Com o passo errado, so o
   primeiro estagio e decodificado. A correcao usa o passo verdadeiro.

Blocos ausentes e tabelas vazias
--------------------------------
Dois blocos do mapcut sao opcionais, e a ausencia de cada um e legitima - nao
arquivo corrompido:

* **tempo de viagem** (`numero_uhes_tempo_viagem` igual a zero) - a propriedade
  `codigos_uhes_tempo_viagem` levanta `KeyError` e `lag_tempo_viagem_por_uhe`
  devolve valores de outro campo;
* **GNL** (nenhuma UTE a GNL) - `dados_gnl` e `codigos_submercados_gnl` falham do
  mesmo jeito, porque o payload nao tem registro algum para percorrer.

Nos dois casos os acessores de `geometry` tratam a ausencia, e as tabelas
correspondentes deste modulo saem **vazias mas com as colunas e os dtypes
declarados**, para que o CSV continue autodescritivo em vez de virar um arquivo
sem cabecalho.

Quando a saida literal nao pode ser produzida
---------------------------------------------
A propriedade `dados_gnl` nem sempre e avaliavel: com poucos estagios o passo
errado dela sai do payload e ela levanta `IndexError`. Nesse caso nao existe
saida literal para auditar, e `mapcut_gnl_idecomp_bruto` sai vazia com um WARNING
explicando o motivo. A previsao fica em `_idecomp_gnl_frame_is_evaluable`, que e
o unico ponto do projeto a espelhar de proposito uma aritmetica defeituosa da
biblioteca.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from idecomp.decomp.mapcut import Mapcut

from .geometry import (
    CutGeometry,
    GeometryError,
    downstream_plant_indices,
    gnl_stage_blocks,
    raw_data_section,
    require_frame,
    require_int,
    require_list,
    travel_time_blocks,
    travel_time_lags,
    travel_time_plant_codes,
    travel_time_plant_count,
)

# Colunas e dtypes das tabelas de tempo de viagem. Ficam declarados aqui porque
# sao usados tanto no caminho normal quanto no caso sem tempo de viagem, em que a
# tabela sai vazia - e vazia com as mesmas colunas e os mesmos dtypes, para que
# quem consome os CSVs nao precise distinguir os dois casos.
TRAVEL_TIME_DTYPES: Mapping[str, str] = {
    "codigo_usina": "int64",
    "numero_horas": "int64",
    "estagio": "int64",
    "indice_lag": "int64",
    "coeficiente_amortecimento": "float64",
}
TRAVEL_TIME_LAGS_DTYPES: Mapping[str, str] = {
    "codigo_usina": "int64",
    "estagio": "int64",
    "lag_maximo": "int64",
}

# Idem para as tabelas de GNL, que ficam vazias em caso sem UTE a GNL.
GNL_DTYPES: Mapping[str, str] = {
    "estagio": "int64",
    "numero_utes_gnl": "int64",
    "codigo_submercado": "int64",
    "indice_lag": "int64",
    "numero_patamares": "int64",
}
GNL_VALUE_DTYPES: Mapping[str, str] = {
    "estagio": "int64",
    "posicao_no_bloco": "int64",
    "valor": "float64",
}
SUBMARKET_DTYPES: Mapping[str, str] = {"codigo_submercado": "int64"}

# Descricao de cada escalar do mapcut, para que o CSV de metadados seja
# autoexplicativo mesmo fora do contexto deste projeto.
METADATA_FIELDS: tuple[tuple[str, str], ...] = (
    ("data_inicio", "Data de inicio do estudo"),
    ("numero_iteracoes", "Numero de iteracoes da politica"),
    ("numero_cortes", "Numero total de cortes de Benders, somando todos os nos"),
    ("numero_estagios", "Numero de estagios do caso"),
    ("numero_semanas", "Numero de estagios semanais"),
    ("numero_cenarios", "Numero de nos da arvore de cenarios"),
    ("numero_submercados", "Numero de submercados"),
    ("numero_uhes", "Numero de usinas hidraulicas"),
    ("numero_uhes_tempo_viagem", "Numero de UHEs com tempo de viagem da agua"),
    ("maximo_lag_tempo_viagem", "Lag maximo de tempo de viagem, em estagios"),
    ("tamanho_corte", "Tamanho de cada registro do cortdeco, em bytes"),
)


def read_mapcut(path: Path, logger: logging.Logger) -> Mapcut:
    """Le o binario mapcut.

    Args:
        path: caminho do arquivo mapcut.
        logger: logger da aplicacao.

    Returns:
        O objeto `Mapcut` do idecomp.
    """
    # `read` e herdado de `SectionFile` e anotado com o tipo da classe base.
    mapcut = cast("Mapcut", Mapcut.read(str(path)))
    logger.info(
        "mapcut lido: %s estagios, %s cenarios, %s UHEs, %s submercados, %s iteracoes",
        mapcut.numero_estagios,
        mapcut.numero_cenarios,
        mapcut.numero_uhes,
        mapcut.numero_submercados,
        mapcut.numero_iteracoes,
    )
    return mapcut


def _empty_table(dtypes: Mapping[str, str]) -> pd.DataFrame:
    """Monta uma tabela sem linhas, mas com as colunas e os dtypes declarados.

    Uma tabela vazia construida com `pd.DataFrame([])` sai tambem *sem colunas*,
    e o CSV correspondente fica sem cabecalho - a mesma patologia do idecomp que
    este modulo corrige. Por isso todo caminho que pode produzir tabela vazia
    passa por aqui.
    """
    return pd.DataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in dtypes.items()}
    )


def _metadata_table(mapcut: Mapcut) -> pd.DataFrame:
    """Monta a tabela de escalares do mapcut (parametro, valor, descricao)."""
    rows = []
    for field, description in METADATA_FIELDS:
        value = getattr(mapcut, field)
        rows.append(
            {
                "parametro": field,
                "valor": value.isoformat() if hasattr(value, "isoformat") else value,
                "descricao": description,
            }
        )
    return pd.DataFrame(rows)


def _hydro_plants_table(mapcut: Mapcut, logger: logging.Logger) -> pd.DataFrame:
    """Monta a tabela de UHEs com a topologia de jusante corrigida.

    A coluna `posicao` e a ordem em que a usina aparece no mapcut, que e a mesma
    ordem dos coeficientes de volume armazenado dentro de cada corte - por isso
    ela e util para casar esta tabela com os coeficientes do cortdeco.
    """
    codes = np.asarray(
        [int(c) for c in require_list(mapcut.codigos_uhes, "codigos_uhes")],
        dtype=np.int64,
    )
    downstream_index = downstream_plant_indices(mapcut)
    if downstream_index.shape != codes.shape:
        raise GeometryError(
            f"a topologia de jusante tem {downstream_index.size} valores, mas o "
            f"mapcut declara {codes.size} UHEs"
        )

    out_of_range = downstream_index[
        (downstream_index < 0) | (downstream_index > codes.size)
    ]
    if out_of_range.size:
        raise GeometryError(
            "indices de usina a jusante fora da faixa valida "
            f"[0, {codes.size}]: {sorted(set(out_of_range.tolist()))[:10]}. "
            "A reinterpretacao dos bits do registro 4 nao se sustenta."
        )

    # Indice 0 significa que a usina nao tem jusante; nesse caso o codigo fica 0.
    downstream_code = np.where(downstream_index > 0, codes[downstream_index - 1], 0)
    logger.info(
        "topologia de jusante corrigida: %d UHEs, das quais %d sem usina a jusante",
        codes.size,
        int((downstream_index == 0).sum()),
    )
    return pd.DataFrame(
        {
            "posicao": np.arange(1, codes.size + 1, dtype=np.int64),
            "codigo_usina": codes,
            "indice_usina_jusante": downstream_index,
            "codigo_usina_jusante": downstream_code,
        }
    )


def _hydro_plants_raw_table(mapcut: Mapcut) -> pd.DataFrame:
    """Reproduz a topologia de jusante como o idecomp a devolve (float32 espurio)."""
    codes = [int(c) for c in require_list(mapcut.codigos_uhes, "codigos_uhes")]
    return pd.DataFrame(
        {
            "posicao": range(1, len(codes) + 1),
            "codigo_usina": codes,
            "codigos_uhes_jusante_idecomp": [
                float(v)
                for v in require_list(
                    mapcut.codigos_uhes_jusante, "codigos_uhes_jusante"
                )
            ],
        }
    )


def _scenario_tree_table(mapcut: Mapcut) -> pd.DataFrame:
    """Monta a arvore de cenarios (no -> no pai), anotada com o estagio de cada no."""
    parents = [
        int(p) for p in require_list(mapcut.indice_no_arvore, "indice_no_arvore")
    ]
    tree = pd.DataFrame(
        {
            "no": range(1, len(parents) + 1),
            "no_pai": parents,
        }
    )
    stages = require_frame(mapcut.registro_ultimo_corte_no, "registro_ultimo_corte_no")[
        ["no", "estagio"]
    ]
    return tree.merge(stages, on="no", how="left")[["no", "estagio", "no_pai"]]


def _stages_table(mapcut: Mapcut) -> pd.DataFrame:
    """Monta a tabela de estagios (primeiro no e patamares de carga)."""
    first_node = [
        int(v)
        for v in require_list(
            mapcut.indice_primeiro_no_estagio, "indice_primeiro_no_estagio"
        )
    ]
    load_blocks = [
        int(v)
        for v in require_list(mapcut.patamares_por_estagio, "patamares_por_estagio")
    ]
    return pd.DataFrame(
        {
            "estagio": range(1, len(first_node) + 1),
            "indice_primeiro_no": first_node,
            "patamares_carga": load_blocks,
        }
    )


def _travel_time_lags_table(mapcut: Mapcut, logger: logging.Logger) -> pd.DataFrame:
    """Monta a tabela de lag maximo de tempo de viagem por usina e estagio.

    O vetor bruto tem `numero_uhes_tempo_viagem * numero_estagios` valores,
    organizados por usina e, dentro dela, por estagio. O `idecomp` usa duas
    convencoes diferentes de indexacao para esse mesmo vetor (a propriedade
    fatia por usina; a rotina de leitura indexa por `usina * estagio`), o que so
    e inofensivo quando todos os lags sao iguais - o caso mais comum. Quando nao
    forem, a atribuicao aqui segue a convencao da propriedade e um WARNING avisa
    que ela nao pode ser conferida por outra via.

    Returns:
        A tabela vazia, com as colunas declaradas, quando o caso nao tem UHE com
        tempo de viagem.
    """
    plant_codes = travel_time_plant_codes(mapcut)
    if not plant_codes:
        return _empty_table(TRAVEL_TIME_LAGS_DTYPES)

    stage_count = require_int(mapcut.numero_estagios, "numero_estagios")
    lags = travel_time_lags(mapcut)
    if len(set(lags)) > 1:
        logger.warning(
            "os lags de tempo de viagem nao sao todos iguais (%s). A atribuicao "
            "usina/estagio segue a convencao da propriedade do idecomp "
            "(fatiamento por usina) e nao pode ser conferida por outra via.",
            sorted(set(lags)),
        )
    rows = []
    for plant_position, plant_code in enumerate(plant_codes):
        plant_lags = lags[
            plant_position * stage_count : (plant_position + 1) * stage_count
        ]
        for stage, lag in enumerate(plant_lags, start=1):
            rows.append(
                {"codigo_usina": plant_code, "estagio": stage, "lag_maximo": lag}
            )
    return pd.DataFrame(rows).astype(dict(TRAVEL_TIME_LAGS_DTYPES))


def _travel_time_table(mapcut: Mapcut, logger: logging.Logger) -> pd.DataFrame:
    """Reconstroi os dados de tempo de viagem a partir do payload bruto.

    Layout do payload (registros 7 e 8 do mapcut), por usina com tempo de viagem:

        [codigo_usina, numero_horas, (indice_lag, coeficiente) x N]

    onde `N = soma sobre os estagios de (lag_maximo + 1)`, porque os lags vao de
    0 a `lag_maximo` inclusive. A localizacao dos blocos e o acumulo do offset -
    exatamente onde o `idecomp` erra - ficam em `geometry.travel_time_blocks`,
    que tambem confere que os blocos consomem o payload inteiro.

    Returns:
        A tabela vazia, com as colunas declaradas, quando o caso nao tem UHE com
        tempo de viagem.
    """
    blocks = travel_time_blocks(mapcut)
    if not blocks:
        logger.info(
            "sem dados de tempo de viagem a reconstruir: o caso nao tem UHE com "
            "tempo de viagem da agua"
        )
        return _empty_table(TRAVEL_TIME_DTYPES)

    raw = raw_data_section(mapcut)["dados_tempo_viagem"]
    stage_count = require_int(mapcut.numero_estagios, "numero_estagios")

    rows = []
    for block in blocks:
        cursor = block.pair_start
        for stage, lag_maximum in enumerate(block.stage_lags, start=1):
            for _ in range(lag_maximum + 1):
                rows.append(
                    {
                        "codigo_usina": block.plant_code,
                        "numero_horas": block.travel_hours,
                        "estagio": stage,
                        "indice_lag": int(raw[cursor]),
                        "coeficiente_amortecimento": float(raw[cursor + 1]),
                    }
                )
                cursor += 2

    logger.info(
        "dados de tempo de viagem reconstruidos: %d linhas (%d UHEs x %d estagios); "
        "o idecomp devolveria %d linhas",
        len(rows),
        len(blocks),
        stage_count,
        len(require_frame(mapcut.dados_tempo_viagem, "dados_tempo_viagem")),
    )
    return pd.DataFrame(rows).astype(dict(TRAVEL_TIME_DTYPES))


def _idecomp_gnl_frame_is_evaluable(mapcut: Mapcut) -> bool:
    """Preve se `Mapcut.dados_gnl` consegue ser avaliada sem levantar excecao.

    ESTE E O UNICO PONTO DO PROJETO QUE ESPELHA DE PROPOSITO UMA ARITMETICA
    DEFEITUOSA DO IDECOMP (versao 1.14.2). Ele reproduz o passo errado
    `1 + 4*n + soma(numero_patamares)` apenas para saber se a varredura da
    propriedade permanece dentro do payload:

    * se permanecer, a propriedade devolve uma tabela - decodificada de forma
      incorreta, o que e justamente o que a tabela de auditoria
      `mapcut_gnl_idecomp_bruto` existe para registrar;
    * se sair do payload, a propriedade levanta `IndexError` e nao ha saida
      literal alguma para auditar.

    A alternativa seria envolver a propriedade em `except IndexError`, o que
    mascararia falhas de outra natureza. Prever e conferir e mais estreito.

    Se o `idecomp` for atualizado, esta funcao tem de ser revista junto: ela
    descreve o comportamento de uma versao especifica, nao um contrato.
    """
    raw = raw_data_section(mapcut)["dados_gnl"]
    stage_count = require_int(mapcut.numero_estagios, "numero_estagios")

    offset = 0
    for _ in range(stage_count):
        if offset >= len(raw) or offset < 0:
            return False
        plant_count = int(raw[offset])
        load_blocks = raw[offset + 2 * plant_count + 1 : offset + 3 * plant_count + 1]
        offset += int(1 + 4 * plant_count + sum(load_blocks))
    return True


def _gnl_tables(
    mapcut: Mapcut, logger: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstroi os dados de GNL a partir do payload bruto.

    Layout do payload (registro 9 do mapcut), um registro por estagio:

        [numero_utes_gnl,
         codigo_submercado x n, indice_lag x n, numero_patamares x n,   (int32)
         valores x (numero_estagios * n)]                              (float64)

    O passo entre estagios e portanto `1 + 3*n + numero_estagios*n`. O `idecomp`
    usa `1 + 4*n + soma(numero_patamares)`, que so coincide por acidente em casos
    pequenos; com o passo errado, apenas o primeiro estagio e decodificado.

    O bloco final de reais e dimensionado pelo DECOMP como
    `numero_estagios * numero_utes_gnl`, mas em geral apenas as primeiras
    `numero_patamares * numero_utes_gnl` posicoes vem preenchidas. Como o
    significado exato de cada posicao nao esta documentado, ele e exportado como
    esta - indexado pela posicao dentro do bloco - em vez de receber rotulos
    inventados.

    Returns:
        A tabela de dados de GNL por estagio/submercado e a tabela do bloco de
        valores reais. Ambas vazias, com as colunas declaradas, quando o caso nao
        tem UTE a GNL.
    """
    blocks = gnl_stage_blocks(mapcut)
    raw = raw_data_section(mapcut)["dados_gnl"]
    stage_count = require_int(mapcut.numero_estagios, "numero_estagios")

    rows: list[dict[str, int]] = []
    value_rows: list[dict[str, float | int]] = []
    for block in blocks:
        for submarket, lag, load_blocks in zip(
            block.submarket_codes,
            block.lag_indices,
            block.load_block_counts,
            strict=True,
        ):
            rows.append(
                {
                    "estagio": block.stage,
                    "numero_utes_gnl": block.plant_count,
                    "codigo_submercado": submarket,
                    "indice_lag": lag,
                    "numero_patamares": load_blocks,
                }
            )
        value_block = raw[block.value_start : block.value_stop]
        for position, value in enumerate(value_block, start=1):
            value_rows.append(
                {
                    "estagio": block.stage,
                    "posicao_no_bloco": position,
                    "valor": float(value),
                }
            )

    # A ausencia de GNL pode chegar de duas formas: payload inteiramente vazio
    # (nenhum registro 9) ou um registro por estagio declarando zero UTEs. As
    # duas terminam sem linha alguma, e e por isso que a decisao olha as linhas e
    # nao os blocos.
    if not rows:
        logger.info("sem dados de GNL a reconstruir: o caso nao tem UTE a GNL")
        return _empty_table(GNL_DTYPES), _empty_table(GNL_VALUE_DTYPES)

    logger.info(
        "dados de GNL reconstruidos: %d linhas cobrindo %d estagios; "
        "o idecomp devolveria %s",
        len(rows),
        stage_count,
        f"{len(require_frame(mapcut.dados_gnl, 'dados_gnl'))} linhas"
        if _idecomp_gnl_frame_is_evaluable(mapcut)
        else "IndexError (a varredura dela sai do payload)",
    )
    return (
        pd.DataFrame(rows).astype(dict(GNL_DTYPES)),
        pd.DataFrame(value_rows).astype(dict(GNL_VALUE_DTYPES)),
    )


def build_mapcut_tables(
    mapcut: Mapcut,
    geometry: CutGeometry,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Monta todas as tabelas exportaveis do mapcut.

    Args:
        mapcut: objeto `Mapcut` ja lido.
        geometry: geometria conferida, usada para as tabelas derivadas do cortdeco.
        logger: logger da aplicacao.

    Returns:
        Dicionario `nome_do_csv -> DataFrame`. Os nomes terminados em
        `_idecomp_bruto` sao a saida literal da biblioteca, mantida para auditoria.
    """
    gnl_table, gnl_values_table = _gnl_tables(mapcut, logger)
    # A tabela bruta do idecomp so pode ser pedida quando o caso tem tempo de
    # viagem; sem isso a propriedade levanta KeyError. O caminho explicito para o
    # caso legitimamente vazio evita enfraquecer o `require_frame`.
    if travel_time_plant_count(mapcut) == 0:
        raw_travel_time = _empty_table(TRAVEL_TIME_DTYPES)
    else:
        raw_travel_time = require_frame(mapcut.dados_tempo_viagem, "dados_tempo_viagem")

    # A saida literal do idecomp para GNL nem sempre existe: quando a varredura
    # da propriedade sai do payload, ela levanta IndexError e nao ha nada para
    # auditar. Nesse caso a tabela sai vazia e o log diz por que.
    if not gnl_table.empty and _idecomp_gnl_frame_is_evaluable(mapcut):
        raw_gnl = require_frame(mapcut.dados_gnl, "dados_gnl")
    else:
        raw_gnl = _empty_table(GNL_DTYPES)
        if not gnl_table.empty:
            logger.warning(
                "a tabela de auditoria mapcut_gnl_idecomp_bruto saiu vazia: a "
                "propriedade dados_gnl do idecomp levanta IndexError neste caso, "
                "porque o passo errado dela (1 + 4n + soma(patamares)) sai do "
                "payload quando o numero de estagios e pequeno. A reconstrucao "
                "correta esta em mapcut_gnl; nao ha saida literal para comparar."
            )
    return {
        "mapcut_metadados": _metadata_table(mapcut),
        "mapcut_usinas_hidraulicas": _hydro_plants_table(mapcut, logger),
        "mapcut_arvore_cenarios": _scenario_tree_table(mapcut),
        "mapcut_estagios": _stages_table(mapcut),
        "mapcut_ultimo_corte_por_no": geometry.last_cut_record_per_node,
        "mapcut_tempo_viagem_lags": _travel_time_lags_table(mapcut, logger),
        "mapcut_tempo_viagem": _travel_time_table(mapcut, logger),
        "mapcut_gnl": gnl_table,
        "mapcut_gnl_bloco_valores": gnl_values_table,
        "mapcut_submercados_gnl": pd.DataFrame(
            {"codigo_submercado": list(geometry.gnl_submarket_codes)}
        ).astype(dict(SUBMARKET_DTYPES))
        if geometry.gnl_submarket_codes
        else _empty_table(SUBMARKET_DTYPES),
        "mapcut_custos": require_frame(mapcut.dados_custos, "dados_custos"),
        "mapcut_usinas_jusante_idecomp_bruto": _hydro_plants_raw_table(mapcut),
        "mapcut_tempo_viagem_idecomp_bruto": raw_travel_time,
        "mapcut_gnl_idecomp_bruto": raw_gnl,
    }
