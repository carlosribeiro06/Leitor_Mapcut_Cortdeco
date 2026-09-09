"""Leitura do binario `cortdeco` e montagem das tabelas exportaveis.

Cada registro do `cortdeco` guarda um corte de Benders da funcao de custo futuro:
um termo constante (`rhs`) seguido dos coeficientes (multiplicadores duais) em
tres blocos, sempre nesta ordem dentro do registro:

1. **volume armazenado** - um coeficiente por UHE, na ordem de `codigos_uhes`;
2. **defluencia passada por tempo de viagem** - um coeficiente por UHE com tempo
   de viagem e por lag;
3. **geracao GNL antecipada** - um coeficiente por submercado, estagio e patamar
   de carga.

Formatos exportados
-------------------
* **wide** (`cortdeco_cortes`) - uma linha por corte e uma coluna por
  coeficiente. E a forma mais proxima do registro binario, boa para auditoria.
* **tidy** - uma tabela longa por bloco de coeficientes, com as dimensoes
  (usina, lag, submercado, patamar) em colunas proprias. E a forma pratica para
  analise.

Correcao aplicada sobre o idecomp 1.14.2
----------------------------------------
Os nomes das colunas do formato wide seguem o padrao
`pi_gnl_sbm{submercado}_pat{patamar}_lag{estagio}` - note que o token `lag`
enumera **estagios** (1 a `numero_estagios`) e o token `pat` enumera patamares.
A propriedade `Cortdeco.coeficientes_geracao_gnl` extrai a coluna `lag` do token
`pat`, de modo que o rotulo sai errado e a dimensao de estagio desaparece: os 42
coeficientes de cada corte colapsam em tres valores ambiguos. A tabela tidy de
GNL deste modulo e derivada diretamente dos nomes das colunas do formato wide,
separando as duas dimensoes; a saida literal da biblioteca e mantida em
`cortdeco_coef_geracao_gnl_idecomp_bruto` para auditoria.

Casos sem UHE com tempo de viagem
---------------------------------
Quando o caso nao tem UHE com tempo de viagem, o bloco 2 tem largura zero: o
corte vai direto do volume armazenado para a geracao GNL. A propriedade
`Cortdeco.coeficientes_defluencia_tempo_viagem` nao trata esse caso - ela faz
`.str.split("uhe", expand=True)[1]` sobre um `melt` sem linhas, e o
`expand=True` devolve um DataFrame sem colunas, de modo que o `[1]` levanta
`KeyError`. A tabela tidy correspondente sai vazia, com as colunas declaradas.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from idecomp.decomp.cortdeco import Cortdeco

from .geometry import CutGeometry, GeometryError, require_frame

# Colunas que identificam o corte; comuns a todas as tabelas tidy.
CUT_KEY_COLUMNS: tuple[str, ...] = ("indice_corte", "no", "estagio")

# Colunas e dtypes da tabela tidy de defluencia por tempo de viagem, declarados
# para que ela saia com o mesmo cabecalho tenha o caso tempo de viagem ou nao.
TRAVEL_TIME_COEFFICIENT_DTYPES: Mapping[str, str] = {
    "indice_corte": "int64",
    "no": "int64",
    "estagio": "int64",
    "codigo_usina": "int64",
    "lag": "int64",
    "valor": "float64",
}

# Idem para a saida literal do idecomp do bloco de GNL, que tem a mesma forma de
# falha quando o caso nao tem UTE a GNL.
GNL_COEFFICIENT_DTYPES: Mapping[str, str] = {
    "indice_corte": "int64",
    "no": "int64",
    "estagio": "int64",
    "codigo_submercado": "int64",
    "lag": "int64",
    "valor": "float64",
}

# `pi_gnl_sbm{submercado}_pat{patamar}_lag{estagio}`
GNL_COLUMN_PATTERN = re.compile(r"^pi_gnl_sbm(\d+)_pat(\d+)_lag(\d+)$")


def read_cortdeco(
    path: Path,
    geometry: CutGeometry,
    logger: logging.Logger,
) -> Cortdeco:
    """Le o binario cortdeco usando a geometria derivada do mapcut.

    Args:
        path: caminho do arquivo cortdeco.
        geometry: geometria ja conferida contra o arquivo.
        logger: logger da aplicacao.

    Returns:
        O objeto `Cortdeco` do idecomp.
    """
    cortdeco = Cortdeco.read(
        str(path),
        tamanho_registro=geometry.record_size_bytes,
        registro_ultimo_corte_no=geometry.last_cut_record_per_node,
        numero_total_cortes=geometry.cuts_per_node,
        numero_patamares_carga=geometry.number_of_load_blocks,
        numero_estagios=geometry.number_of_stages,
        codigos_uhes=list(geometry.hydro_plant_codes),
        codigos_uhes_tempo_viagem=list(geometry.travel_time_plant_codes),
        codigos_submercados=list(geometry.gnl_submarket_codes),
        lag_maximo_tempo_viagem=geometry.max_travel_time_lag,
    )
    cuts = cortdeco.cortes
    if cuts is None or cuts.empty:
        raise GeometryError(
            "a leitura do cortdeco nao devolveu nenhum corte. Confira se o "
            "mapcut e o cortdeco pertencem ao mesmo caso."
        )
    logger.info(
        "cortdeco lido: %d cortes x %d colunas (%d coeficientes + %d chaves)",
        len(cuts),
        cuts.shape[1],
        geometry.coefficient_count,
        len(CUT_KEY_COLUMNS),
    )
    return cortdeco


def validate_cuts(
    cuts: pd.DataFrame,
    geometry: CutGeometry,
    logger: logging.Logger,
) -> None:
    """Confere a tabela de cortes contra a geometria esperada.

    As duas conferencias visam o modo tipico de falha da leitura: quando o numero
    de cortes por no e subestimado, faltam linhas; quando e superestimado, sobram
    linhas inteiramente zeradas (a lista encadeada termina antes de preencher a
    tabela). Nenhuma das duas gera excecao - os CSVs continuam sendo gerados -,
    mas ambas produzem WARNING para nao passarem em branco numa execucao oficial.
    """
    expected_columns = len(CUT_KEY_COLUMNS) + geometry.coefficient_count
    if cuts.shape[1] != expected_columns:
        logger.warning(
            "a tabela de cortes tem %d colunas, mas a geometria preve %d",
            cuts.shape[1],
            expected_columns,
        )
    if len(cuts) != geometry.total_cut_records:
        logger.warning(
            "foram lidos %d cortes, mas o arquivo tem %d registros. Cortes "
            "podem ter ficado de fora, ou linhas nulas podem ter sido incluidas.",
            len(cuts),
            geometry.total_cut_records,
        )

    coefficient_columns = [c for c in cuts.columns if c not in CUT_KEY_COLUMNS]
    all_zero = int((cuts[coefficient_columns] == 0).all(axis=1).sum())
    if all_zero:
        logger.warning(
            "%d cortes tem todos os %d coeficientes iguais a zero. Isso costuma "
            "indicar excesso de linhas alocadas (cortes por no superestimado); "
            "revise cortdeco.cuts_per_node_source antes de usar os CSVs.",
            all_zero,
            len(coefficient_columns),
        )
    else:
        logger.info(
            "integridade dos cortes: nenhuma das %d linhas esta inteiramente "
            "zerada; a contagem de cortes por no (%d) e consistente",
            len(cuts),
            geometry.cuts_per_node,
        )

    per_stage = cuts.groupby("estagio").size().to_dict()
    logger.info("cortes por estagio: %s", per_stage)


def _rhs_table(cuts: pd.DataFrame) -> pd.DataFrame:
    """Extrai o termo constante de cada corte."""
    return cuts[[*CUT_KEY_COLUMNS, "rhs"]].copy()


def _gnl_tidy_table(cuts: pd.DataFrame) -> pd.DataFrame:
    """Monta a tabela longa de coeficientes de GNL, separando estagio e patamar.

    Os nomes das colunas do formato wide carregam as tres dimensoes, e sao a
    unica fonte necessaria: `submercado` vem do token `sbm`, `patamar` do token
    `pat` e `estagio_coeficiente` do token `lag` (que enumera estagios, apesar do
    nome). A coluna `estagio` continua sendo o estagio do no que construiu o
    corte, para nao confundir as duas nocoes.
    """
    gnl_columns = [c for c in cuts.columns if GNL_COLUMN_PATTERN.match(c)]
    if not gnl_columns:
        return pd.DataFrame(
            columns=[
                *CUT_KEY_COLUMNS,
                "codigo_submercado",
                "estagio_coeficiente",
                "patamar",
                "valor",
            ]
        )
    melted = cuts.melt(
        id_vars=list(CUT_KEY_COLUMNS),
        value_vars=gnl_columns,
        var_name="coluna",
        value_name="valor",
    )
    parsed = melted["coluna"].str.extract(GNL_COLUMN_PATTERN)
    melted["codigo_submercado"] = parsed[0].astype("int64")
    melted["patamar"] = parsed[1].astype("int64")
    melted["estagio_coeficiente"] = parsed[2].astype("int64")
    return (
        melted[
            [
                *CUT_KEY_COLUMNS,
                "codigo_submercado",
                "estagio_coeficiente",
                "patamar",
                "valor",
            ]
        ]
        .sort_values(
            [
                "estagio",
                "no",
                "indice_corte",
                "codigo_submercado",
                "estagio_coeficiente",
                "patamar",
            ]
        )
        .reset_index(drop=True)
    )


def build_cortdeco_tables(
    cortdeco: Cortdeco,
    geometry: CutGeometry,
    write_wide: bool,
    write_tidy: bool,
    logger: logging.Logger,
) -> dict[str, pd.DataFrame]:
    """Monta as tabelas exportaveis do cortdeco.

    Args:
        cortdeco: objeto `Cortdeco` ja lido.
        geometry: geometria conferida.
        write_wide: incluir a tabela wide (uma coluna por coeficiente).
        write_tidy: incluir as tabelas longas por bloco de coeficientes.
        logger: logger da aplicacao.

    Returns:
        Dicionario `nome_do_csv -> DataFrame`.
    """
    cuts = require_frame(cortdeco.cortes, "cortes")
    tables: dict[str, pd.DataFrame] = {}
    if write_wide:
        tables["cortdeco_cortes"] = cuts
    if write_tidy:
        # Volume armazenado e defluencia por tempo de viagem vem das propriedades
        # do idecomp, que decodificam esses dois blocos corretamente.
        tables["cortdeco_rhs"] = _rhs_table(cuts)
        tables["cortdeco_coef_volume_armazenado"] = require_frame(
            cortdeco.coeficientes_volume_armazenado, "coeficientes_volume_armazenado"
        )
        if geometry.travel_time_plant_codes:
            tables["cortdeco_coef_defluencia_tempo_viagem"] = require_frame(
                cortdeco.coeficientes_defluencia_tempo_viagem,
                "coeficientes_defluencia_tempo_viagem",
            )
        else:
            # Bloco de largura zero: a propriedade do idecomp levanta KeyError
            # nesse caso (veja o topo do modulo).
            tables["cortdeco_coef_defluencia_tempo_viagem"] = pd.DataFrame(
                {
                    name: pd.Series(dtype=dtype)
                    for name, dtype in TRAVEL_TIME_COEFFICIENT_DTYPES.items()
                }
            )
        tables["cortdeco_coef_geracao_gnl"] = _gnl_tidy_table(cuts)
        if geometry.gnl_submarket_codes:
            tables["cortdeco_coef_geracao_gnl_idecomp_bruto"] = require_frame(
                cortdeco.coeficientes_geracao_gnl, "coeficientes_geracao_gnl"
            )
        else:
            # Bloco de largura zero: a propriedade do idecomp levanta KeyError
            # nesse caso, pelo mesmo motivo do bloco de tempo de viagem.
            tables["cortdeco_coef_geracao_gnl_idecomp_bruto"] = pd.DataFrame(
                {
                    name: pd.Series(dtype=dtype)
                    for name, dtype in GNL_COEFFICIENT_DTYPES.items()
                }
            )
        logger.info(
            "tabelas tidy montadas: %d coeficientes de volume armazenado, "
            "%d de defluencia por tempo de viagem, %d de geracao GNL "
            "(esperado %d x %d cortes)",
            len(tables["cortdeco_coef_volume_armazenado"]),
            len(tables["cortdeco_coef_defluencia_tempo_viagem"]),
            len(tables["cortdeco_coef_geracao_gnl"]),
            geometry.gnl_coefficient_count,
            len(cuts),
        )
    return tables
