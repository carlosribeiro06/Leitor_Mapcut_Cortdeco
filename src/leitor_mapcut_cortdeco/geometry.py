"""Geometria de leitura do cortdeco, derivada do mapcut e conferida contra o arquivo.

Por que este modulo existe
--------------------------
O `cortdeco` nao e autodescritivo. Ele e um arquivo de registros de tamanho fixo
onde cada registro guarda um corte de Benders, e os cortes de um mesmo no estao
ligados por uma *lista encadeada*: o primeiro campo (`int32`) de cada registro
contem o indice do proximo registro do mesmo no, e o valor 0 encerra a cadeia.
Todo o resto - tamanho do registro, quais usinas ocupam quais posicoes, onde
comeca a cadeia de cada no - vem do `mapcut`.

O `idecomp` precisa receber essa geometria pronta em `Cortdeco.read(...)`. Este
modulo monta essa geometria a partir do mapcut e, principalmente, **confere** o
resultado contra o tamanho real do arquivo antes de qualquer leitura. Se a
geometria estiver errada, a leitura nao falha: ela devolve numeros plausiveis e
errados, o que e muito pior. Dai as tres verificacoes de integridade abaixo.

As tres verificacoes de integridade
-----------------------------------
1. **Tamanho multiplo do registro** - o arquivo tem de ser um numero inteiro de
   registros; caso contrario o tamanho do registro lido do mapcut esta errado.
2. **Contagem de registros** - o numero de registros tem de ser
   `cortes_por_no * numero_de_nos_que_constroem_corte + 1`. O `+ 1` e um
   registro extra que o DECOMP grava para o ultimo estagio que constroi cortes.
3. **Preenchimento nulo** - o registro e superdimensionado pelo DECOMP: apenas
   `4 + 8 * numero_de_coeficientes` bytes sao usados e o restante e zero. Se
   algum byte do preenchimento for diferente de zero, a contagem de coeficientes
   esta subestimada e a leitura estaria truncando dados reais.

Blocos opcionais do corte
-------------------------
Dois dos tres blocos de coeficientes podem ter largura zero, e ambos os casos sao
legitimos - nao sinal de arquivo corrompido:

* **tempo de viagem** - `numero_uhes_tempo_viagem` igual a zero, e os registros 7
  e 8 do mapcut nao existem;
* **geracao GNL** - nenhuma UTE a GNL no caso, e o registro 9 nao traz submercado
  algum.

O `idecomp` 1.14.2 nao trata nenhum dos dois, e as propriedades correspondentes
levantam excecao ou devolvem valores de outro campo. Por isso os dois blocos sao
derivados aqui do payload bruto; veja as secoes "Tempo de viagem da agua" e
"Geracao GNL antecipada" mais abaixo.

Como os codigos decodificados dimensionam os blocos de coeficientes do corte, um
erro neles desloca todo o layout em silencio - motivo pelo qual cada derivacao
exige que os blocos consumam o payload inteiro e confere os codigos contra os
escalares declarados pelo caso.

Numero de cortes por no
-----------------------
Esse valor dimensiona os vetores de leitura e **nao esta gravado explicitamente**
no mapcut: ele equivale ao numero de iteracoes da politica. Como um erro aqui
produz linhas silenciosamente zeradas, ele e obtido por duas vias independentes,
sempre comparadas entre si:

* `numero_iteracoes` do mapcut;
* geometria do arquivo: `(numero_de_registros - 1) / numero_de_nos_com_corte`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, SupportsInt, TypeVar, cast

import numpy as np
import pandas as pd
from idecomp.decomp.mapcut import Mapcut
from idecomp.decomp.modelos.mapcut import SecaoDadosMapcut

from .config import CortdecoSettings

T = TypeVar("T")

# Cada registro comeca com um int32 (indice do proximo corte do no) seguido
# pelos coeficientes em float64.
LINK_FIELD_BYTES = 4
COEFFICIENT_BYTES = 8


class GeometryError(Exception):
    """A geometria derivada do mapcut e inconsistente com o cortdeco."""


@dataclass(frozen=True)
class CutGeometry:
    """Tudo o que e preciso saber para ler o cortdeco, ja conferido.

    Attributes:
        record_size_bytes: tamanho de cada registro de corte, em bytes.
        number_of_stages: numero de estagios do estudo.
        number_of_load_blocks: numero de patamares de carga.
        max_travel_time_lag: lag maximo de tempo de viagem da agua.
        hydro_plant_codes: codigos das UHEs, na ordem em que seus coeficientes
            de volume armazenado aparecem no registro.
        travel_time_plant_codes: codigos das UHEs com tempo de viagem.
        gnl_submarket_codes: codigos dos submercados com despacho antecipado (GNL).
        cuts_per_node: numero de cortes de cada no (o valor efetivamente usado).
        cuts_per_node_from_iterations: valor obtido de `mapcut.numero_iteracoes`.
        cuts_per_node_from_file: valor derivado do tamanho do arquivo, ou None
            quando a divisao nao e exata.
        cuts_per_node_origin: qual via forneceu o valor usado.
        cut_building_node_count: numero de nos que constroem cortes.
        total_cut_records: numero de registros existentes no arquivo.
        total_cuts_reported: `mapcut.numero_cortes` (total, somando todos os nos).
        last_cut_record_per_node: tabela no/estagio/indice do ultimo corte, que e
            o ponto de entrada da lista encadeada de cada no.
    """

    record_size_bytes: int
    number_of_stages: int
    number_of_load_blocks: int
    max_travel_time_lag: int
    hydro_plant_codes: tuple[int, ...]
    travel_time_plant_codes: tuple[int, ...]
    gnl_submarket_codes: tuple[int, ...]
    cuts_per_node: int
    cuts_per_node_from_iterations: int
    cuts_per_node_from_file: int | None
    cuts_per_node_origin: str
    cut_building_node_count: int
    total_cut_records: int
    total_cuts_reported: int
    last_cut_record_per_node: pd.DataFrame

    @property
    def storage_coefficient_count(self) -> int:
        """Coeficientes de volume armazenado: um por UHE."""
        return len(self.hydro_plant_codes)

    @property
    def travel_time_coefficient_count(self) -> int:
        """Coeficientes de defluencia passada: uma UHE com tempo de viagem por lag."""
        return len(self.travel_time_plant_codes) * self.max_travel_time_lag

    @property
    def gnl_coefficient_count(self) -> int:
        """Coeficientes de geracao GNL: submercado x estagio x patamar de carga."""
        return (
            len(self.gnl_submarket_codes)
            * self.number_of_stages
            * self.number_of_load_blocks
        )

    @property
    def coefficient_count(self) -> int:
        """Total de coeficientes por corte, incluindo o termo constante (rhs)."""
        return (
            1
            + self.storage_coefficient_count
            + self.travel_time_coefficient_count
            + self.gnl_coefficient_count
        )

    @property
    def used_bytes_per_record(self) -> int:
        """Bytes efetivamente ocupados em cada registro."""
        return LINK_FIELD_BYTES + COEFFICIENT_BYTES * self.coefficient_count

    @property
    def padding_bytes_per_record(self) -> int:
        """Bytes de preenchimento no fim de cada registro (devem ser todos nulos)."""
        return self.record_size_bytes - self.used_bytes_per_record

    @property
    def expected_total_records(self) -> int:
        """Registros esperados: um por corte de cada no, mais o registro extra."""
        return self.cuts_per_node * self.cut_building_node_count + 1


# ---------------------------------------------------------------------------
# Fronteira com o idecomp
#
# Toda propriedade do `Mapcut` e do `Cortdeco` e anotada como `X | None`, e os
# escalares vem como tipos do numpy (`np.int32`). Os acessores abaixo concentram
# essa fronteira num unico lugar: eles falham com uma mensagem que nomeia o campo
# nao decodificado, em vez de deixar um `None` seguir adiante e reaparecer mais
# tarde como um erro sem contexto, e convertem os tipos do numpy para tipos
# nativos antes que eles entrem nos calculos de offset e nas mensagens de log.
# ---------------------------------------------------------------------------


def require_int(value: SupportsInt | None, name: str) -> int:
    """Exige um escalar inteiro decodificado, convertendo-o para `int` nativo."""
    if value is None:
        raise GeometryError(f"o idecomp nao decodificou o campo {name}")
    return int(value)


def require_list(value: list[T] | None, name: str) -> list[T]:
    """Exige uma lista decodificada e nao vazia."""
    if value is None:
        raise GeometryError(f"o idecomp nao decodificou o campo {name}")
    if not value:
        raise GeometryError(f"o campo {name} veio vazio do idecomp")
    return value


def require_frame(value: pd.DataFrame | None, name: str) -> pd.DataFrame:
    """Exige uma tabela decodificada e nao vazia."""
    if value is None:
        raise GeometryError(f"o idecomp nao decodificou a tabela {name}")
    if value.empty:
        raise GeometryError(f"a tabela {name} veio vazia do idecomp")
    return value


# ---------------------------------------------------------------------------
# Tempo de viagem da agua
#
# O bloco de tempo de viagem e opcional (veja o topo do modulo). Quando ele nao
# existe, o `idecomp` 1.14.2 falha de duas maneiras diferentes:
#
# * `codigos_uhes_tempo_viagem` indexa a coluna 'codigo_usina' de um DataFrame
#   que ficou sem coluna alguma, porque o laco que o preenche nao roda, e levanta
#   `KeyError`. A falha acontece dentro da propriedade, antes de haver retorno,
#   portanto nenhum dos guardas acima chega a ser alcancado;
# * `lag_tempo_viagem_por_uhe` fatia com `[-(0 * numero_estagios):]`, que em
#   Python equivale a `[0:]`, e devolve a lista inteira de `dados_estagios` em
#   vez de uma lista vazia. Esse e o pior dos dois, porque nao levanta excecao
#   nenhuma: valores de outro campo chegariam como se fossem lags.
#
# Por isso os dados de tempo de viagem sao derivados do payload bruto dos
# registros 7 e 8, e nao das propriedades da biblioteca. A contagem declarada
# pelo DECOMP (`numero_uhes_tempo_viagem`) e a autoridade sobre quantos blocos
# existem; o layout e conferido exigindo que os blocos consumam o payload
# inteiro, e os codigos decodificados sao conferidos contra a lista de UHEs do
# caso.
# ---------------------------------------------------------------------------


def raw_data_section(mapcut: Mapcut) -> dict[str, list[Any]]:
    """Devolve o payload bruto da secao de dados do mapcut.

    As correcoes de decodificacao precisam do vetor de valores como ele saiu do
    arquivo, e nao das tabelas derivadas pelo idecomp.
    """
    section = mapcut.data.get_sections_of_type(SecaoDadosMapcut)
    if section is None or isinstance(section, list):
        found = 0 if section is None else len(section)
        raise GeometryError(
            "o mapcut nao contem exatamente uma secao de dados; "
            f"foram encontradas {found}"
        )
    return cast("dict[str, list[Any]]", section.data)


def travel_time_plant_count(mapcut: Mapcut) -> int:
    """Numero de UHEs com tempo de viagem declarado pelo DECOMP.

    E a autoridade sobre a existencia do bloco: zero significa que o caso nao tem
    tempo de viagem e que nenhuma propriedade do idecomp sobre o assunto pode ser
    consultada, porque todas falham nesse caso.
    """
    count = require_int(mapcut.numero_uhes_tempo_viagem, "numero_uhes_tempo_viagem")
    if count < 0:
        raise GeometryError(f"numero_uhes_tempo_viagem negativo no mapcut: {count}")
    return count


def travel_time_lags(mapcut: Mapcut) -> list[int]:
    """Lag maximo de cada par usina/estagio, achatado por usina.

    Returns:
        Lista vazia quando o caso nao tem tempo de viagem - sem consultar a
        propriedade do idecomp, que nesse caso devolveria a lista inteira de
        `dados_estagios`.

    Raises:
        GeometryError: o vetor nao tem `numero_uhes_tempo_viagem x
            numero_estagios` valores, ou contem lag negativo.
    """
    count = travel_time_plant_count(mapcut)
    if count == 0:
        return []

    stage_count = require_int(mapcut.numero_estagios, "numero_estagios")
    lags = [
        int(v)
        for v in require_list(
            mapcut.lag_tempo_viagem_por_uhe, "lag_tempo_viagem_por_uhe"
        )
    ]
    expected = count * stage_count
    if len(lags) != expected:
        raise GeometryError(
            f"o vetor de lags de tempo de viagem tem {len(lags)} valores, mas "
            f"eram esperados {expected} ({count} UHEs x {stage_count} estagios)"
        )
    negative = sorted({lag for lag in lags if lag < 0})
    if negative:
        raise GeometryError(f"lags de tempo de viagem negativos: {negative}")
    return lags


@dataclass(frozen=True)
class TravelTimeBlock:
    """Um bloco de tempo de viagem dentro do payload dos registros 7 e 8.

    Attributes:
        plant_code: codigo da UHE a que o bloco pertence.
        travel_hours: tempo de viagem da agua, em horas.
        stage_lags: lag maximo de cada estagio, na ordem dos estagios.
        pair_start: posicao, no payload, do primeiro par
            (indice_lag, coeficiente) do bloco.
    """

    plant_code: int
    travel_hours: int
    stage_lags: tuple[int, ...]
    pair_start: int

    @property
    def pair_count(self) -> int:
        """Pares (indice_lag, coeficiente) do bloco: os lags vao de 0 a lag_maximo."""
        return sum(lag + 1 for lag in self.stage_lags)

    @property
    def pair_stop(self) -> int:
        """Posicao imediatamente apos o ultimo valor do bloco."""
        return self.pair_start + 2 * self.pair_count


def travel_time_blocks(mapcut: Mapcut) -> list[TravelTimeBlock]:
    """Localiza os blocos de tempo de viagem no payload bruto do mapcut.

    Layout do payload (registros 7 e 8), por usina com tempo de viagem:

        [codigo_usina, numero_horas, (indice_lag, coeficiente) x N]

    onde `N = soma sobre os estagios de (lag_maximo + 1)`, porque os lags vao de
    0 a `lag_maximo` inclusive. Os blocos das usinas sao consecutivos, e o offset
    tem de ser acumulado de um bloco para o proximo - e exatamente onde o
    `idecomp` erra, junto com a nao reinicializacao dos acumuladores.

    Exigir que os blocos consumam o payload inteiro e o que confere, por uma via
    independente, que a contagem declarada e o layout assumido concordam.

    Returns:
        Um bloco por UHE com tempo de viagem, na ordem do payload; lista vazia
        quando o caso nao tem tempo de viagem.

    Raises:
        GeometryError: o payload termina antes do previsto, ou sobram valores
            depois do ultimo bloco.
    """
    count = travel_time_plant_count(mapcut)
    if count == 0:
        return []

    stage_count = require_int(mapcut.numero_estagios, "numero_estagios")
    lags = travel_time_lags(mapcut)
    raw = raw_data_section(mapcut)["dados_tempo_viagem"]

    blocks: list[TravelTimeBlock] = []
    offset = 0
    for position in range(count):
        if offset + 2 > len(raw):
            raise GeometryError(
                f"o payload de tempo de viagem acabou na UHE {position + 1} de "
                f"{count}: o bloco comecaria em {offset} e o payload tem "
                f"{len(raw)} valores"
            )
        block = TravelTimeBlock(
            plant_code=int(raw[offset]),
            travel_hours=int(raw[offset + 1]),
            stage_lags=tuple(
                lags[position * stage_count : (position + 1) * stage_count]
            ),
            pair_start=offset + 2,
        )
        if block.pair_stop > len(raw):
            raise GeometryError(
                f"o bloco de tempo de viagem da UHE {block.plant_code} exigiria "
                f"{block.pair_stop} valores, mas o payload tem {len(raw)}"
            )
        blocks.append(block)
        offset = block.pair_stop

    if offset != len(raw):
        raise GeometryError(
            f"a leitura dos blocos de tempo de viagem consumiu {offset} dos "
            f"{len(raw)} valores do payload. O layout assumido nao corresponde "
            "ao arquivo."
        )
    return blocks


def travel_time_plant_codes(mapcut: Mapcut) -> tuple[int, ...]:
    """Codigos das UHEs com tempo de viagem, na ordem do payload.

    Os codigos decodificados sao conferidos contra a lista de UHEs do caso: uma
    usina com tempo de viagem que nao esteja entre as UHEs do mapcut, ou um
    codigo repetido, denunciam que o layout assumido nao corresponde ao arquivo.

    Returns:
        Tupla vazia quando o caso nao tem tempo de viagem.

    Raises:
        GeometryError: codigo repetido, ou ausente da lista de UHEs do caso.
    """
    codes = tuple(block.plant_code for block in travel_time_blocks(mapcut))
    if not codes:
        return ()

    if len(set(codes)) != len(codes):
        raise GeometryError(
            f"codigos de UHE com tempo de viagem repetidos: {sorted(codes)}"
        )
    known = {int(c) for c in require_list(mapcut.codigos_uhes, "codigos_uhes")}
    unknown = sorted(set(codes) - known)
    if unknown:
        raise GeometryError(
            f"as UHEs com tempo de viagem {unknown} nao estao entre as UHEs do "
            "caso. O layout assumido para os registros 7 e 8 nao corresponde ao "
            "arquivo."
        )
    return codes


def travel_time_register_counts(mapcut: Mapcut) -> tuple[int, int]:
    """Registros de tempo de viagem que existem, e os que o idecomp consome.

    A seccao de tempo de viagem ocupa um registro de 48020 bytes por cada lag de
    cada par usina/estagio. O `idecomp` decide quantos registros ler indexando o
    vetor de lags por `(usina * estagio) - 1`, quando o indice correto seria
    `(usina - 1) * numero_estagios + estagio - 1`. As duas formulas coincidem
    apenas quando todos os lags sao iguais - o caso mais comum, e o unico
    observado ate agora.

    Quando nao coincidem, o `idecomp` para de ler a seccao no lugar errado e
    **todos os registros seguintes saem de posicao**: os registros 9 e 10, isto e
    os dados de GNL e de custos, passam a ser lidos de outro trecho do arquivo.
    Nada nisso levanta excecao, e nenhum numero derivado desses payloads e
    confiavel - por isso a divergencia e tratada como erro em
    `build_cut_geometry`.

    Returns:
        O par (registros existentes, registros que o idecomp le). Vale (0, 0)
        quando o caso nao tem UHE com tempo de viagem.
    """
    count = travel_time_plant_count(mapcut)
    if count == 0:
        return (0, 0)

    stage_count = require_int(mapcut.numero_estagios, "numero_estagios")
    lags = travel_time_lags(mapcut)

    actual = sum(
        lags[plant * stage_count + stage] + 1
        for plant in range(count)
        for stage in range(stage_count)
    )
    consumed = 0
    for plant in range(1, count + 1):
        for stage in range(1, stage_count + 1):
            index = plant * stage - 1
            if 0 <= index < len(lags):
                consumed += lags[index] + 1
    return (actual, consumed)


# ---------------------------------------------------------------------------
# Geracao GNL antecipada
#
# O registro 9 do mapcut traz, por estagio:
#
#     [numero_utes_gnl,
#      codigo_submercado x n, indice_lag x n, numero_patamares x n,  (int32)
#      valores x (numero_estagios * n)]                             (float64)
#
# O passo entre estagios e portanto `1 + 3*n + numero_estagios*n`. O `idecomp`
# usa `1 + 4*n + soma(numero_patamares)`, que nao coincide com o verdadeiro em
# nenhum dos decks observados, e o efeito depende do tamanho do payload:
#
# * com muitos estagios o passo errado e MENOR que o verdadeiro, o offset avanca
#   devagar, permanece dentro do payload e decodifica lixo sem levantar excecao -
#   apenas o primeiro estagio sai correto;
# * com poucos estagios o payload e curto, o offset cai no meio de um registro,
#   le um codigo de submercado como se fosse `numero_utes_gnl` e toma reais do
#   bloco de valores como se fossem patamares. O passo resultante estoura o
#   payload e a propriedade levanta `IndexError`.
#
# Por isso os dados de GNL - inclusive os codigos de submercado que dimensionam o
# bloco de coeficientes do corte - sao derivados aqui do payload bruto.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GnlStageBlock:
    """O registro de GNL de um estagio dentro do payload do registro 9.

    Attributes:
        stage: estagio a que o registro pertence, comecando em 1.
        plant_count: numero de UTEs a GNL declarado no registro.
        submarket_codes: codigo do submercado de cada UTE, na ordem do payload.
        lag_indices: indice de lag (antecipacao do despacho) de cada UTE.
        load_block_counts: numero de patamares de cada UTE.
        value_start: posicao, no payload, do primeiro real do bloco de valores.
        value_stop: posicao imediatamente apos o ultimo valor do registro.
    """

    stage: int
    plant_count: int
    submarket_codes: tuple[int, ...]
    lag_indices: tuple[int, ...]
    load_block_counts: tuple[int, ...]
    value_start: int
    value_stop: int


def gnl_stage_blocks(mapcut: Mapcut) -> list[GnlStageBlock]:
    """Localiza os registros de GNL no payload bruto, um por estagio.

    Exigir que os registros consumam o payload inteiro e o que confere, por uma
    via independente, que o passo verdadeiro e o layout assumido concordam.

    Returns:
        Um bloco por estagio; lista vazia quando o caso nao tem UTE a GNL.

    Raises:
        GeometryError: o payload termina antes do previsto, sobram valores depois
            do ultimo estagio, ou um registro declara contagem negativa.
    """
    raw = raw_data_section(mapcut)["dados_gnl"]
    stage_count = require_int(mapcut.numero_estagios, "numero_estagios")
    if not raw:
        return []

    blocks: list[GnlStageBlock] = []
    offset = 0
    for stage in range(1, stage_count + 1):
        if offset >= len(raw):
            raise GeometryError(
                f"o payload de GNL acabou no estagio {stage} de {stage_count}: o "
                f"registro comecaria em {offset} e o payload tem {len(raw)} valores"
            )
        plant_count = int(raw[offset])
        if plant_count < 0:
            raise GeometryError(
                f"o registro de GNL do estagio {stage} declara {plant_count} UTEs"
            )
        stride = 1 + 3 * plant_count + stage_count * plant_count
        if offset + stride > len(raw):
            raise GeometryError(
                f"o registro de GNL do estagio {stage} exigiria {stride} valores a "
                f"partir do offset {offset}, mas o payload tem {len(raw)}"
            )
        first = offset + 1
        blocks.append(
            GnlStageBlock(
                stage=stage,
                plant_count=plant_count,
                submarket_codes=tuple(int(v) for v in raw[first : first + plant_count]),
                lag_indices=tuple(
                    int(v) for v in raw[first + plant_count : first + 2 * plant_count]
                ),
                load_block_counts=tuple(
                    int(v)
                    for v in raw[first + 2 * plant_count : first + 3 * plant_count]
                ),
                value_start=first + 3 * plant_count,
                value_stop=offset + stride,
            )
        )
        offset += stride

    if offset != len(raw):
        raise GeometryError(
            f"a leitura dos registros de GNL consumiu {offset} dos {len(raw)} "
            "valores do payload. O layout assumido nao corresponde ao arquivo."
        )
    return blocks


def gnl_submarket_codes(mapcut: Mapcut) -> tuple[int, ...]:
    """Codigos dos submercados com despacho antecipado de GNL.

    Os codigos aparecem repetidos, um conjunto por estagio; aqui eles sao
    reduzidos preservando a ordem de primeira ocorrencia, que e a ordem em que os
    coeficientes de GNL aparecem dentro do corte.

    Returns:
        Tupla vazia quando o caso nao tem UTE a GNL.

    Raises:
        GeometryError: um estagio repete o mesmo submercado, ou o total de
            submercados distintos excede `numero_submercados` do caso.
    """
    blocks = gnl_stage_blocks(mapcut)
    if not blocks:
        return ()

    codes: list[int] = []
    for block in blocks:
        if len(set(block.submarket_codes)) != len(block.submarket_codes):
            raise GeometryError(
                f"o registro de GNL do estagio {block.stage} repete submercados "
                f"({list(block.submarket_codes)}). O layout assumido para o "
                "registro 9 nao corresponde ao arquivo."
            )
        for code in block.submarket_codes:
            if code not in codes:
                codes.append(code)

    submarket_count = require_int(mapcut.numero_submercados, "numero_submercados")
    if len(codes) > submarket_count:
        raise GeometryError(
            f"foram decodificados {len(codes)} submercados com GNL ({sorted(codes)}), "
            f"mais do que os {submarket_count} submercados do caso. O layout "
            "assumido para o registro 9 nao corresponde ao arquivo."
        )
    return tuple(codes)


def _uniform_load_blocks(load_blocks_per_stage: list[int]) -> int:
    """Reduz os patamares por estagio a um unico valor, exigindo uniformidade.

    O leitor de cortes do `idecomp` recebe um unico numero de patamares e o aplica
    a todos os estagios. Se o caso tiver estagios com numeros diferentes de
    patamares, essa premissa nao vale e a largura do bloco GNL sairia errada -
    entao o correto e parar, e nao escolher um valor no lugar do usuario.
    """
    values = {int(v) for v in load_blocks_per_stage}
    if len(values) > 1:
        raise GeometryError(
            "o caso tem numero de patamares de carga diferente entre estagios "
            f"({sorted(values)}). O leitor de cortes do idecomp assume um unico "
            "valor para todos os estagios, portanto a largura do bloco de "
            "coeficientes GNL nao pode ser determinada com seguranca. "
            "Revise o caso antes de prosseguir."
        )
    return values.pop()


def _count_cut_building_nodes(last_cut_record_per_node: pd.DataFrame) -> int:
    """Conta os nos que constroem cortes.

    O ultimo estagio do caso nao constroi cortes (nao ha funcao de custo futuro
    depois dele), logo seus nos ficam de fora. Esse mesmo numero e o passo
    (`stride`) entre registros consecutivos de um mesmo no dentro do cortdeco.
    """
    if last_cut_record_per_node.empty:
        raise GeometryError(
            "a tabela de ultimo corte por no, lida do mapcut, esta vazia"
        )
    last_stage = last_cut_record_per_node["estagio"].max()
    count = int((last_cut_record_per_node["estagio"] != last_stage).sum())
    if count <= 0:
        raise GeometryError(
            "nenhum no constroi cortes segundo o mapcut; o caso tem apenas um estagio?"
        )
    return count


def _resolve_cuts_per_node(
    from_iterations: int,
    from_file: int | None,
    settings: CortdecoSettings,
    logger: logging.Logger,
) -> tuple[int, str]:
    """Escolhe o numero de cortes por no e registra a conferencia entre as vias.

    Returns:
        O valor escolhido e um rotulo indicando de onde ele veio.
    """
    if settings.cuts_per_node_override is not None:
        logger.warning(
            "cortes por no fixado manualmente em %d via "
            "cortdeco.cuts_per_node_override (numero_iteracoes=%s, "
            "geometria do arquivo=%s)",
            settings.cuts_per_node_override,
            from_iterations,
            from_file,
        )
        return settings.cuts_per_node_override, "cuts_per_node_override"

    if from_file is None:
        logger.warning(
            "nao foi possivel derivar os cortes por no da geometria do arquivo "
            "(a divisao nao e exata); usando numero_iteracoes=%d",
            from_iterations,
        )
    elif from_file != from_iterations:
        logger.warning(
            "DIVERGENCIA na geometria: mapcut.numero_iteracoes=%d mas a "
            "geometria do arquivo indica %d cortes por no. Sera usada a fonte "
            "configurada (%s). Confira o par mapcut/cortdeco antes de usar os "
            "CSVs em estudo oficial.",
            from_iterations,
            from_file,
            settings.cuts_per_node_source,
        )
    else:
        logger.info(
            "cortes por no conferidos por duas vias independentes: %d "
            "(numero_iteracoes e geometria do arquivo concordam)",
            from_iterations,
        )

    if settings.cuts_per_node_source == "file_geometry":
        if from_file is None:
            raise GeometryError(
                "cortdeco.cuts_per_node_source='file_geometry', mas "
                "(numero_de_registros - 1) nao e divisivel pelo numero de nos "
                "que constroem cortes; use 'numero_iteracoes' ou informe "
                "cuts_per_node_override"
            )
        return from_file, "file_geometry"
    return from_iterations, "numero_iteracoes"


def _require_travel_time_register_alignment(mapcut: Mapcut) -> None:
    """Interrompe a execucao se o idecomp leu a seccao de tempo de viagem torta.

    Ver `travel_time_register_counts`: quando as duas contagens divergem, os
    payloads de GNL e de custos vieram de posicoes erradas do arquivo. Parar aqui
    e obrigatorio, porque nada a jusante consegue detectar isso - os numeros saem
    plausiveis e errados.
    """
    actual, consumed = travel_time_register_counts(mapcut)
    if actual != consumed:
        raise GeometryError(
            f"o idecomp leria {consumed} registros de tempo de viagem, mas o "
            f"arquivo tem {actual}. Isso acontece quando os lags nao sao todos "
            "iguais, porque a biblioteca indexa o vetor de lags por "
            "(usina * estagio) em vez de (usina - 1) * numero_estagios + estagio. "
            "Com a leitura desalinhada, os registros de GNL e de custos foram "
            "lidos de outro trecho do arquivo e nenhum valor derivado deles e "
            "confiavel. Este caso exige correcao no leitor antes de prosseguir."
        )


def build_cut_geometry(
    mapcut: Mapcut,
    cortdeco_path: Path,
    settings: CortdecoSettings,
    logger: logging.Logger,
) -> CutGeometry:
    """Monta a geometria de leitura do cortdeco a partir do mapcut.

    Args:
        mapcut: objeto `Mapcut` ja lido.
        cortdeco_path: caminho do binario de cortes, usado para derivar o numero
            de cortes por no a partir do tamanho do arquivo.
        settings: preferencias da secao 'cortdeco' do settings.json.
        logger: logger da aplicacao.

    Returns:
        A geometria pronta para alimentar `Cortdeco.read`.

    Raises:
        GeometryError: mapcut incompleto, patamares nao uniformes entre estagios,
            tamanho de arquivo incompativel com o tamanho do registro, ou
            desalinhamento na leitura dos registros de tempo de viagem.
    """
    _require_travel_time_register_alignment(mapcut)

    record_size_bytes = require_int(mapcut.tamanho_corte, "tamanho_corte")
    if record_size_bytes <= 0:
        raise GeometryError(
            f"tamanho de registro invalido no mapcut: {record_size_bytes} bytes"
        )

    last_cut_record_per_node = require_frame(
        mapcut.registro_ultimo_corte_no, "registro_ultimo_corte_no"
    )
    cut_building_node_count = _count_cut_building_nodes(last_cut_record_per_node)

    file_size = cortdeco_path.stat().st_size
    if file_size % record_size_bytes != 0:
        raise GeometryError(
            f"{cortdeco_path.name} tem {file_size} bytes, que nao e multiplo do "
            f"tamanho do registro informado pelo mapcut ({record_size_bytes} bytes; "
            f"resto {file_size % record_size_bytes}). O par mapcut/cortdeco "
            "provavelmente nao pertence ao mesmo caso."
        )
    total_cut_records = file_size // record_size_bytes

    # Desconta o registro extra gravado pelo DECOMP antes de dividir pelos nos.
    remainder = (total_cut_records - 1) % cut_building_node_count
    cuts_per_node_from_file = (
        (total_cut_records - 1) // cut_building_node_count if remainder == 0 else None
    )
    cuts_per_node_from_iterations = require_int(
        mapcut.numero_iteracoes, "numero_iteracoes"
    )
    cuts_per_node, origin = _resolve_cuts_per_node(
        cuts_per_node_from_iterations, cuts_per_node_from_file, settings, logger
    )

    geometry = CutGeometry(
        record_size_bytes=record_size_bytes,
        number_of_stages=require_int(mapcut.numero_estagios, "numero_estagios"),
        number_of_load_blocks=_uniform_load_blocks(
            require_list(mapcut.patamares_por_estagio, "patamares_por_estagio")
        ),
        max_travel_time_lag=require_int(
            mapcut.maximo_lag_tempo_viagem, "maximo_lag_tempo_viagem"
        ),
        hydro_plant_codes=tuple(
            int(c) for c in require_list(mapcut.codigos_uhes, "codigos_uhes")
        ),
        travel_time_plant_codes=travel_time_plant_codes(mapcut),
        gnl_submarket_codes=gnl_submarket_codes(mapcut),
        cuts_per_node=cuts_per_node,
        cuts_per_node_from_iterations=cuts_per_node_from_iterations,
        cuts_per_node_from_file=cuts_per_node_from_file,
        cuts_per_node_origin=origin,
        cut_building_node_count=cut_building_node_count,
        total_cut_records=total_cut_records,
        total_cuts_reported=require_int(mapcut.numero_cortes, "numero_cortes"),
        last_cut_record_per_node=last_cut_record_per_node,
    )
    _log_geometry(geometry, logger)
    return geometry


def _log_geometry(geometry: CutGeometry, logger: logging.Logger) -> None:
    """Registra a geometria em detalhe, para o log bastar como trilha de auditoria."""
    if not geometry.travel_time_plant_codes:
        logger.info(
            "o caso nao tem UHE com tempo de viagem da agua "
            "(numero_uhes_tempo_viagem = 0): o bloco de coeficientes de "
            "defluencia passada nao existe dentro do corte"
        )
    if not geometry.gnl_submarket_codes:
        logger.info(
            "o caso nao tem UTE a GNL: o bloco de coeficientes de geracao "
            "antecipada nao existe dentro do corte"
        )
    logger.info(
        "geometria do corte: %d bytes por registro, %d coeficientes uteis "
        "(%d bytes) + %d bytes de preenchimento",
        geometry.record_size_bytes,
        geometry.coefficient_count,
        geometry.used_bytes_per_record,
        geometry.padding_bytes_per_record,
    )
    logger.info(
        "blocos de coeficientes: 1 rhs + %d volume armazenado + %d defluencia "
        "por tempo de viagem (%d UHEs x %d lags) + %d geracao GNL "
        "(%d submercados x %d estagios x %d patamares)",
        geometry.storage_coefficient_count,
        geometry.travel_time_coefficient_count,
        len(geometry.travel_time_plant_codes),
        geometry.max_travel_time_lag,
        geometry.gnl_coefficient_count,
        len(geometry.gnl_submarket_codes),
        geometry.number_of_stages,
        geometry.number_of_load_blocks,
    )
    logger.info(
        "registros: %d no arquivo; %d nos constroem cortes x %d cortes por no "
        "+ 1 registro extra = %d esperados (origem dos cortes por no: %s)",
        geometry.total_cut_records,
        geometry.cut_building_node_count,
        geometry.cuts_per_node,
        geometry.expected_total_records,
        geometry.cuts_per_node_origin,
    )


def validate_geometry(
    geometry: CutGeometry,
    cortdeco_path: Path,
    logger: logging.Logger,
) -> None:
    """Confere a geometria contra o conteudo real do cortdeco.

    Executa as verificacoes 2 e 3 descritas no topo do modulo (a 1 ja ocorreu em
    `build_cut_geometry`). Uma contagem de registros diferente da esperada gera
    WARNING - a leitura ainda funciona, porque a lista encadeada tem terminador
    proprio -, mas preenchimento nao nulo e erro, porque significaria estar
    descartando coeficientes reais.

    Raises:
        GeometryError: os coeficientes nao cabem no registro, ou o preenchimento
            do registro contem bytes nao nulos.
    """
    if geometry.padding_bytes_per_record < 0:
        raise GeometryError(
            f"os {geometry.coefficient_count} coeficientes deduzidos do mapcut "
            f"ocupam {geometry.used_bytes_per_record} bytes, mais do que o "
            f"registro de {geometry.record_size_bytes} bytes. A geometria esta "
            "errada e a leitura invadiria o registro seguinte."
        )

    if geometry.total_cut_records != geometry.expected_total_records:
        logger.warning(
            "contagem de registros inesperada: o arquivo tem %d registros, "
            "mas a geometria preve %d. A leitura segue (a lista encadeada tem "
            "terminador proprio), porem confira o par mapcut/cortdeco.",
            geometry.total_cut_records,
            geometry.expected_total_records,
        )

    non_zero = _count_non_zero_padding(geometry, cortdeco_path)
    if non_zero > 0:
        raise GeometryError(
            f"{non_zero} bytes nao nulos encontrados na regiao de preenchimento "
            f"dos registros (apos os {geometry.used_bytes_per_record} bytes "
            "uteis). Isso indica que o numero de coeficientes deduzido do mapcut "
            f"({geometry.coefficient_count}) esta subestimado e que a leitura "
            "estaria truncando coeficientes reais."
        )
    logger.info(
        "integridade confirmada: os %d bytes de preenchimento de cada um dos %d "
        "registros sao inteiramente nulos, logo os %d coeficientes cobrem todo o "
        "conteudo util do corte",
        geometry.padding_bytes_per_record,
        geometry.total_cut_records,
        geometry.coefficient_count,
    )


def _count_non_zero_padding(geometry: CutGeometry, cortdeco_path: Path) -> int:
    """Conta bytes nao nulos na regiao de preenchimento de todos os registros."""
    if geometry.padding_bytes_per_record == 0:
        return 0
    non_zero = 0
    with cortdeco_path.open("rb") as handle:
        for record_index in range(geometry.total_cut_records):
            handle.seek(
                record_index * geometry.record_size_bytes
                + geometry.used_bytes_per_record
            )
            padding = handle.read(geometry.padding_bytes_per_record)
            non_zero += len(padding) - padding.count(0)
    return non_zero


def downstream_plant_indices(mapcut: Mapcut) -> np.ndarray:
    """Recupera os indices de usina a jusante gravados no registro 4 do mapcut.

    Correcao de defeito do idecomp 1.14.2
    -------------------------------------
    O registro 4 do mapcut guarda a topologia da cascata como `int32`, mas o
    `idecomp` o le como `float32`. O resultado e uma lista de denormais sem
    sentido (3e-45, 4e-45, ...), que sao na verdade os bits dos inteiros
    reinterpretados. Aqui os bits sao lidos de volta como `int32`.

    O valor recuperado e um **indice posicional 1-based** na lista
    `codigos_uhes`, e nao um codigo de usina (verificado: os valores ficam todos
    em `[0, numero_de_uhes]` e incluem numeros que nao sao codigos validos de
    usina). O valor 0 significa que a usina nao tem jusante no caso.

    Returns:
        Vetor de indices, com o mesmo comprimento de `codigos_uhes`.
    """
    raw = require_list(mapcut.codigos_uhes_jusante, "codigos_uhes_jusante")
    reinterpreted: np.ndarray = (
        np.asarray(raw, dtype=np.float32).view(np.int32).astype(np.int64)
    )
    return reinterpreted
