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
from typing import SupportsInt, TypeVar

import numpy as np
import pandas as pd
from idecomp.decomp.mapcut import Mapcut

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
            ou tamanho de arquivo incompativel com o tamanho do registro.
    """
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
        travel_time_plant_codes=tuple(
            int(c)
            for c in require_list(
                mapcut.codigos_uhes_tempo_viagem, "codigos_uhes_tempo_viagem"
            )
        ),
        gnl_submarket_codes=tuple(
            int(c)
            for c in require_list(
                mapcut.codigos_submercados_gnl, "codigos_submercados_gnl"
            )
        ),
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
