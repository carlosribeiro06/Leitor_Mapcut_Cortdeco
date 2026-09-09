"""Fixtures compartilhadas pelos testes.

Duas estrategias convivem aqui:

* **caso sintetico** - um mapcut falso e um cortdeco minusculo gravado byte a
  byte, com valores escolhidos para que cada coeficiente lido possa ser
  conferido individualmente. E o que permite testar a geometria e a leitura da
  lista encadeada sem depender de nenhum arquivo grande.
* **caso real** - os binarios `mapcut.rv0` / `cortdeco.rv0` da raiz do projeto,
  usados nos testes de integracao. Quando nao estao presentes (uma copia limpa
  do repositorio, por exemplo), esses testes sao ignorados em vez de falhar.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from leitor_mapcut_cortdeco.config import load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dimensoes do caso sintetico. Sao pequenas de proposito, para que o arquivo
# gerado caiba em algumas centenas de bytes e possa ser conferido na mao.
SYNTHETIC_STAGES = 3
SYNTHETIC_PLANT_CODES = (11, 22)
SYNTHETIC_TRAVEL_TIME_PLANT_CODES = (22,)
SYNTHETIC_GNL_SUBMARKET_CODES = (1,)
SYNTHETIC_MAX_TRAVEL_TIME_LAG = 2
SYNTHETIC_LOAD_BLOCKS = 2
SYNTHETIC_CUTS_PER_NODE = 3
SYNTHETIC_RECORD_SIZE = 128

# 1 rhs + 2 volume armazenado + 1 UHE x 2 lags + 1 submercado x 3 estagios x 2 patamares
SYNTHETIC_COEFFICIENT_COUNT = 11

# Tempo de viagem do caso sintetico: uma usina, com o mesmo lag maximo em todos
# os estagios. O payload dos registros 7 e 8 e montado a partir daqui.
SYNTHETIC_TRAVEL_TIME_HOURS = 6
SYNTHETIC_TRAVEL_TIME_STAGE_LAGS = (SYNTHETIC_MAX_TRAVEL_TIME_LAG,) * SYNTHETIC_STAGES

# O mesmo caso, mas sem nenhuma UHE com tempo de viagem: o bloco de defluencia
# passada desaparece e sobram 1 rhs + 2 volume armazenado + 6 GNL coeficientes.
SYNTHETIC_COEFFICIENT_COUNT_WITHOUT_TRAVEL_TIME = 9

# Valores espurios que o idecomp devolve em `lag_tempo_viagem_por_uhe` quando o
# caso nao tem tempo de viagem: o fatiamento `[-(0 * numero_estagios):]` equivale
# a `[0:]` e entrega a lista `dados_estagios` inteira. O duble reproduz isso para
# que os testes provem que o codigo de producao nao consome esse vetor.
SYNTHETIC_BOGUS_TRAVEL_TIME_LAGS = (1, 3, 6, 0, 0, 1, 2, 3)

# GNL do caso sintetico: um submercado, com a mesma antecipacao em todos os
# estagios. O payload do registro 9 e montado a partir daqui.
SYNTHETIC_GNL_LAG = 2
SYNTHETIC_SUBMARKET_COUNT = 2

# O mesmo caso, mas sem nenhuma UTE a GNL: sobram
# 1 rhs + 2 volume armazenado + 1 UHE x 2 lags de tempo de viagem.
SYNTHETIC_COEFFICIENT_COUNT_WITHOUT_GNL = 5

# Caso de GNL com dois submercados e tres patamares. Existe para reproduzir a
# condicao em que o passo errado do idecomp SAI do payload: o passo verdadeiro
# (1 + 3n + nper*n) e o dele (1 + 4n + soma(patamares)) coincidem apenas quando
# nper == numero_patamares + 1, o que vale no sintetico padrao (3 == 2 + 1) mas
# nao aqui (3 != 3 + 1). Com dois submercados o offset dele cai no meio do
# registro seguinte, le um codigo de submercado como se fosse a contagem de UTEs
# e toma reais do bloco de valores como se fossem patamares.
SYNTHETIC_GNL_SUBMARKET_CODES_MAIOR = (1, 3)
SYNTHETIC_LOAD_BLOCKS_MAIOR = 3

# 1 rhs + 2 volume armazenado + 1 UHE x 2 lags + 2 submercados x 3 estagios x 3
# patamares. Nao cabe no registro de 128 bytes do sintetico padrao.
SYNTHETIC_COEFFICIENT_COUNT_GNL_MAIOR = 23
SYNTHETIC_RECORD_SIZE_GNL_MAIOR = 256

# Encadeamento dos registros (1-based) de cada no, na ordem em que sao lidos.
# O no 1 (estagio 1) tem 3 cortes; o no 2 (estagio 2) tem 4, porque e o ultimo
# estagio que constroi cortes e o DECOMP grava um registro extra.
SYNTHETIC_CHAINS: dict[int, tuple[int, ...]] = {
    1: (6, 4, 2),
    2: (7, 5, 3, 1),
}
SYNTHETIC_TOTAL_RECORDS = 7


@dataclass
class FakeSection:
    """Secao de dados falsa: expoe apenas o payload bruto do mapcut."""

    data: dict[str, list[Any]]


@dataclass
class FakeSectionFile:
    """Reproduz o acesso `mapcut.data.get_sections_of_type(...)` do cfinterface."""

    section: FakeSection

    def get_sections_of_type(self, _section_type: Any) -> FakeSection:
        return self.section


@dataclass
class FakeMapcut:
    """Dublê do objeto `Mapcut` com apenas os atributos que a geometria consome."""

    tamanho_corte: int
    numero_iteracoes: int
    numero_cortes: int
    numero_estagios: int
    numero_submercados: int
    patamares_por_estagio: list[int]
    numero_uhes_tempo_viagem: int
    maximo_lag_tempo_viagem: int
    lag_tempo_viagem_por_uhe: list[int]
    codigos_uhes: list[int]
    registro_ultimo_corte_no: pd.DataFrame
    codigos_uhes_jusante: list[Any]
    data: FakeSectionFile

    @property
    def codigos_submercados_gnl(self) -> list[int]:
        """Tripwire: o codigo de producao nao pode consultar esta propriedade.

        Mesma razao de `codigos_uhes_tempo_viagem`: a propriedade do idecomp
        derruba a leitura (levanta `IndexError` quando o passo errado dela sai do
        payload) ou devolve lixo silencioso (so o primeiro estagio decodificado).
        Os codigos passaram a vir do payload bruto, e o duble levanta sempre para
        que qualquer reincidencia apareca como falha de teste.
        """
        raise IndexError("list index out of range")

    @property
    def dados_tempo_viagem(self) -> pd.DataFrame:
        """Saida bruta do idecomp, so para a contagem que vai ao log.

        O codigo de producao consulta esta propriedade em um unico lugar: a linha
        de log que compara quantas linhas a reconstrucao produziu com quantas a
        biblioteca produziria. O duble nao reproduz o defeito de acumulador - o
        numero de linhas aqui e arbitrario e nenhum teste depende dele.
        """
        return pd.DataFrame(
            {
                "codigo_usina": [SYNTHETIC_TRAVEL_TIME_PLANT_CODES[0]],
                "numero_horas": [SYNTHETIC_TRAVEL_TIME_HOURS],
                "estagio": [1],
                "indice_lag": [0],
                "coeficiente_amortecimento": [1.0],
            }
        )

    @property
    def dados_gnl(self) -> pd.DataFrame:
        """Idem, para o bloco de GNL: existe apenas para a contagem do log."""
        return pd.DataFrame(
            {
                "estagio": [1],
                "numero_utes_gnl": [1],
                "codigo_submercado": [SYNTHETIC_GNL_SUBMARKET_CODES[0]],
                "indice_lag": [SYNTHETIC_GNL_LAG],
                "numero_patamares": [SYNTHETIC_LOAD_BLOCKS],
            }
        )

    @property
    def codigos_uhes_tempo_viagem(self) -> list[int]:
        """Tripwire: o codigo de producao nao pode consultar esta propriedade.

        Os codigos das UHEs com tempo de viagem passaram a ser derivados do
        payload bruto justamente porque esta propriedade do idecomp e
        inutilizavel - com `numero_uhes_tempo_viagem = 0` ela levanta
        `KeyError('codigo_usina')`, e com duas usinas ou mais pode atribuir o
        codigo errado. O duble levanta sempre, para que qualquer reincidencia
        apareca como falha de teste em vez de passar silenciosamente.
        """
        raise KeyError("codigo_usina")


def synthetic_last_cut_table() -> pd.DataFrame:
    """Tabela de entrada da lista encadeada de cada no do caso sintetico.

    O ultimo estagio nao constroi cortes, por isso seu no tem indice 0 - valor
    que tambem sinaliza ao leitor do idecomp o fim da varredura.
    """
    return pd.DataFrame(
        {
            "no": [1, 2, 3],
            "estagio": [1, 2, 3],
            # O no do estagio 2 aponta para 5; o leitor soma o passo (2 nos) para
            # alcancar o registro extra 7.
            "indice_ultimo_corte": [6, 5, 0],
        }
    )


def synthetic_gnl_payload(
    stage_count: int = SYNTHETIC_STAGES,
    submarket_codes: tuple[int, ...] = SYNTHETIC_GNL_SUBMARKET_CODES,
    load_blocks: int = SYNTHETIC_LOAD_BLOCKS,
) -> list[Any]:
    """Payload do registro 9 do mapcut: um registro por estagio.

    Layout de cada registro:
    `[numero_utes_gnl, codigo_submercado x n, indice_lag x n,
      numero_patamares x n, valores x (numero_estagios * n)]`.

    Os parametros ficam abertos porque o passo verdadeiro
    (`1 + 3n + nper*n`) e o passo errado do idecomp
    (`1 + 4n + soma(patamares)`) coincidem exatamente quando
    `nper == numero_patamares + 1` - o caso do sintetico padrao. Para exercitar a
    divergencia e preciso montar um payload com outras dimensoes.
    """
    count = len(submarket_codes)
    payload: list[Any] = []
    for stage in range(1, stage_count + 1):
        payload.append(count)
        payload += list(submarket_codes)
        payload += [SYNTHETIC_GNL_LAG] * count
        payload += [load_blocks] * count
        payload += [float(stage * 100 + i) for i in range(stage_count * count)]
    return payload


def synthetic_travel_time_payload() -> list[Any]:
    """Payload dos registros 7 e 8 para a unica UHE com tempo de viagem.

    Layout: `[codigo_usina, numero_horas, (indice_lag, coeficiente) x N]`, com os
    lags de cada estagio indo de 0 a `lag_maximo` inclusive. O coeficiente
    codifica estagio e lag no proprio valor, para que o teste possa afirmar que
    cada linha reconstruida veio da posicao certa do payload.
    """
    payload: list[Any] = [
        SYNTHETIC_TRAVEL_TIME_PLANT_CODES[0],
        SYNTHETIC_TRAVEL_TIME_HOURS,
    ]
    for stage, lag_maximum in enumerate(SYNTHETIC_TRAVEL_TIME_STAGE_LAGS, start=1):
        for lag in range(lag_maximum + 1):
            payload += [lag, stage + lag / 10]
    return payload


@pytest.fixture
def fake_mapcut() -> FakeMapcut:
    """Mapcut sintetico coerente com o cortdeco de `synthetic_cortdeco`."""
    return FakeMapcut(
        tamanho_corte=SYNTHETIC_RECORD_SIZE,
        numero_iteracoes=SYNTHETIC_CUTS_PER_NODE,
        numero_cortes=SYNTHETIC_CUTS_PER_NODE * 2,
        numero_estagios=SYNTHETIC_STAGES,
        numero_submercados=SYNTHETIC_SUBMARKET_COUNT,
        patamares_por_estagio=[SYNTHETIC_LOAD_BLOCKS] * SYNTHETIC_STAGES,
        numero_uhes_tempo_viagem=len(SYNTHETIC_TRAVEL_TIME_PLANT_CODES),
        maximo_lag_tempo_viagem=SYNTHETIC_MAX_TRAVEL_TIME_LAG,
        lag_tempo_viagem_por_uhe=list(SYNTHETIC_TRAVEL_TIME_STAGE_LAGS),
        codigos_uhes=list(SYNTHETIC_PLANT_CODES),
        registro_ultimo_corte_no=synthetic_last_cut_table(),
        # Topologia gravada como int32 e lida como float32, tal como o idecomp faz:
        # a usina 1 deflui para a posicao 2 e a usina 2 nao tem jusante.
        codigos_uhes_jusante=list(np.array([2, 0], dtype=np.int32).view(np.float32)),
        data=FakeSectionFile(
            FakeSection(
                {
                    "dados_tempo_viagem": synthetic_travel_time_payload(),
                    "dados_gnl": synthetic_gnl_payload(),
                }
            )
        ),
    )


@pytest.fixture
def fake_mapcut_sem_tempo_viagem() -> FakeMapcut:
    """Mapcut sintetico de um caso sem nenhuma UHE com tempo de viagem.

    Reproduz as duas armadilhas do idecomp nesse caso: a propriedade
    `codigos_uhes_tempo_viagem` levanta `KeyError` (veja `FakeMapcut`) e
    `lag_tempo_viagem_por_uhe` devolve valores espurios em vez de lista vazia.
    O payload dos registros 7 e 8 esta vazio, porque esses registros nao existem.
    """
    return FakeMapcut(
        tamanho_corte=SYNTHETIC_RECORD_SIZE,
        numero_iteracoes=SYNTHETIC_CUTS_PER_NODE,
        numero_cortes=SYNTHETIC_CUTS_PER_NODE * 2,
        numero_estagios=SYNTHETIC_STAGES,
        numero_submercados=SYNTHETIC_SUBMARKET_COUNT,
        patamares_por_estagio=[SYNTHETIC_LOAD_BLOCKS] * SYNTHETIC_STAGES,
        numero_uhes_tempo_viagem=0,
        maximo_lag_tempo_viagem=0,
        lag_tempo_viagem_por_uhe=list(SYNTHETIC_BOGUS_TRAVEL_TIME_LAGS),
        codigos_uhes=list(SYNTHETIC_PLANT_CODES),
        registro_ultimo_corte_no=synthetic_last_cut_table(),
        codigos_uhes_jusante=list(np.array([2, 0], dtype=np.int32).view(np.float32)),
        data=FakeSectionFile(
            FakeSection(
                {
                    "dados_tempo_viagem": [],
                    "dados_gnl": synthetic_gnl_payload(),
                }
            )
        ),
    )


@pytest.fixture
def fake_mapcut_gnl_maior() -> FakeMapcut:
    """Mapcut sintetico em que o passo errado do idecomp sai do payload.

    E a condicao do deck rv4: poucos estagios e mais de um submercado com GNL.
    Nesse caso `Mapcut.dados_gnl` levanta `IndexError`, e a leitura tem de
    continuar funcionando porque nao depende dela.
    """
    return FakeMapcut(
        tamanho_corte=SYNTHETIC_RECORD_SIZE_GNL_MAIOR,
        numero_iteracoes=SYNTHETIC_CUTS_PER_NODE,
        numero_cortes=SYNTHETIC_CUTS_PER_NODE * 2,
        numero_estagios=SYNTHETIC_STAGES,
        numero_submercados=SYNTHETIC_SUBMARKET_COUNT,
        patamares_por_estagio=[SYNTHETIC_LOAD_BLOCKS_MAIOR] * SYNTHETIC_STAGES,
        numero_uhes_tempo_viagem=len(SYNTHETIC_TRAVEL_TIME_PLANT_CODES),
        maximo_lag_tempo_viagem=SYNTHETIC_MAX_TRAVEL_TIME_LAG,
        lag_tempo_viagem_por_uhe=list(SYNTHETIC_TRAVEL_TIME_STAGE_LAGS),
        codigos_uhes=list(SYNTHETIC_PLANT_CODES),
        registro_ultimo_corte_no=synthetic_last_cut_table(),
        codigos_uhes_jusante=list(np.array([2, 0], dtype=np.int32).view(np.float32)),
        data=FakeSectionFile(
            FakeSection(
                {
                    "dados_tempo_viagem": synthetic_travel_time_payload(),
                    "dados_gnl": synthetic_gnl_payload(
                        submarket_codes=SYNTHETIC_GNL_SUBMARKET_CODES_MAIOR,
                        load_blocks=SYNTHETIC_LOAD_BLOCKS_MAIOR,
                    ),
                }
            )
        ),
    )


@pytest.fixture
def fake_mapcut_sem_gnl() -> FakeMapcut:
    """Mapcut sintetico de um caso sem nenhuma UTE a GNL.

    O DECOMP grava o registro 9 de todo estagio mesmo sem UTE a GNL: o registro
    fica com `numero_utes_gnl = 0` e passo 1. E o encoding mais provavel do caso
    real, e por isso o preferido aqui em vez de payload inteiramente vazio - as
    duas formas sao cobertas nos testes.
    """
    return FakeMapcut(
        tamanho_corte=SYNTHETIC_RECORD_SIZE,
        numero_iteracoes=SYNTHETIC_CUTS_PER_NODE,
        numero_cortes=SYNTHETIC_CUTS_PER_NODE * 2,
        numero_estagios=SYNTHETIC_STAGES,
        numero_submercados=SYNTHETIC_SUBMARKET_COUNT,
        patamares_por_estagio=[SYNTHETIC_LOAD_BLOCKS] * SYNTHETIC_STAGES,
        numero_uhes_tempo_viagem=len(SYNTHETIC_TRAVEL_TIME_PLANT_CODES),
        maximo_lag_tempo_viagem=SYNTHETIC_MAX_TRAVEL_TIME_LAG,
        lag_tempo_viagem_por_uhe=list(SYNTHETIC_TRAVEL_TIME_STAGE_LAGS),
        codigos_uhes=list(SYNTHETIC_PLANT_CODES),
        registro_ultimo_corte_no=synthetic_last_cut_table(),
        codigos_uhes_jusante=list(np.array([2, 0], dtype=np.int32).view(np.float32)),
        data=FakeSectionFile(
            FakeSection(
                {
                    "dados_tempo_viagem": synthetic_travel_time_payload(),
                    "dados_gnl": [0] * SYNTHETIC_STAGES,
                }
            )
        ),
    )


def synthetic_coefficient(record_index: int, coefficient_index: int) -> float:
    """Valor deterministico de um coeficiente, unico por (registro, coeficiente).

    Codificar a posicao no proprio valor permite afirmar, no teste, que cada
    coeficiente veio do registro certo - o que nao seria possivel com valores
    aleatorios ou repetidos.
    """
    return record_index * 100.0 + coefficient_index


def write_synthetic_cortdeco(
    path: Path,
    padding_byte: int = 0,
    coefficient_count: int = SYNTHETIC_COEFFICIENT_COUNT,
    record_size: int = SYNTHETIC_RECORD_SIZE,
) -> None:
    """Grava um cortdeco sintetico de 7 registros.

    Cada registro contem o indice do proximo corte do mesmo no (`int32`), os
    coeficientes (`float64`) e o preenchimento nulo ate completar o registro.

    Args:
        path: destino do arquivo.
        padding_byte: valor usado no preenchimento. Diferente de zero, serve para
            testar a deteccao de coeficientes truncados.
        coefficient_count: quantos coeficientes gravar por registro. O padrao e o
            do caso sintetico com tempo de viagem; o caso sem tempo de viagem usa
            um registro mais curto, sem o bloco de defluencia passada.
        record_size: tamanho de cada registro, em bytes. Casos com mais
            coeficientes precisam de registro maior, senao os coeficientes nao
            cabem no registro.
    """
    next_record = {0: 0}  # registro 0 nao existe; mantem o dicionario homogeneo
    for chain in SYNTHETIC_CHAINS.values():
        for position, record in enumerate(chain):
            has_next = position + 1 < len(chain)
            next_record[record] = chain[position + 1] if has_next else 0

    used = 4 + 8 * coefficient_count
    padding = bytes([padding_byte]) * (record_size - used)
    with path.open("wb") as handle:
        for record in range(1, SYNTHETIC_TOTAL_RECORDS + 1):
            handle.write(np.int32(next_record[record]).tobytes())
            handle.write(
                np.array(
                    [
                        synthetic_coefficient(record, index)
                        for index in range(coefficient_count)
                    ],
                    dtype=np.float64,
                ).tobytes()
            )
            handle.write(padding)


@pytest.fixture
def synthetic_cortdeco(tmp_path: Path) -> Path:
    """Cortdeco sintetico valido, com preenchimento nulo."""
    path = tmp_path / "cortdeco.sintetico"
    write_synthetic_cortdeco(path)
    return path


@pytest.fixture
def synthetic_cortdeco_sem_tempo_viagem(tmp_path: Path) -> Path:
    """Cortdeco sintetico do caso sem tempo de viagem (registro mais curto)."""
    path = tmp_path / "cortdeco.sintetico_sem_tempo_viagem"
    write_synthetic_cortdeco(
        path, coefficient_count=SYNTHETIC_COEFFICIENT_COUNT_WITHOUT_TRAVEL_TIME
    )
    return path


@pytest.fixture
def synthetic_cortdeco_sem_gnl(tmp_path: Path) -> Path:
    """Cortdeco sintetico do caso sem GNL (registro mais curto ainda)."""
    path = tmp_path / "cortdeco.sintetico_sem_gnl"
    write_synthetic_cortdeco(
        path, coefficient_count=SYNTHETIC_COEFFICIENT_COUNT_WITHOUT_GNL
    )
    return path


@pytest.fixture
def synthetic_cortdeco_gnl_maior(tmp_path: Path) -> Path:
    """Cortdeco do caso de GNL com dois submercados e tres patamares."""
    path = tmp_path / "cortdeco.sintetico_gnl_maior"
    write_synthetic_cortdeco(
        path,
        coefficient_count=SYNTHETIC_COEFFICIENT_COUNT_GNL_MAIOR,
        record_size=SYNTHETIC_RECORD_SIZE_GNL_MAIOR,
    )
    return path


@pytest.fixture
def quiet_logger() -> logging.Logger:
    """Logger que nao imprime nada, para nao poluir a saida dos testes."""
    logger = logging.getLogger("teste_leitor_mapcut_cortdeco")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


@pytest.fixture
def caplog_logger(caplog: pytest.LogCaptureFixture) -> logging.Logger:
    """Logger cujas mensagens ficam disponiveis em `caplog`."""
    logger = logging.getLogger("teste_leitor_capturado")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    caplog.set_level(logging.DEBUG, logger=logger.name)
    return logger


def valid_settings_dict() -> dict[str, Any]:
    """Configuracao minima e valida.

    Todos os caminhos sao relativos, portanto ela se ancora sozinha no
    diretorio onde o settings.json for gravado.
    """
    return {
        "paths": {
            "input_directory": ".",
            "mapcut_file": "mapcut.rv0",
            "cortdeco_file": "cortdeco.rv0",
            "output_directory": "output",
        },
        "logging": {
            "level": "INFO",
            "directory": "logs",
            "filename": "leitor.log",
            "max_bytes": 1024,
            "backup_count": 1,
            "use_rich": False,
        },
        "export": {
            "separator": ",",
            "decimal": ".",
            "float_format": None,
            "encoding": "utf-8",
            "write_mapcut": True,
            "write_cortdeco_wide": True,
            "write_cortdeco_tidy": True,
        },
        "cortdeco": {
            "cuts_per_node_source": "numero_iteracoes",
            "cuts_per_node_override": None,
        },
    }


@pytest.fixture
def settings_file(tmp_path: Path) -> Path:
    """Grava um settings.json valido em um diretorio temporario."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(valid_settings_dict(), indent=2), encoding="utf-8")
    return path


def _real_deck_path(attribute: str) -> Path:
    """Localiza um binario de entrada do caso presente no diretorio do projeto.

    O nome do arquivo muda a cada revisao do caso (mapcut.rv0, mapcut.rv4, ...),
    entao fixa-lo aqui faz a suite de integracao inteira ser ignorada em silencio
    na proxima troca de deck - foi o que aconteceu. A fonte da verdade e o
    proprio `settings.json` do projeto, resolvido pelo codigo de producao: assim
    o teste segue o deck configurado e ainda exercita `load_settings`.
    """
    settings_path = PROJECT_ROOT / "settings.json"
    if not settings_path.is_file():
        pytest.skip(f"settings.json ausente: {settings_path}")
    settings = load_settings(settings_path)
    path: Path = getattr(settings.paths, attribute)
    if not path.is_file():
        pytest.skip(f"binario real ausente: {path}")
    return path


@pytest.fixture(scope="session")
def real_mapcut_path() -> Path:
    """Caminho do mapcut real; ignora o teste quando ele nao esta no repositorio."""
    return _real_deck_path("mapcut_path")


@pytest.fixture(scope="session")
def real_cortdeco_path() -> Path:
    """Caminho do cortdeco real; ignora o teste quando ele nao esta no repositorio."""
    return _real_deck_path("cortdeco_path")
