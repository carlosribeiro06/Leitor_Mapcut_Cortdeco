"""Testes da carga e validacao do settings.json.

O foco e a validacao *fail fast*: cada teste confirma que um defeito especifico
da configuracao produz `ConfigError` com uma mensagem que aponta a chave errada,
em vez de deixar a execucao seguir com um valor implicito.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conftest import valid_settings_dict
from leitor_mapcut_cortdeco.config import ConfigError, load_settings


def _write(tmp_path: Path, payload: dict[str, Any]) -> Path:
    """Grava um settings.json em `tmp_path` e devolve seu caminho."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_carrega_configuracao_valida(settings_file: Path, tmp_path: Path) -> None:
    settings = load_settings(settings_file)

    assert settings.paths.mapcut_path == tmp_path / "mapcut.rv0"
    assert settings.paths.cortdeco_path == tmp_path / "cortdeco.rv0"
    assert settings.paths.output_directory == tmp_path / "output"
    assert settings.logging.log_path == tmp_path / "logs" / "leitor.log"
    assert settings.export.float_format is None
    assert settings.export.write_cortdeco is True
    assert settings.cortdeco.cuts_per_node_source == "numero_iteracoes"
    assert settings.settings_path == settings_file


def test_caminhos_relativos_sao_ancorados_no_settings(tmp_path: Path) -> None:
    """Um caminho relativo segue o settings.json, nao o diretorio de trabalho."""
    nested = tmp_path / "config"
    nested.mkdir()
    payload = valid_settings_dict()
    payload["paths"]["input_directory"] = "../dados"
    path = _write(nested, payload)

    settings = load_settings(path)

    assert settings.paths.input_directory == tmp_path / "dados"


def test_caminho_absoluto_e_respeitado(tmp_path: Path) -> None:
    payload = valid_settings_dict()
    absolute = (tmp_path / "entrada_absoluta").resolve()
    payload["paths"]["input_directory"] = str(absolute)

    settings = load_settings(_write(tmp_path, payload))

    assert settings.paths.input_directory == absolute


def test_arquivo_inexistente(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="nao encontrado"):
        load_settings(tmp_path / "ausente.json")


def test_json_malformado(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text('{"paths": ', encoding="utf-8")

    with pytest.raises(ConfigError, match="JSON malformado"):
        load_settings(path)


def test_secao_ausente(tmp_path: Path) -> None:
    payload = valid_settings_dict()
    del payload["export"]

    with pytest.raises(ConfigError, match="secao obrigatoria ausente.*export"):
        load_settings(_write(tmp_path, payload))


def test_chave_ausente(tmp_path: Path) -> None:
    payload = valid_settings_dict()
    del payload["paths"]["mapcut_file"]

    with pytest.raises(ConfigError, match="paths.mapcut_file"):
        load_settings(_write(tmp_path, payload))


def test_booleano_nao_serve_como_inteiro(tmp_path: Path) -> None:
    """`isinstance(True, int)` e verdadeiro; a validacao nao deve cair nisso."""
    payload = valid_settings_dict()
    payload["logging"]["max_bytes"] = True

    with pytest.raises(ConfigError, match="nao booleano"):
        load_settings(_write(tmp_path, payload))


def test_nivel_de_log_invalido(tmp_path: Path) -> None:
    payload = valid_settings_dict()
    payload["logging"]["level"] = "VERBOSO"

    with pytest.raises(ConfigError, match="logging.level invalido"):
        load_settings(_write(tmp_path, payload))


def test_nivel_de_log_aceita_minusculas(tmp_path: Path) -> None:
    payload = valid_settings_dict()
    payload["logging"]["level"] = "debug"

    assert load_settings(_write(tmp_path, payload)).logging.level == "DEBUG"


def test_max_bytes_deve_ser_positivo(tmp_path: Path) -> None:
    payload = valid_settings_dict()
    payload["logging"]["max_bytes"] = 0

    with pytest.raises(ConfigError, match="max_bytes deve ser positivo"):
        load_settings(_write(tmp_path, payload))


def test_nome_de_arquivo_com_diretorio_e_rejeitado(tmp_path: Path) -> None:
    """O diretorio pertence a `input_directory`; misturar os dois esconde erros."""
    payload = valid_settings_dict()
    payload["paths"]["mapcut_file"] = "subdir/mapcut.rv0"

    with pytest.raises(ConfigError, match="sem diretorio"):
        load_settings(_write(tmp_path, payload))


def test_separador_com_mais_de_um_caractere(tmp_path: Path) -> None:
    payload = valid_settings_dict()
    payload["export"]["separator"] = ";;"

    with pytest.raises(ConfigError, match="separator deve ter exatamente"):
        load_settings(_write(tmp_path, payload))


def test_separador_igual_ao_decimal_torna_o_csv_ambiguo(tmp_path: Path) -> None:
    payload = valid_settings_dict()
    payload["export"]["separator"] = ","
    payload["export"]["decimal"] = ","

    with pytest.raises(ConfigError, match="ambiguo"):
        load_settings(_write(tmp_path, payload))


def test_formato_excel_pt_br_e_valido(tmp_path: Path) -> None:
    payload = valid_settings_dict()
    payload["export"]["separator"] = ";"
    payload["export"]["decimal"] = ","

    settings = load_settings(_write(tmp_path, payload))

    assert (settings.export.separator, settings.export.decimal) == (";", ",")


def test_float_format_deve_ser_string_ou_nulo(tmp_path: Path) -> None:
    payload = valid_settings_dict()
    payload["export"]["float_format"] = 10

    with pytest.raises(ConfigError, match="float_format deve ser string ou null"):
        load_settings(_write(tmp_path, payload))


def test_fonte_de_cortes_por_no_invalida(tmp_path: Path) -> None:
    payload = valid_settings_dict()
    payload["cortdeco"]["cuts_per_node_source"] = "chute"

    with pytest.raises(ConfigError, match="cuts_per_node_source invalido"):
        load_settings(_write(tmp_path, payload))


@pytest.mark.parametrize("valor", [0, -5])
def test_override_de_cortes_por_no_deve_ser_positivo(
    tmp_path: Path, valor: int
) -> None:
    payload = valid_settings_dict()
    payload["cortdeco"]["cuts_per_node_override"] = valor

    with pytest.raises(ConfigError, match="override deve ser positivo"):
        load_settings(_write(tmp_path, payload))


def test_override_de_cortes_por_no_aceito(tmp_path: Path) -> None:
    payload = valid_settings_dict()
    payload["cortdeco"]["cuts_per_node_override"] = 42

    settings = load_settings(_write(tmp_path, payload))

    assert settings.cortdeco.cuts_per_node_override == 42


def test_chaves_de_comentario_sao_ignoradas(tmp_path: Path) -> None:
    """As chaves `_comentario` do settings.json distribuido nao devem incomodar."""
    payload = valid_settings_dict()
    payload["_comentario"] = ["texto livre"]
    payload["paths"]["_comentario"] = "outro texto"

    assert load_settings(_write(tmp_path, payload)) is not None
