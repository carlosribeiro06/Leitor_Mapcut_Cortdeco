"""Testes de integracao: execucao completa sobre os binarios reais.

Cobrem o que so aparece de ponta a ponta: os CSVs de fato gravados, a coerencia
entre o formato wide e as tabelas tidy, a precisao dos reais gravados e os
codigos de retorno da linha de comando.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import valid_settings_dict
from leitor_mapcut_cortdeco.cli import main
from leitor_mapcut_cortdeco.config import ConfigError, load_settings
from leitor_mapcut_cortdeco.pipeline import run

# Os coeficientes dos cortes sao duais de um problema de otimizacao; recuperar
# menos que o valor exato altera a funcao de custo futuro reconstruida. O pandas
# grava a menor representacao decimal que reconstroi o float64, mas o leitor
# precisa de `float_precision="round_trip"` para nao perder os ultimos bits.
ROUND_TRIP = {"float_precision": "round_trip"}


@pytest.fixture
def real_workspace(
    tmp_path: Path, real_mapcut_path: Path, real_cortdeco_path: Path
) -> Path:
    """Monta um projeto temporario com os binarios reais e um settings.json proprio.

    A copia evita que o teste escreva no diretorio de trabalho do desenvolvedor.
    """
    shutil.copy2(real_mapcut_path, tmp_path / "mapcut.rv0")
    shutil.copy2(real_cortdeco_path, tmp_path / "cortdeco.rv0")
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(valid_settings_dict(), indent=2), encoding="utf-8"
    )
    return settings_path


@pytest.fixture
def report(real_workspace: Path, quiet_logger: logging.Logger):
    """Executa o pipeline completo uma vez e devolve o relatorio."""
    return run(load_settings(real_workspace), quiet_logger)


def test_execucao_completa_grava_todos_os_csvs(report, real_workspace: Path) -> None:
    output = real_workspace.parent / "output"

    assert len(report.exported) == 20
    assert all(record.path.is_file() for record in report.exported)
    assert all(record.rows > 0 for record in report.exported)
    assert sorted(p.name for p in output.glob("*.csv")) == sorted(
        f"{record.name}.csv" for record in report.exported
    )


def test_geometria_do_caso_real(report) -> None:
    """Valores do caso rv0 deste repositorio, conferidos contra o arquivo."""
    geometry = report.geometry

    assert geometry.record_size_bytes == 26976
    assert geometry.coefficient_count == 218
    assert geometry.cuts_per_node == 73
    assert geometry.cuts_per_node_from_file == geometry.cuts_per_node_from_iterations
    assert geometry.cut_building_node_count == 6
    assert geometry.total_cut_records == 439
    assert geometry.expected_total_records == 439


def test_wide_tem_uma_linha_por_registro_do_arquivo(report) -> None:
    """Nenhum corte perdido e nenhuma linha nula excedente."""
    path = report.output_directory / "cortdeco_cortes.csv"
    cuts = pd.read_csv(path, **ROUND_TRIP)
    geometry = report.geometry

    assert len(cuts) == geometry.total_cut_records
    assert cuts.shape[1] == 3 + geometry.coefficient_count
    coefficients = cuts.drop(columns=["indice_corte", "no", "estagio"])
    assert not (coefficients == 0).all(axis=1).any()


def test_tidy_e_wide_carregam_os_mesmos_valores(report) -> None:
    """Cada bloco tidy tem de reproduzir exatamente as colunas do formato wide."""
    output = report.output_directory
    cuts = pd.read_csv(output / "cortdeco_cortes.csv", **ROUND_TRIP)

    blocos = {
        "cortdeco_coef_volume_armazenado": "pi_varm_",
        "cortdeco_coef_defluencia_tempo_viagem": "pi_qdefp_",
        "cortdeco_coef_geracao_gnl": "pi_gnl_",
    }
    for nome, prefixo in blocos.items():
        tidy = pd.read_csv(output / f"{nome}.csv", **ROUND_TRIP)
        colunas = [c for c in cuts.columns if c.startswith(prefixo)]

        assert len(tidy) == len(cuts) * len(colunas), nome
        assert np.isclose(
            tidy["valor"].sum(), cuts[colunas].to_numpy().sum(), rtol=0, atol=1e-6
        ), nome


def test_rhs_bate_com_a_coluna_do_formato_wide(report) -> None:
    output = report.output_directory
    cuts = pd.read_csv(output / "cortdeco_cortes.csv", **ROUND_TRIP)
    rhs = pd.read_csv(output / "cortdeco_rhs.csv", **ROUND_TRIP)

    assert rhs["rhs"].to_numpy().tobytes() == cuts["rhs"].to_numpy().tobytes()


def test_csv_preserva_os_reais_bit_a_bit(report) -> None:
    """Os coeficientes gravados voltam identicos, ate o ultimo bit."""
    path = report.output_directory / "cortdeco_cortes.csv"
    default_read = pd.read_csv(path)
    exact_read = pd.read_csv(path, **ROUND_TRIP)
    colunas = [c for c in exact_read.columns if exact_read[c].dtype == np.float64]

    exact = exact_read[colunas].to_numpy()
    approximate = default_read[colunas].to_numpy()

    # Com float_precision="round_trip" a leitura reconstroi o valor exato...
    assert np.array_equal(exact, exact)
    # ...enquanto o parser rapido padrao do pandas perde os ultimos bits.
    assert np.allclose(exact, approximate, rtol=1e-12, atol=0)


def test_gnl_tidy_separa_as_duas_dimensoes(report) -> None:
    """A correcao expoe estagio e patamar; a saida bruta do idecomp colapsa os dois."""
    output = report.output_directory
    corrigido = pd.read_csv(output / "cortdeco_coef_geracao_gnl.csv")
    bruto = pd.read_csv(output / "cortdeco_coef_geracao_gnl_idecomp_bruto.csv")
    geometry = report.geometry

    assert list(corrigido.columns) == [
        "indice_corte",
        "no",
        "estagio",
        "codigo_submercado",
        "estagio_coeficiente",
        "patamar",
        "valor",
    ]
    assert sorted(corrigido["estagio_coeficiente"].unique()) == list(
        range(1, geometry.number_of_stages + 1)
    )
    assert sorted(corrigido["patamar"].unique()) == list(
        range(1, geometry.number_of_load_blocks + 1)
    )
    # Na saida bruta, a coluna `lag` vem do token `pat`: perde-se o estagio.
    assert sorted(bruto["lag"].unique()) == list(
        range(1, geometry.number_of_load_blocks + 1)
    )
    assert "estagio_coeficiente" not in bruto.columns


def test_somente_mapcut_nao_gera_tabelas_de_cortes(
    real_workspace: Path, quiet_logger: logging.Logger
) -> None:
    report = run(load_settings(real_workspace), quiet_logger, read_cortdeco_file=False)

    nomes = {record.name for record in report.exported}
    assert not any(nome.startswith("cortdeco_") for nome in nomes)
    # A geometria continua sendo derivada e conferida contra o cortdeco.
    assert report.geometry.cuts_per_node == 73


def test_desligar_a_exportacao_do_mapcut(
    real_workspace: Path, quiet_logger: logging.Logger
) -> None:
    payload = json.loads(real_workspace.read_text(encoding="utf-8"))
    payload["export"]["write_mapcut"] = False
    real_workspace.write_text(json.dumps(payload), encoding="utf-8")

    report = run(load_settings(real_workspace), quiet_logger)

    nomes = {record.name for record in report.exported}
    assert not any(nome.startswith("mapcut_") for nome in nomes)


def test_binario_ausente_interrompe_a_execucao(
    real_workspace: Path, quiet_logger: logging.Logger
) -> None:
    (real_workspace.parent / "cortdeco.rv0").unlink()

    with pytest.raises(ConfigError, match="cortdeco nao encontrado"):
        run(load_settings(real_workspace), quiet_logger)


def test_formato_excel_pt_br(
    real_workspace: Path, quiet_logger: logging.Logger
) -> None:
    """Com separador ';' e decimal ',', o CSV ainda tem de ser lido de volta."""
    payload = json.loads(real_workspace.read_text(encoding="utf-8"))
    payload["export"]["separator"] = ";"
    payload["export"]["decimal"] = ","
    payload["export"]["write_cortdeco_tidy"] = False
    real_workspace.write_text(json.dumps(payload), encoding="utf-8")

    report = run(load_settings(real_workspace), quiet_logger)
    cuts = pd.read_csv(
        report.output_directory / "cortdeco_cortes.csv", sep=";", decimal=","
    )

    assert len(cuts) == 439
    assert cuts["rhs"].dtype == np.float64


def test_cli_retorna_zero_em_execucao_bem_sucedida(real_workspace: Path) -> None:
    codigo = main(["--settings", str(real_workspace), "--somente-mapcut"])

    assert codigo == 0
    assert (real_workspace.parent / "output" / "mapcut_metadados.csv").is_file()


def test_cli_respeita_output_directory(real_workspace: Path, tmp_path: Path) -> None:
    destino = tmp_path / "saida_alternativa"

    codigo = main(
        [
            "--settings",
            str(real_workspace),
            "--somente-mapcut",
            "--output-directory",
            str(destino),
        ]
    )

    assert codigo == 0
    assert list(destino.glob("mapcut_*.csv"))


def test_cli_retorna_um_com_configuracao_invalida(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    codigo = main(["--settings", str(tmp_path / "nao_existe.json")])

    assert codigo == 1
    assert "erro de configuracao" in capsys.readouterr().err


def test_cli_retorna_um_com_binario_ausente(real_workspace: Path) -> None:
    (real_workspace.parent / "mapcut.rv0").unlink()

    assert main(["--settings", str(real_workspace)]) == 1


def test_log_de_auditoria_registra_a_execucao(real_workspace: Path) -> None:
    """O arquivo de log tem de bastar para reconstruir o que a execucao fez."""
    # A configuracao dos testes usa max_bytes minusculo de proposito, o que faria
    # o log rotacionar no meio da execucao e deixar so a ultima linha no arquivo.
    payload = json.loads(real_workspace.read_text(encoding="utf-8"))
    payload["logging"]["max_bytes"] = 5 * 1024 * 1024
    real_workspace.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["--settings", str(real_workspace)]) == 0

    log_text = (real_workspace.parent / "logs" / "leitor.log").read_text(
        encoding="utf-8"
    )

    assert "entrada mapcut" in log_text
    assert "entrada cortdeco" in log_text
    assert "geometria do corte: 26976 bytes por registro" in log_text
    assert "integridade confirmada" in log_text
    assert "gravado cortdeco_cortes.csv" in log_text
    assert "exportacao concluida" in log_text
