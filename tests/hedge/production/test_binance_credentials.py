from pathlib import Path

import pytest

from freqtrade.hedge.production.binance_credentials import (
    CredentialFileError,
    load_binance_credentials,
)


def test_two_line_credential_file_is_loaded_without_repr_leak(tmp_path: Path) -> None:
    source = tmp_path / "credentials.txt"
    source.write_text("api-key-value\napi-secret-value\n", encoding="utf-8")
    credentials = load_binance_credentials(source)
    assert credentials.environment() == {
        "FREQTRADE__EXCHANGE__KEY": "api-key-value",
        "FREQTRADE__EXCHANGE__SECRET": "api-secret-value",
    }
    assert "api-secret-value" not in repr(credentials)


def test_labelled_and_json_credentials_are_supported(tmp_path: Path) -> None:
    labelled = tmp_path / "labelled.txt"
    labelled.write_text("api_key=key\nsecret: secret\n", encoding="utf-8")
    assert load_binance_credentials(labelled).environment()["FREQTRADE__EXCHANGE__KEY"] == "key"
    structured = tmp_path / "credentials.json"
    structured.write_text('{"apiKey":"key","apiSecret":"secret"}', encoding="utf-8")
    assert (
        load_binance_credentials(structured).environment()["FREQTRADE__EXCHANGE__SECRET"]
        == "secret"
    )


def test_two_line_secret_may_include_label_delimiters(tmp_path: Path) -> None:
    source = tmp_path / "credentials.txt"
    source.write_text(
        "key-material=with-delimiter\nsecret-material:with-delimiter\n",
        encoding="utf-8",
    )
    credentials = load_binance_credentials(source)
    environment = credentials.environment()
    assert environment["FREQTRADE__EXCHANGE__KEY"] == "key-material=with-delimiter"
    assert environment["FREQTRADE__EXCHANGE__SECRET"] == "secret-material:with-delimiter"


def test_conflicting_labels_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "credentials.txt"
    source.write_text("key=first\napi_key=second\nsecret=value\n", encoding="utf-8")
    with pytest.raises(CredentialFileError, match="conflicting"):
        load_binance_credentials(source)


def test_invalid_credential_file_never_includes_value_in_error(tmp_path: Path) -> None:
    source = tmp_path / "invalid.txt"
    source.write_text("super-secret-value\n", encoding="utf-8")
    with pytest.raises(CredentialFileError) as error:
        load_binance_credentials(source)
    assert "super-secret-value" not in str(error.value)
