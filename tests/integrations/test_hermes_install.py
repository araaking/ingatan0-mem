"""The block `ingatan setup --client hermes` prints for `$HERMES_HOME/config.yaml`."""

from __future__ import annotations

from ingatan.integrations.hermes.install import mcp_block


def test_block_names_business_and_agent_and_no_data_folder_by_default() -> None:
    block = mcp_block("tokoku", "hermes")
    assert 'INGATAN_BUSINESS: "tokoku"' in block
    assert 'INGATAN_AGENT: "hermes"' in block
    assert "INGATAN_HOME" not in block


def test_block_names_the_data_folder_when_one_is_set() -> None:
    block = mcp_block("tokoku", "hermes", home="/srv/data/ingatan")
    assert "      INGATAN_HOME: '/srv/data/ingatan'\n" in block
    assert block.endswith("    enabled: true")


def test_the_data_folder_is_written_as_a_literal_yaml_string() -> None:
    # Single quotes: a backslash stays a backslash, and a quote is doubled.
    windows = mcp_block("tokoku", "hermes", home="C:\\Users\\budi\\ingatan data")
    assert "INGATAN_HOME: 'C:\\Users\\budi\\ingatan data'" in windows
    quoted = mcp_block("tokoku", "hermes", home="/srv/budi's data")
    assert "INGATAN_HOME: '/srv/budi''s data'" in quoted
