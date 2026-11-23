"""What a Hermes user adds to `$HERMES_HOME/config.yaml` to run the Ingatan MCP server.

Ingatan does not edit that file: it is YAML, no YAML parser is a dependency, and a merge
made without one could damage another application's configuration. `ingatan setup` prints
this block instead.
"""

from __future__ import annotations

__all__ = ["mcp_block"]


def mcp_block(business: str, agent: str, home: str | None = None) -> str:
    """The `mcp_servers` block for one business and agent (see docs/setup-hermes.md).

    Hermes passes an MCP server only the `env` listed here, so the scope must be in the
    block. The names are quoted: a valid name such as `no` or `123` is not a YAML string
    otherwise. `home` is the data folder when it is not the default one (`INGATAN_HOME` is
    set where this runs): without it in the block, the server Hermes starts would open the
    default folder, another store than the one the command line works on. A path goes in
    single quotes, where a backslash is a backslash and only a quote needs doubling.
    """
    data_folder = ""
    if home is not None:
        quoted = home.replace("'", "''")
        data_folder = f"      INGATAN_HOME: '{quoted}'\n"
    return (
        "mcp_servers:\n"
        "  ingatan:\n"
        "    command: ingatan\n"
        '    args: ["serve"]\n'
        "    env:\n"
        f'      INGATAN_BUSINESS: "{business}"\n'
        f'      INGATAN_AGENT: "{agent}"\n'
        f"{data_folder}"
        "    enabled: true"
    )
