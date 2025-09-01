import click


@click.command()
@click.option(
    "--config",
    "config_path",
    default="mcp_agent_config.yaml",
    help="(deprecated) path to MCP agent config",
)
@click.option("--host", default="0.0.0.0")
@click.option("--port", default=8001, type=int)
def main_cli(config_path: str, host: str, port: int):
    click.echo("eval_protocol.mcp_agent.main is deprecated and disabled.")


if __name__ == "__main__":
    main_cli()
