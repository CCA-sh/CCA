"""Minimal FastMCP stdio server used by tests/test_mcp.py."""
from mcp.server.fastmcp import FastMCP

server = FastMCP("fixture")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.resource("fixture://greeting")
def greeting() -> str:
    """A static greeting resource."""
    return "hello from fixture"


if __name__ == "__main__":
    server.run()
