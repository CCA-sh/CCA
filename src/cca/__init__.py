"""cca — a complete code agent CLI with pluggable LLM backends."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cca-cli")
except PackageNotFoundError:
    # A raw source checkout may be imported before the package is installed.
    __version__ = "0+unknown"
