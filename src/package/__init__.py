from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("research-package")  # Must match the exact name in pyproject.toml
except PackageNotFoundError:
    __version__ = "unknown"