"""DROMA MCP - Model Context Protocol server for drug-omics association analysis.

Version 0.2.0 - FastMCP 2.13+ Compatible
"""

__version__ = "0.2.0"
__author__ = "DROMA Team"
__email__ = "contact@droma.org"

from .server import droma_mcp

__all__ = ["droma_mcp", "__version__"] 