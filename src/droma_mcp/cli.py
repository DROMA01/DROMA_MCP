"""Command-line interface for DROMA MCP server."""

import os
import asyncio
import json
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

import typer
from typing_extensions import Annotated

from . import __version__, __author__


class Module(str, Enum):
    """Available server modules."""
    ALL = "all"
    DATA_LOADING = "data_loading"
    DATABASE_QUERY = "database_query"
    DATASET_MANAGEMENT = "dataset_management"
    VISUALIZATION = "visualization"
    ANALYSIS = "analysis"


class Transport(str, Enum):
    """Available transport protocols."""
    STDIO = "stdio"
    HTTP = "http"  # FastMCP 2.13+ uses "http" for streamable HTTP
    SSE = "sse"


class DromaMCPCLI:
    """DROMA MCP Server CLI manager."""
    
    def __init__(self):
        self.app = typer.Typer(
            name="droma-mcp",
            help="DROMA MCP Server - Model Context Protocol server for drug-omics association analysis",
            add_completion=False
        )
        self._setup_commands()
    
    def _setup_commands(self):
        """Setup all CLI commands."""
        self.app.command(name="run")(self.run)
        self.app.command(name="test")(self.test_connection)
        self.app.command(name="info")(self.info)
        self.app.command(name="export-config")(self.export_config)
        self.app.command(name="validate")(self.validate_setup)
        self.app.command(name="benchmark")(self.benchmark)
    
    def _setup_environment(
        self,
        module: Module,
        db_path: Optional[str] = None,
        r_libs: Optional[str] = None,
        verbose: bool = False
    ) -> None:
        """Setup environment variables for the server."""
        os.environ['DROMA_MCP_MODULE'] = module.value
        
        if db_path:
            os.environ['DROMA_DB_PATH'] = db_path
        
        if r_libs:
            os.environ['R_LIBS'] = r_libs
        
        if verbose:
            os.environ['DROMA_MCP_VERBOSE'] = "1"
    
    def _validate_dependencies(self) -> Dict[str, bool]:
        """Validate all dependencies."""
        results = {}
        
        # Test Python dependencies
        try:
            import pandas as pd
            import numpy as np
            import fastmcp
            results['python_deps'] = True
        except ImportError:
            results['python_deps'] = False
        
        # Test R integration
        try:
            import rpy2.robjects as robjects
            from rpy2.robjects import pandas2ri
            r = robjects.r
            # Test pandas2ri converter availability (no need to activate globally)
            _ = pandas2ri.converter
            results['r_integration'] = True
        except ImportError:
            results['r_integration'] = False
        
        # Test DROMA packages
        results['droma_packages'] = False
        if results['r_integration']:
            try:
                import rpy2.robjects as robjects
                r = robjects.r
                r('library(DROMA.Set)')
                r('library(DROMA.R)')
                results['droma_packages'] = True
            except Exception:
                results['droma_packages'] = False
        
        return results
    
    def run(
        self,
        module: Annotated[Module, typer.Option(
            "-m", "--module",
            help="Server module to load"
        )] = Module.ALL,
        transport: Annotated[Transport, typer.Option(
            "-t", "--transport",
            help="Transport protocol to use"
        )] = Transport.STDIO,
        host: Annotated[str, typer.Option(
            "--host",
            help="Host to bind to (for HTTP transports)"
        )] = "127.0.0.1",
        port: Annotated[int, typer.Option(
            "-p", "--port",
            help="Port to bind to (for HTTP transports)"
        )] = 8000,
        path: Annotated[str, typer.Option(
            "--path",
            help="Path for streamable HTTP (default: /mcp)"
        )] = "/mcp",
        db_path: Annotated[Optional[str], typer.Option(
            "--db-path",
            help="Path to DROMA SQLite database"
        )] = None,
        r_libs: Annotated[Optional[str], typer.Option(
            "--r-libs",
            help="Path to R libraries (for DROMA.Set and DROMA.R packages)"
        )] = None,
        verbose: Annotated[bool, typer.Option(
            "-v", "--verbose",
            help="Enable verbose logging"
        )] = False,
        validate_deps: Annotated[bool, typer.Option(
            "--validate-deps",
            help="Validate dependencies before starting"
        )] = True,
    ) -> None:
        """Start DROMA MCP Server with specified configuration."""
        
        # Validate dependencies if requested
        if validate_deps:
            typer.echo("Validating dependencies...")
            deps = self._validate_dependencies()
            if not all(deps.values()):
                typer.echo("⚠️  Some dependencies are missing:", err=True)
                for dep, status in deps.items():
                    status_icon = "✓" if status else "✗"
                    typer.echo(f"  {status_icon} {dep}")
                
                if not typer.confirm("Continue anyway?"):
                    raise typer.Exit(1)
            else:
                typer.echo("✓ All dependencies validated")
        
        # Setup environment
        self._setup_environment(module, db_path, r_libs, verbose)
        
        if verbose:
            typer.echo(f"Starting DROMA MCP Server v{__version__}")
            typer.echo(f"Configuration:")
            typer.echo(f"  Module: {module.value}")
            typer.echo(f"  Transport: {transport.value}")
            if transport != Transport.STDIO:
                typer.echo(f"  Host: {host}")
                typer.echo(f"  Port: {port}")
                if transport == Transport.HTTP:
                    typer.echo(f"  Path: {path}")
            if db_path:
                typer.echo(f"  Database: {db_path}")
            if r_libs:
                typer.echo(f"  R Libraries: {r_libs}")
        
        try:
            # Import and setup server
            from .server import droma_mcp
            from .util import setup_server
            
            # Run setup
            asyncio.run(setup_server())
            
            # Start server with appropriate transport
            self._start_server(transport, host, port, path)
            
        except ImportError as e:
            typer.echo(f"Error importing server modules: {e}", err=True)
            typer.echo("Make sure all dependencies are installed:", err=True)
            typer.echo("  pip install -e .", err=True)
            raise typer.Exit(1)
            
        except Exception as e:
            typer.echo(f"Error starting server: {e}", err=True)
            raise typer.Exit(1)
    
    def _start_server(self, transport: Transport, host: str, port: int, path: str):
        """Start the server with the specified transport."""
        from .server import droma_mcp
        
        if transport == Transport.STDIO:
            typer.echo("✓ Starting server with STDIO transport...")
            typer.echo("  Ready for AI assistant connections")
            droma_mcp.run()
            
        elif transport == Transport.HTTP:
            from .util import get_data_export, get_figure
            from starlette.routing import Route
            
            typer.echo(f"✓ Starting server with HTTP transport on {host}:{port}{path}")
            typer.echo(f"  API endpoint: http://{host}:{port}{path}")
            typer.echo(f"  Download endpoints available at /download/*")
            
            # Add HTTP routes for data export and figures (FastMCP 2.13 compatible)
            try:
                # Try to use the new API if available
                droma_mcp.add_http_routes([
                    Route("/download/export/{data_id}", endpoint=get_data_export),
                    Route("/download/figure/{figure_name}", endpoint=get_figure)
                ])
            except AttributeError:
                # Fallback to old API
                droma_mcp._additional_http_routes = [
                    Route("/download/export/{data_id}", endpoint=get_data_export),
                    Route("/download/figure/{figure_name}", endpoint=get_figure)
                ]
            
            # Use "http" transport for FastMCP 2.13+
            droma_mcp.run(
                transport="http",
                host=host,
                port=port,
                path=path
            )
            
        elif transport == Transport.SSE:
            typer.echo(f"✓ Starting server with SSE transport on {host}:{port}")
            typer.echo(f"  SSE endpoint: http://{host}:{port}/sse")
            droma_mcp.run(
                transport="sse",
                host=host,
                port=port
            )
    
    def test_connection(
        self,
        db_path: Annotated[Optional[str], typer.Option(
            "--db-path",
            help="Path to DROMA SQLite database"
        )] = None,
        r_libs: Annotated[Optional[str], typer.Option(
            "--r-libs",
            help="Path to R libraries"
        )] = None,
    ) -> None:
        """Test DROMA MCP server configuration and dependencies."""
        
        typer.echo("Testing DROMA MCP Server configuration...")
        
        # Test dependencies
        typer.echo("\n1. Testing dependencies...")
        deps = self._validate_dependencies()
        
        for dep, status in deps.items():
            status_icon = "✓" if status else "✗"
            dep_name = dep.replace('_', ' ').title()
            typer.echo(f"  {status_icon} {dep_name}")
        
        # Test R version and packages details
        if deps['r_integration']:
            try:
                import rpy2.robjects as robjects
                r = robjects.r
                r_version = r('R.version.string')[0]
                typer.echo(f"    R Version: {r_version}")
                
                if deps['droma_packages']:
                    # Check package versions
                    try:
                        droma_set_version = r('packageVersion("DROMA.Set")')[0]
                        droma_r_version = r('packageVersion("DROMA.R")')[0]
                        typer.echo(f"    DROMA.Set: v{droma_set_version}")
                        typer.echo(f"    DROMA.R: v{droma_r_version}")
                    except:
                        pass
            except:
                pass
        
        # Test database connection
        if db_path:
            typer.echo(f"\n2. Testing database: {db_path}")
            self._test_database(db_path)
        else:
            typer.echo("\n2. No database path provided (use --db-path to test)")
        
        # Test server import
        typer.echo("\n3. Testing server import...")
        try:
            from .server import droma_mcp
            typer.echo("  ✓ DROMA MCP server can be imported")
        except ImportError as e:
            typer.echo(f"  ✗ Server import failed: {e}")
        
        # Overall status
        all_critical_deps = deps['python_deps'] and deps['r_integration']
        typer.echo(f"\n{'✓' if all_critical_deps else '✗'} Configuration test completed!")
        
        if not all_critical_deps:
            typer.echo("Critical dependencies missing. Please install required packages.")
            raise typer.Exit(1)
    
    def _test_database(self, db_path: str):
        """Test database connection and structure."""
        if not Path(db_path).exists():
            typer.echo(f"  ✗ Database file not found: {db_path}")
            return
        
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check for required tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['sample_anno', 'drug_anno']
            for table in required_tables:
                if table in tables:
                    typer.echo(f"    ✓ Table '{table}' found")
                else:
                    typer.echo(f"    ! Table '{table}' not found")
            
            conn.close()
            typer.echo("  ✓ Database connection successful")
            
        except Exception as e:
            typer.echo(f"  ✗ Database connection failed: {e}")
    
    def info(self) -> None:
        """Display information about DROMA MCP server."""
        
        typer.echo(f"""
DROMA MCP Server v{__version__}
{__author__}

A Model Context Protocol server for drug-omics association analysis using DROMA.

Available modules:
  • all                - All modules (default)
  • data_loading       - Data loading, caching, and normalization operations
  • database_query     - Database query and exploration operations
  • dataset_management - Dataset loading and management operations

Available transports:
  • stdio - Standard input/output (default, for AI assistants)
  • http  - HTTP with streaming support (FastMCP 2.13+)
  • sse   - Server-Sent Events

Usage Examples:
  droma-mcp run                        # Start with default settings (STDIO)
  droma-mcp run -m data_loading        # Start with only data loading module
  droma-mcp run -t http -p 8080        # Start HTTP server on port 8080
  droma-mcp test --db-path path/to/db  # Test configuration
  droma-mcp validate                   # Validate installation
  droma-mcp benchmark                  # Run performance benchmark

Environment Variables:
  DROMA_DB_PATH         - Default database path
  R_LIBS                - R library path
  DROMA_MCP_VERBOSE     - Enable verbose logging

Documentation: https://github.com/mugpeng/DROMA
""")
    
    def export_config(
        self,
        output: Annotated[str, typer.Option(
            "-o", "--output",
            help="Output file path"
        )] = "droma-mcp-config.json",
        transport: Annotated[Transport, typer.Option(
            "-t", "--transport",
            help="Transport protocol"
        )] = Transport.STDIO,
        host: Annotated[str, typer.Option(
            "--host",
            help="Host for HTTP transports"
        )] = "127.0.0.1",
        port: Annotated[int, typer.Option(
            "-p", "--port",
            help="Port for HTTP transports"
        )] = 8000,
    ) -> None:
        """Export MCP client configuration file."""
        
        config = self._generate_config(transport, host, port)
        
        with open(output, 'w') as f:
            json.dump(config, f, indent=2)
        
        typer.echo(f"MCP client configuration exported to: {output}")
    
    def _generate_config(self, transport: Transport, host: str, port: int) -> Dict[str, Any]:
        """Generate MCP client configuration for FastMCP 2.13+."""
        if transport == Transport.STDIO:
            return {
                "mcpServers": {
                    "droma-mcp": {
                        "command": "droma-mcp",
                        "args": ["run", "--module", "all", "--transport", "stdio"],
                        "env": {
                            "DROMA_DB_PATH": "${DROMA_DB_PATH}"
                        }
                    }
                }
            }
        elif transport == Transport.HTTP:
            return {
                "mcpServers": {
                    "droma-mcp": {
                        "url": f"http://{host}:{port}/mcp",
                        "transport": "http"
                    }
                }
            }
        elif transport == Transport.SSE:
            return {
                "mcpServers": {
                    "droma-mcp": {
                        "url": f"http://{host}:{port}/sse",
                        "transport": "sse"
                    }
                }
            }
    
    def validate_setup(self) -> None:
        """Validate complete DROMA MCP setup."""
        typer.echo("Validating DROMA MCP setup...")
        
        # Check dependencies
        deps = self._validate_dependencies()
        
        # Check environment variables
        env_vars = {
            'DROMA_DB_PATH': os.environ.get('DROMA_DB_PATH'),
            'R_LIBS': os.environ.get('R_LIBS')
        }
        
        # Check file structure
        package_files = [
            'src/droma_mcp/__init__.py',
            'src/droma_mcp/server/__init__.py',
            'src/droma_mcp/schema/__init__.py',
        ]
        
        # Report results
        typer.echo("\n📋 Validation Results:")
        typer.echo("━━━━━━━━━━━━━━━━━━━━")
        
        typer.echo("Dependencies:")
        for dep, status in deps.items():
            icon = "✅" if status else "❌"
            typer.echo(f"  {icon} {dep.replace('_', ' ').title()}")
        
        typer.echo("\nEnvironment:")
        for var, value in env_vars.items():
            icon = "✅" if value else "⚠️"
            status = f"Set to: {value}" if value else "Not set"
            typer.echo(f"  {icon} {var}: {status}")
        
        typer.echo("\nPackage Structure:")
        for file_path in package_files:
            exists = Path(file_path).exists()
            icon = "✅" if exists else "❌"
            typer.echo(f"  {icon} {file_path}")
        
        # Overall assessment
        all_deps = all(deps.values())
        typer.echo(f"\n{'🎉' if all_deps else '⚠️'} Overall Status: {'Ready to use' if all_deps else 'Issues found'}")
        
        if not all_deps:
            typer.echo("\n💡 Recommendations:")
            if not deps['python_deps']:
                typer.echo("  • Install Python dependencies: pip install -e .")
            if not deps['r_integration']:
                typer.echo("  • Install R integration: pip install rpy2")
            if not deps['droma_packages']:
                typer.echo("  • Install DROMA R packages")
    
    def benchmark(
        self,
        iterations: Annotated[int, typer.Option(
            "-n", "--iterations",
            help="Number of benchmark iterations"
        )] = 5,
        module: Annotated[Module, typer.Option(
            "-m", "--module",
            help="Module to benchmark"
        )] = Module.ALL,
    ) -> None:
        """Run performance benchmark."""
        import time
        
        typer.echo(f"Running DROMA MCP benchmark ({iterations} iterations)...")
        
        # Test import performance
        start_time = time.time()
        try:
            from .server import droma_mcp
            import_time = time.time() - start_time
            typer.echo(f"✓ Import time: {import_time:.3f}s")
        except Exception as e:
            typer.echo(f"✗ Import failed: {e}")
            return
        
        # Test R integration performance
        if self._validate_dependencies()['r_integration']:
            start_time = time.time()
            try:
                import rpy2.robjects as robjects
                r = robjects.r
                r('library(DROMA.Set)')
                r_time = time.time() - start_time
                typer.echo(f"✓ R setup time: {r_time:.3f}s")
            except Exception as e:
                typer.echo(f"✗ R setup failed: {e}")
        
        typer.echo("Benchmark completed!")


# Create CLI instance
cli = DromaMCPCLI()
app = cli.app

# For backwards compatibility
if __name__ == "__main__":
    app() 