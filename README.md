[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/mugpeng-droma-mcp-badge.png)](https://mseep.ai/app/mugpeng-droma-mcp)

# DROMA MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for **DROMA** (Drug Response Omics association MAp) - enabling natural language interactions with drug-omics association analysis.

[![Website](https://img.shields.io/website?url=https%3A//droma01.github.io/)](https://droma01.github.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.13+-green.svg)](https://github.com/jlowin/fastmcp)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/mugpeng/DROMA_MCP/releases)


## 🚀 Overview

DROMA MCP Server bridges the gap between AI assistants and cancer pharmacogenomics analysis by providing a natural language interface to the [DROMA.R](https://github.com/mugpeng/DROMA_R) and [DROMA.Set](https://github.com/mugpeng/DROMA_Set) packages.

It is a part of [DROMA project](https://github.com/mugpeng/DROMA). Visit the [official DROMA website](https://droma01.github.io/) for comprehensive documentation and interactive examples.

### Key Features

- **🔗 Natural Language Interface**: Ask questions about drug-omics associations in plain English
- **📊 Dataset Management**: Load and manage DROMA datasets (CCLE, gCSI, etc.) in memory
- **📈 Data Loading & Normalization**: Load molecular profiles and treatment response data with automatic z-score normalization
- **🗂️ Multi-Project Support**: Seamlessly work with data across multiple research projects
- **💾 Smart Caching**: Efficient data caching with metadata tracking for faster access
- **📤 Data Export**: Export analysis results to various formats (CSV, Excel, JSON)
- **⚡ Multi-Modal Support**: Works with various transport protocols (STDIO, HTTP, SSE)
- **🔄 R Integration**: Seamless integration with existing DROMA R packages via rpy2
- **🚄 Performance Optimizations**: Memory management, asynchronous processing, and connection pooling
- **🛡️ Robust Error Handling**: Comprehensive validation, logging, and graceful error recovery
- **🎛️ Class-Based CLI**: Modern, type-safe command-line interface with comprehensive help
- **🔐 Enterprise Authentication**: Optional OAuth, API Key, and JWT authentication (FastMCP 2.13+)
- **🔌 Modular Architecture**: Clean separation with mount/import patterns for composability

### 🏎️ Performance Features

- **Asynchronous Processing**: Non-blocking I/O operations for better responsiveness
- **Memory Management**: Automatic memory monitoring and garbage collection
- **Connection Pooling**: Efficient R environment management
- **Smart Caching**: LRU cache with size limits and automatic eviction
- **Batch Operations**: Process multiple datasets efficiently
- **Performance Monitoring**: Built-in metrics tracking and reporting

## 📦 Installation

### Prerequisites

- Python 3.10+
- R 4.0+ with DROMA.Set and DROMA.R packages
- DROMA SQLite database
- FastMCP 2.13+ (automatically installed)

### Install via pip

```bash
pip install droma-mcp
```

### Development Installation

```bash
git clone https://github.com/mugpeng/DROMA_MCP
cd DROMA_MCP
pip install -e .
```

### ⬆️ Upgrading from v0.1.0

If you're upgrading from an older version:

```bash
# Upgrade to latest version
pip install --upgrade 'droma-mcp>=0.2.0'

# Update transport protocol in commands:
# OLD: --transport streamable-http
# NEW: --transport http

# See UPGRADE_GUIDE.md for detailed migration steps
```

### R Dependencies

Ensure you have the DROMA R packages installed:

```r
# Install DROMA.Set and DROMA.R packages
# devtools::install_github("mugpeng/DROMA_Set")
# devtools::install_github("mugpeng/DROMA_R")
```

## 🚀 Quick Start

### 1. Validate Setup

```bash
# Check dependencies and configuration
droma-mcp validate

# Test specific database connection
droma-mcp test --db-path path/to/droma.sqlite --r-libs path/to/R/libs
```

### 2. Performance Benchmark

```bash
# Run performance benchmark
droma-mcp benchmark

# Benchmark specific module
droma-mcp benchmark --module data_loading --iterations 10
```

### 3. Start the Server

```bash
# STDIO mode (for AI assistants) - default
droma-mcp run --db-path path/to/droma.sqlite

# HTTP mode (for web applications)
droma-mcp run --transport http --port 8000 --db-path path/to/droma.sqlite

# With verbose logging and dependency validation
droma-mcp run --verbose --validate-deps --db-path path/to/droma.sqlite
```

### 4. MCP Client Configuration

Export a configuration file for your MCP client:

```bash
# Generate STDIO configuration
droma-mcp export-config -o droma-config.json

# Generate HTTP configuration
droma-mcp export-config -o droma-http-config.json --transport http --port 8000
```

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "droma-mcp": {
      "command": "droma-mcp",
      "args": ["run", "--db-path", "path/to/droma.sqlite"]
    }
  }
}
```

## 🖥️ CLI Reference

The DROMA MCP CLI provides comprehensive commands for server management and testing:

### Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `run` | Start the MCP server | `droma-mcp run --db-path db.sqlite` |
| `test` | Test configuration and dependencies | `droma-mcp test --db-path db.sqlite` |
| `validate` | Validate complete setup | `droma-mcp validate` |
| `benchmark` | Run performance benchmark | `droma-mcp benchmark -n 10` |
| `export-config` | Export MCP client configuration | `droma-mcp export-config -o config.json` |
| `info` | Display server information | `droma-mcp info` |

### Global Options

- `--help`: Show help for any command
- `--verbose`: Enable verbose logging
- `--db-path`: Path to DROMA SQLite database
- `--r-libs`: Path to R libraries

### Transport Options

- `--transport`: Choose transport protocol (`stdio`, `http`, `sse`)
  - **Note:** FastMCP 2.13+ uses `http` instead of `streamable-http`
- `--host`: Host for HTTP transports (default: `127.0.0.1`)
- `--port`: Port for HTTP transports (default: `8000`)
- `--path`: Path for HTTP (default: `/mcp`)

### Module Selection

- `--module`: Choose server module (`all`, `data_loading`, `database_query`, `dataset_management`)

### Examples

```bash
# Get detailed help
droma-mcp --help
droma-mcp run --help

# Validate environment
droma-mcp validate

# Test with specific database
droma-mcp test --db-path /path/to/droma.db

# Start server with all modules
droma-mcp run --module all --db-path /path/to/droma.db

# Start HTTP server with data loading only
droma-mcp run --module data_loading --transport http --port 8080

# Export configuration for HTTP mode
droma-mcp export-config --transport http --port 8080 -o http-config.json

# Run performance benchmark
droma-mcp benchmark --iterations 5 --module data_loading
```

## 🔐 Authentication (Optional)

**New in v0.2.0:** Enterprise-grade authentication for production deployments.

### When Do You Need Authentication?

- ✅ **HTTP/SSE deployments** exposed to the internet
- ✅ **Multi-user environments** requiring access control
- ✅ **Production systems** with sensitive data
- ❌ **Local STDIO mode** (not needed for AI assistants)
- ❌ **Internal networks** with existing security

### Supported Authentication Methods

```python
from droma_mcp.auth import create_auth_provider

# 1. API Key (simplest, for internal services)
auth = create_auth_provider("api_key", api_keys=["secret-key-123"])

# 2. Google OAuth (for user authentication)
auth = create_auth_provider(
    "google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    base_url="https://your-server.com"
)

# 3. GitHub OAuth
auth = create_auth_provider(
    "github",
    client_id=os.getenv("GITHUB_CLIENT_ID"),
    client_secret=os.getenv("GITHUB_CLIENT_SECRET"),
    base_url="https://your-server.com"
)

# 4. Microsoft Azure AD
auth = create_auth_provider(
    "azure",
    client_id=os.getenv("AZURE_CLIENT_ID"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET"),
    base_url="https://your-server.com",
    tenant_id="your-tenant-id"
)
```

### Using Environment Variables

```bash
# API Key authentication
export DROMA_MCP_AUTH_PROVIDER=api_key
export DROMA_MCP_API_KEYS=key1,key2,key3

# Google OAuth
export DROMA_MCP_AUTH_PROVIDER=google
export DROMA_MCP_CLIENT_ID=your-client-id
export DROMA_MCP_CLIENT_SECRET=your-client-secret
export DROMA_MCP_BASE_URL=https://your-server.com

# Start server (authentication auto-loaded from env)
droma-mcp run --transport http --port 8000
```

**Note:** Authentication is completely **optional** and only needed for web deployments. Local STDIO mode (default) does not require authentication.

## 🛠️ Available Tools

### Dataset Management (Required First Step)

- **`load_dataset`**: Load DROMA datasets (CCLE, gCSI, etc.) into memory from database
- **`list_loaded_datasets`**: Show which datasets are currently loaded in memory
- **`set_active_dataset`**: Set the active dataset for subsequent operations
- **`unload_dataset`**: Remove datasets from memory to free up resources

### Data Loading & Analysis

- **`load_molecular_profiles_normalized`**: Load molecular profiles (mRNA, CNV, methylation, etc.) with z-score normalization
- **`load_treatment_response_normalized`**: Load drug response data with normalization
- **`load_multi_project_molecular_profiles_normalized`**: Load data across multiple projects
- **`load_multi_project_treatment_response_normalized`**: Load treatment response across projects
- **`check_zscore_normalization`**: Verify normalization status of cached data
- **`get_cached_data_info`**: Get information about cached datasets
- **`export_cached_data`**: Export data to CSV/Excel/JSON formats

### Database Query & Exploration

- **`get_droma_annotation`**: Retrieve sample or drug annotation data from the database
- **`list_droma_samples`**: List all available samples for a project with filtering options
- **`list_droma_features`**: List all available features (genes, drugs) for a project and data type
- **`list_droma_projects`**: List all projects available in the DROMA database

## 💬 Example Usage with AI Assistants

### Essential Workflow: Load Dataset First

**⚠️ Important**: Before using any data loading functions, you must first load the dataset into memory:

> "Load the CCLE dataset from the database and set it as active"

> "Show me which datasets are currently loaded in memory"

### Data Loading Examples

After loading datasets, you can perform data operations:

> "Load mRNA expression data for ABCB1 gene from the CCLE dataset with z-score normalization"

> "Get Paclitaxel drug response data across multiple projects, including only overlapping samples"

### Database Exploration

> "List all projects available in the DROMA database"

> "Show me all available samples for the gCSI project that have mRNA data"

> "Get the first 100 genes available in the CCLE mRNA dataset"

> "Retrieve sample annotations for breast cancer cell lines in the gCSI project"

### Complete Workflow Example

1. **First, load your dataset:**
   > "Load the CCLE dataset into memory"

2. **Then, load and analyze data:**
   > "Load mRNA expression data for TP53 gene from CCLE with z-score normalization"

3. **Check results:**
   > "Show me information about cached datasets and export the mRNA data to CSV"

### Data Management & Analysis

> "Check if my cached molecular data has been z-score normalized and show me the statistics"

> "Load multi-project treatment response data for Doxorubicin and verify the normalization status"

> "Unload the gCSI dataset to free up memory"

## 🏗️ Architecture

The DROMA MCP server follows a streamlined modular architecture:

```
src/droma_mcp/
├── __init__.py          # Package initialization
├── __main__.py          # Entry point
├── cli.py               # Command-line interface
├── util.py              # Utility functions
├── schema/              # Pydantic data models
│   ├── __init__.py
│   ├── data_loading.py      # Data loading schemas
│   ├── database_query.py    # Database query schemas
│   └── dataset_management.py # Dataset management schemas
└── server/              # MCP server modules
    ├── __init__.py          # Server setup & state
    ├── dataset_management.py # Dataset loading/management
    ├── data_loading.py      # Data loading operations
    └── database_query.py    # Database exploration operations
```

### Key Components

- **Modular Design**: Separate modules for dataset management, data loading, and database exploration
- **State Management**: Centralized data and R environment management with active dataset tracking
- **Type Safety**: Comprehensive Pydantic validation for all inputs
- **R Integration**: Seamless R-Python data exchange via rpy2
- **Database Access**: Direct SQLite database querying for exploration
- **Caching System**: Efficient data caching with metadata tracking

## 🔧 Configuration

### Environment Variables

- `DROMA_DB_PATH`: Default path to DROMA SQLite database
- `R_LIBS`: Path to R libraries
- `DROMA_MCP_MODULE`: Server module to load (`all`, `data_loading`, `database_query`, `dataset_management`)
- `DROMA_MCP_VERBOSE`: Enable verbose logging

### Command Line Options

```bash
droma-mcp run --help
```

## 🧪 Development

### Testing the Framework

```bash
# Validate complete setup
droma-mcp validate

# Test dependencies and database connection
droma-mcp test --db-path path/to/droma.db

# Run performance benchmark
droma-mcp benchmark --iterations 5

# Test CLI functionality
droma-mcp info
droma-mcp export-config -o test-config.json
```

### Import Testing

```bash
# Test core imports
python -c "from src.droma_mcp.cli import cli; print('✓ CLI works')"
python -c "from src.droma_mcp.server import droma_mcp; print('✓ Server works')"
python -c "from src.droma_mcp.util import setup_server; print('✓ Utils work')"
```

### Code Quality

```bash
black src/
isort src/
mypy src/
```

### Adding New Tools

1. Define Pydantic schemas in `schema/`
2. Implement server functions in `server/`
3. Register tools with FastMCP decorators
4. Update CLI module loading logic

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Check if all dependencies are installed
droma-mcp validate

# Install missing dependencies
pip install -e .
```

#### R Integration Issues
```bash
# Test R environment
droma-mcp test --r-libs /path/to/R/libs

# Install R dependencies
R -e "install.packages('rpy2')"
```

#### Database Connection Problems
```bash
# Test database connection
droma-mcp test --db-path /path/to/droma.db

# Check database file exists and is readable
ls -la /path/to/droma.db
```

#### Server Startup Issues
```bash
# Run with verbose logging
droma-mcp run --verbose --validate-deps

# Test specific module
droma-mcp run --module data_loading --verbose
```

### Environment Variables

If experiencing issues, set these environment variables:

```bash
export DROMA_DB_PATH="/path/to/droma.sqlite"
export R_LIBS="/path/to/R/libraries"
export DROMA_MCP_VERBOSE="1"
```

### Performance Issues

```bash
# Run benchmark to identify bottlenecks
droma-mcp benchmark --iterations 10

# Monitor memory usage during operations
droma-mcp run --verbose
```

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MPL-2 License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [DROMA](https://github.com/mugpeng/DROMA) - Main DROMA project
- [DROMA.Set](https://github.com/mugpeng/DROMA_Set) - R package for data management
- [DROMA.R](https://github.com/mugpeng/DROMA_R) - R package for analysis functions
- [FastMCP](https://github.com/jlowin/fastmcp) - Python framework for MCP servers
- [Model Context Protocol](https://modelcontextprotocol.io/) - Open standard for AI tool integration

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mugpeng/DROMA_MCP/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mugpeng/DROMA_MCP/discussions)
- **Email**: [Contact DROMA Team](mailto:yc47680@um.edu.mo)

## 📊 Citation

If you use DROMA MCP in your research, please cite:

```bibtex
@software{droma_mcp2024,
  title={DROMA MCP: Model Context Protocol Server for Drug-Omics Association Analysis},
  author={DROMA Development Team},
  year={2024},
  url={https://github.com/mugpeng/DROMA_MCP}
}
```

---

**DROMA MCP** - Bridging AI and Cancer Pharmacogenomics 🧬💊🤖 