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

### R Dependencies

Ensure you have the DROMA R packages installed:

```r
# Install DROMA.Set and DROMA.R packages
# devtools::install_github("mugpeng/DROMA_Set")
# devtools::install_github("mugpeng/DROMA_R")
```

## 🚀 Quick Start

### 1. Start the Server

```bash
# STDIO mode (for AI assistants) - default
droma-mcp run --db-path path/to/droma.sqlite

# HTTP mode (for web applications)
droma-mcp run --transport http --port 8000 --db-path path/to/droma.sqlite
```

### 2. MCP Client Configuration

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

## 🖥️ CLI Commands

| Command | Description |
|---------|-------------|
| `run` | Start the MCP server |
| `test` | Test configuration and dependencies |
| `validate` | Validate complete setup |
| `export-config` | Export MCP client configuration |

For detailed help: `droma-mcp --help` or `droma-mcp run --help`

## 🛠️ Available Tools

### Dataset Management

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

### Visualization & Analysis

- **`plot_drug_sensitivity_rank`**: Create drug sensitivity rank plots
- **`analyze_drug_omic_pair`**: Analyze associations between drugs and omic features

## 💬 Example Usage

### Essential Workflow

**⚠️ Important**: Before using any data loading functions, you must first load the dataset into memory:

> "Load the CCLE dataset from the database and set it as active"

> "Load mRNA expression data for ABCB1 gene from the CCLE dataset with z-score normalization"

> "List all projects available in the DROMA database"

> "Show me all available samples for the gCSI project that have mRNA data"

## 🔧 Configuration

### Environment Variables

- `DROMA_DB_PATH`: Default path to DROMA SQLite database
- `R_LIBS`: Path to R libraries
- `DROMA_MCP_MODULE`: Server module to load (`all`, `data_loading`, `database_query`, `dataset_management`, `visualization`, `analysis`)
- `DROMA_MCP_VERBOSE`: Enable verbose logging

## 🔧 Troubleshooting

### Common Issues

**Import Errors**: Run `droma-mcp validate` to check dependencies

**R Integration Issues**: Ensure R and DROMA packages are installed correctly

**Database Connection**: Verify database path with `droma-mcp test --db-path path/to/droma.db`

For more help, see the [official DROMA website](https://droma01.github.io/) or open an issue on GitHub.

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

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

## Citation

If you use DROMA MCP in your research, please cite:

```
Li, S., Peng, Y., Chen, M. et al. Facilitating integrative and personalized oncology omics analysis with UCSCXenaShiny. Commun Biol 7, 1200 (2024). https://doi.org/10.1038/s42003-024-06891-2

```

---

**DROMA MCP** - Bridging AI and Cancer Pharmacogenomics 🧬💊🤖
