"""DROMA MCP Server initialization and state management."""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any, Dict, Optional
from fastmcp import FastMCP
import pandas as pd
from pathlib import Path


class DromaState:
    """Manages DROMA datasets and analysis state."""
    
    def __init__(self):
        self.datasets: Dict[str, Any] = {}  # {dataset_id: DromaSet_object}
        self.multidatasets: Dict[str, Any] = {}  # {dataset_id: MultiDromaSet_object}
        self.active_dataset: Optional[str] = None
        self.active_multidataset: Optional[str] = None
        self.analysis_cache: Dict[str, Any] = {}
        self.data_cache: Dict[str, Any] = {}  # Cache for loaded data
        self.metadata: Dict[str, Any] = {}
        
        # R environment setup
        self._setup_r_environment()
    
    def _setup_r_environment(self):
        """Initialize R environment and load DROMA packages."""
        try:
            import rpy2.robjects as robjects
            
            # Load required R libraries
            robjects.r('''
                library(DROMA.Set)
                library(DROMA.R)
            ''')
            
            self.r = robjects.r
            print("R environment initialized successfully")
            
        except Exception as e:
            print(f"Warning: Could not initialize R environment: {e}")
            self.r = None
    
    def load_dataset(self, dataset_id: str, db_path: str, dataset_type: str = "DromaSet"):
        """Load DROMA dataset by ID."""
        if self.r is None:
            raise RuntimeError("R environment not available")
            
        try:
            if dataset_type == "DromaSet":
                # Load single DromaSet
                # R function: createDromaSetFromDatabase(projects, db_path, db_group=NULL, load_metadata=TRUE, dataset_type=NULL, auto_load=FALSE, con=NULL)
                self.r(f'''
                    {dataset_id} <- createDromaSetFromDatabase(
                        projects = "{dataset_id}",
                        db_path = "{db_path}",
                        load_metadata = TRUE
                    )
                ''')
                self.datasets[dataset_id] = dataset_id  # Store R object name
                
            elif dataset_type == "MultiDromaSet":
                # Load MultiDromaSet (assuming dataset_id is comma-separated project names)
                project_names = dataset_id.split(",")
                project_names_r = 'c("' + '", "'.join(project_names) + '")'
                
                # R function: createMultiDromaSetFromDatabase(project_names, db_path, db_groups=NULL, load_metadata=TRUE, dataset_types=NULL, auto_load=FALSE, con=NULL)
                self.r(f'''
                    {dataset_id.replace(",", "_")} <- createMultiDromaSetFromDatabase(
                        project_names = {project_names_r},
                        db_path = "{db_path}",
                        load_metadata = TRUE
                    )
                ''')
                self.multidatasets[dataset_id] = dataset_id.replace(",", "_")
                
            print(f"Successfully loaded dataset: {dataset_id}")
            return True
            
        except Exception as e:
            print(f"Error loading dataset {dataset_id}: {e}")
            return False
    
    def get_dataset(self, dataset_id: Optional[str] = None) -> Optional[str]:
        """Get active or specified dataset."""
        if dataset_id:
            return self.datasets.get(dataset_id)
        return self.datasets.get(self.active_dataset) if self.active_dataset else None
    
    def get_multidataset(self, dataset_id: Optional[str] = None) -> Optional[str]:
        """Get active or specified multidataset."""
        if dataset_id:
            return self.multidatasets.get(dataset_id)
        return self.multidatasets.get(self.active_multidataset) if self.active_multidataset else None
    
    def cache_data(self, key: str, data: Any, metadata: Optional[Dict] = None):
        """Cache data with optional metadata."""
        self.data_cache[key] = {
            'data': data,
            'metadata': metadata or {},
            'timestamp': pd.Timestamp.now()
        }
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve cached data."""
        cached = self.data_cache.get(key)
        return cached['data'] if cached else None
    
    def list_datasets(self) -> Dict[str, str]:
        """List all loaded datasets."""
        return {
            'datasets': list(self.datasets.keys()),
            'multidatasets': list(self.multidatasets.keys())
        }
    
    def set_active_dataset(self, dataset_id: str, dataset_type: str = "DromaSet"):
        """Set the active dataset."""
        if dataset_type == "DromaSet" and dataset_id in self.datasets:
            self.active_dataset = dataset_id
        elif dataset_type == "MultiDromaSet" and dataset_id in self.multidatasets:
            self.active_multidataset = dataset_id
        else:
            raise ValueError(f"Dataset {dataset_id} not found")


@asynccontextmanager
async def droma_lifespan(server: FastMCP) -> AsyncIterator[DromaState]:
    """Lifespan context manager for DROMA MCP server."""
    print("Initializing DROMA MCP Server...")
    
    # Create DROMA state
    state = DromaState()
    
    # Set up temp directories for exports
    export_dir = Path.home() / ".droma_mcp" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    yield state
    
    print("Shutting down DROMA MCP Server...")


# Create the main FastMCP server instance with improved configuration
droma_mcp = FastMCP(
    name="DROMA-MCP-Server",
    lifespan=droma_lifespan
)

# Setup function that can be called before running
async def setup_server():
    """Setup function for server initialization."""
    from ..util import setup_server as util_setup
    await util_setup()

# Module loading based on environment variable
module = os.environ.get('DROMA_MCP_MODULE', 'all')

# Load modules using improved FastMCP 2.13 patterns
# We use mount() for dynamic composition (tools can access parent state)
# For static composition, use import_server() instead

if module in ['all', 'data_loading']:
    from .data_loading import data_loading_mcp
    # Mount with path prefix for better organization
    droma_mcp.mount(data_loading_mcp, prefix="data")

if module in ['all', 'database_query']:
    from .database_query import database_query_mcp
    droma_mcp.mount(database_query_mcp, prefix="query")

if module in ['all', 'dataset_management']:
    from .dataset_management import dataset_management_mcp
    droma_mcp.mount(dataset_management_mcp, prefix="datasets")

# Add server metadata
print(f"✓ DROMA MCP Server v0.2.0 initialized with module: {module}")
print(f"✓ FastMCP version: 2.13.x compatible")
print(f"✓ Available transports: STDIO, HTTP, SSE")

__all__ = ["droma_mcp", "DromaState", "setup_server"] 