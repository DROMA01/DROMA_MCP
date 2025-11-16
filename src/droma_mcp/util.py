"""Utility functions for DROMA MCP server."""

import tempfile
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# Global storage for exports and figures
EXPORTS: Dict[str, str] = {}
FIGURES: Dict[str, str] = {}


def get_server_url() -> Union[str, None]:
    """
    Get the server URL if running in HTTP transport mode.

    Returns:
        str: Base URL for HTTP transport, None for STDIO/SSE
    """
    transport = os.environ.get('DROMA_MCP_TRANSPORT', '').lower()
    if transport == 'http':
        host = os.environ.get('DROMA_MCP_HOST', '127.0.0.1')
        port = os.environ.get('DROMA_MCP_PORT', '8000')
        return f"http://{host}:{port}"
    return None


def get_figure_url(figure_id: str) -> str:
    """
    Generate accessible URL for a figure.

    Args:
        figure_id: The figure identifier

    Returns:
        str: URL to access the figure
    """
    base_url = get_server_url()
    if base_url:
        return f"{base_url}/download/figure/{figure_id}"
    return ""


def get_export_url(data_id: str) -> str:
    """
    Generate accessible URL for data export.

    Args:
        data_id: The data export identifier

    Returns:
        str: URL to access the exported data
    """
    base_url = get_server_url()
    if base_url:
        return f"{base_url}/download/export/{data_id}"
    return ""


def save_analysis_result(
    result_df: pd.DataFrame, 
    name: Optional[str] = None,
    format: str = "csv"
) -> str:
    """
    Save analysis results for download.
    
    Args:
        result_df: DataFrame to save
        name: Optional filename (auto-generated if None)
        format: File format ('csv', 'excel', 'json')
    
    Returns:
        Export identifier for retrieval
        
    Raises:
        ValueError: If format is not supported
        IOError: If file cannot be written
    """
    if format not in ['csv', 'excel', 'json']:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'excel', or 'json'")
    
    if name is None:
        name = f"droma_analysis_{len(EXPORTS)}.{format}"
    
    # Ensure name has correct extension
    if not name.endswith(f'.{format}'):
        name = f"{name}.{format}"
    
    # Create temp directory
    temp_dir = Path(tempfile.gettempdir()) / "droma_mcp_exports"
    temp_dir.mkdir(exist_ok=True)
    
    # Save file
    filepath = temp_dir / name
    
    try:
        if format == "csv":
            result_df.to_csv(filepath, index=True)  # Keep index for feature names
        elif format == "excel":
            result_df.to_excel(filepath, index=True, engine='openpyxl')  # Keep index for feature names
        elif format == "json":
            result_df.to_json(filepath, orient='split', indent=2)  # Use 'split' to preserve index
        
        # Store in global registry
        export_id = name.replace(f'.{format}', '')
        EXPORTS[export_id] = str(filepath)
        
        logger.info(f"Saved analysis result: {export_id} ({format})")
        return export_id
        
    except Exception as e:
        logger.error(f"Failed to save analysis result: {e}")
        raise IOError(f"Could not save file {filepath}: {e}")


def save_figure(fig_path: Union[str, Path], name: Optional[str] = None) -> str:
    """
    Save figure for download.
    
    Args:
        fig_path: Path to the figure file
        name: Optional filename (auto-generated if None)
    
    Returns:
        Figure identifier for retrieval
        
    Raises:
        FileNotFoundError: If figure file doesn't exist
        ValueError: If file is not a valid image format
    """
    fig_path = Path(fig_path)
    
    if not fig_path.exists():
        raise FileNotFoundError(f"Figure file not found: {fig_path}")
    
    # Validate image format
    valid_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.svg', '.eps'}
    if fig_path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Invalid image format: {fig_path.suffix}. "
                        f"Supported formats: {', '.join(valid_extensions)}")
    
    if name is None:
        name = f"droma_figure_{len(FIGURES)}{fig_path.suffix}"
    
    # Ensure name has correct extension
    if not name.endswith(fig_path.suffix):
        name = f"{name}{fig_path.suffix}"
    
    # Create temp directory and copy file
    temp_dir = Path(tempfile.gettempdir()) / "droma_mcp_figures"
    temp_dir.mkdir(exist_ok=True)
    
    dest_path = temp_dir / name
    
    try:
        import shutil
        
        # Only copy if source and destination are different
        if fig_path.resolve() != dest_path.resolve():
            shutil.copy2(fig_path, dest_path)
        
        # Store in global registry
        fig_id = name.replace(fig_path.suffix, '')
        FIGURES[fig_id] = str(dest_path)
        
        logger.info(f"Saved figure: {fig_id}")
        return fig_id
        
    except Exception as e:
        logger.error(f"Failed to save figure: {e}")
        raise IOError(f"Could not copy figure {fig_path}: {e}")


async def get_data_export(request: Request) -> Union[FileResponse, JSONResponse]:
    """
    Handle data export download requests.
    
    Args:
        request: Starlette request object
        
    Returns:
        FileResponse for valid files, JSONResponse for errors
    """
    try:
        data_id = request.path_params.get('data_id')
        
        if not data_id:
            return JSONResponse(
                {"error": "Missing data_id parameter"}, 
                status_code=400
            )
        
        # Check if export exists
        if data_id not in EXPORTS:
            return JSONResponse(
                {"error": f"Export not found: {data_id}"}, 
                status_code=404
            )
        
        filepath = EXPORTS[data_id]
        
        # Verify file still exists
        if not Path(filepath).exists():
            return JSONResponse(
                {"error": f"Export file not found on disk: {filepath}"}, 
                status_code=404
            )
        
        # Return file
        filename = Path(filepath).name
        return FileResponse(
            filepath,
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Export download error: {e}")
        return JSONResponse(
            {"error": f"Internal server error: {str(e)}"}, 
            status_code=500
        )


async def get_figure(request: Request) -> Union[FileResponse, JSONResponse]:
    """
    Handle figure download requests.
    
    Args:
        request: Starlette request object
        
    Returns:
        FileResponse for valid files, JSONResponse for errors
    """
    try:
        figure_name = request.path_params.get('figure_name')
        
        if not figure_name:
            return JSONResponse(
                {"error": "Missing figure_name parameter"}, 
                status_code=400
            )
        
        # Check if figure exists
        if figure_name not in FIGURES:
            return JSONResponse(
                {"error": f"Figure not found: {figure_name}"}, 
                status_code=404
            )
        
        filepath = FIGURES[figure_name]
        
        # Verify file still exists
        if not Path(filepath).exists():
            return JSONResponse(
                {"error": f"Figure file not found on disk: {filepath}"}, 
                status_code=404
            )
        
        # Return file with appropriate content type
        return FileResponse(filepath)
        
    except Exception as e:
        logger.error(f"Figure download error: {e}")
        return JSONResponse(
            {"error": f"Internal server error: {str(e)}"}, 
            status_code=500
        )


def list_available_files() -> Dict[str, Any]:
    """
    List all available files for download.
    
    Returns:
        Dictionary with exports and figures information
    """
    try:
        exports_info = {}
        for export_id, filepath in EXPORTS.items():
            path = Path(filepath)
            if path.exists():
                exports_info[export_id] = {
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "created": path.stat().st_mtime,
                    "format": path.suffix[1:]  # Remove dot
                }
        
        figures_info = {}
        for fig_id, filepath in FIGURES.items():
            path = Path(filepath)
            if path.exists():
                figures_info[fig_id] = {
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "created": path.stat().st_mtime,
                    "format": path.suffix[1:]  # Remove dot
                }
        
        return {
            "exports": exports_info,
            "figures": figures_info,
            "total_exports": len(exports_info),
            "total_figures": len(figures_info)
        }
        
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return {"error": str(e)}


def cleanup_temp_files(max_age_hours: int = 24) -> Dict[str, int]:
    """
    Clean up old temporary files.
    
    Args:
        max_age_hours: Maximum age of files to keep (in hours)
        
    Returns:
        Summary of cleanup operation
    """
    import time
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    cleaned_exports = 0
    cleaned_figures = 0
    
    try:
        # Clean exports
        for export_id in list(EXPORTS.keys()):
            filepath = Path(EXPORTS[export_id])
            if filepath.exists():
                file_age = current_time - filepath.stat().st_mtime
                if file_age > max_age_seconds:
                    filepath.unlink()
                    del EXPORTS[export_id]
                    cleaned_exports += 1
            else:
                # File doesn't exist, remove from registry
                del EXPORTS[export_id]
                cleaned_exports += 1
        
        # Clean figures
        for fig_id in list(FIGURES.keys()):
            filepath = Path(FIGURES[fig_id])
            if filepath.exists():
                file_age = current_time - filepath.stat().st_mtime
                if file_age > max_age_seconds:
                    filepath.unlink()
                    del FIGURES[fig_id]
                    cleaned_figures += 1
            else:
                # File doesn't exist, remove from registry
                del FIGURES[fig_id]
                cleaned_figures += 1
        
        logger.info(f"Cleanup completed: {cleaned_exports} exports, {cleaned_figures} figures removed")
        
        return {
            "cleaned_exports": cleaned_exports,
            "cleaned_figures": cleaned_figures,
            "remaining_exports": len(EXPORTS),
            "remaining_figures": len(FIGURES)
        }
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        return {"error": str(e)}


def format_data_size(size_bytes: int) -> str:
    """Format data size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def validate_cache_key(cache_key: str) -> bool:
    """Validate cache key format."""
    # Basic validation - alphanumeric and underscores only
    import re
    return bool(re.match(r'^[a-zA-Z0-9_]+$', cache_key))


def generate_cache_key(prefix: str, dataset_id: str, data_type: str) -> str:
    """Generate a standardized cache key."""
    # Remove any problematic characters and create a clean key
    clean_dataset = "".join(c for c in dataset_id if c.isalnum() or c in ['_', '-'])
    clean_type = "".join(c for c in data_type if c.isalnum() or c in ['_', '-'])
    
    return f"{prefix}_{clean_dataset}_{clean_type}"


class DataValidator:
    """Utility class for data validation."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, min_rows: int = 1, min_cols: int = 1) -> Dict[str, Any]:
        """Validate pandas DataFrame."""
        validation_result = {
            "valid": True,
            "issues": [],
            "info": {
                "shape": df.shape,
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "null_counts": df.isnull().sum().to_dict()
            }
        }
        
        # Check dimensions
        if df.shape[0] < min_rows:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Too few rows: {df.shape[0]} < {min_rows}")
        
        if df.shape[1] < min_cols:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Too few columns: {df.shape[1]} < {min_cols}")
        
        # Check for all-null columns
        all_null_cols = df.columns[df.isnull().all()].tolist()
        if all_null_cols:
            validation_result["issues"].append(f"All-null columns: {all_null_cols}")
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            validation_result["issues"].append(f"Duplicate columns: {duplicate_cols}")
        
        return validation_result
    
    @staticmethod
    def check_normalization_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Check quality of z-score normalization."""
        values = df.values.flatten()
        values = values[~pd.isna(values)]  # Remove NaN values
        
        if len(values) == 0:
            return {"valid": False, "reason": "No valid values found"}
        
        mean_val = values.mean()
        std_val = values.std()
        
        quality_check = {
            "mean": float(mean_val),
            "std": float(std_val),
            "min": float(values.min()),
            "max": float(values.max()),
            "is_well_normalized": abs(mean_val) < 0.1 and 0.8 < std_val < 1.2,
            "mean_centered": abs(mean_val) < 0.1,
            "unit_variance": 0.8 < std_val < 1.2
        }
        
        return quality_check


# Async setup function for server initialization
async def setup_server() -> None:
    """Setup function called during server initialization."""
    # Create temp directories
    temp_base = Path(tempfile.gettempdir()) / "droma_mcp"
    (temp_base / "exports").mkdir(parents=True, exist_ok=True)
    (temp_base / "figures").mkdir(parents=True, exist_ok=True)
    
    # Clean up old files
    cleanup_temp_files(max_age_hours=24)
    
    print("DROMA MCP utility services initialized") 