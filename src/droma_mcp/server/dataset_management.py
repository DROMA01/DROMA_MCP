"""DROMA MCP server for dataset management operations."""

import os
from fastmcp import FastMCP, Context
from typing import Dict, Any, Optional
from pathlib import Path

from ..schema.dataset_management import (
    LoadDatasetModel,
    ListDatasetsModel,
    SetActiveDatasetModel,
    UnloadDatasetModel
)

# Create sub-MCP server for dataset management
dataset_management_mcp = FastMCP("DROMA-Dataset-Management")


def _get_database_path(request_db_path: Optional[str] = None) -> str:
    """Get database path from request or environment."""
    db_path = request_db_path or os.environ.get('DROMA_DB_PATH')
    
    if not db_path:
        raise RuntimeError("No database path configured. Set DROMA_DB_PATH environment variable or provide db_path")
    
    if not Path(db_path).exists():
        raise RuntimeError(f"Database file not found: {db_path}")
    
    return db_path


@dataset_management_mcp.tool()
async def load_dataset(
    ctx: Context,
    request: LoadDatasetModel
) -> Dict[str, Any]:
    """
    Load a DROMA dataset into memory.
    
    This loads datasets (like CCLE, gCSI) from the database into R memory
    
    Equivalent to R function: createDromaSetFromDatabase() or createMultiDromaSetFromDatabase()
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    try:
        # Get database path
        db_path = _get_database_path(request.db_path)
        
        await ctx.info(f"Loading dataset '{request.dataset_id}' of type '{request.dataset_type}' from {db_path}")
        
        # Load the dataset using DromaState method
        success = droma_state.load_dataset(
            dataset_id=request.dataset_id,
            db_path=db_path,
            dataset_type=request.dataset_type
        )
        
        if not success:
            return {
                "status": "error",
                "message": f"Failed to load dataset '{request.dataset_id}'"
            }
        
        # Set as active if requested
        if request.set_active:
            try:
                droma_state.set_active_dataset(request.dataset_id, request.dataset_type)
                active_msg = f" and set as active {request.dataset_type.lower()}"
            except Exception as e:
                active_msg = f" but failed to set as active: {e}"
        else:
            active_msg = ""
        
        # Get current state for confirmation
        datasets_info = droma_state.list_datasets()
        
        return {
            "status": "success",
            "dataset_id": request.dataset_id,
            "dataset_type": request.dataset_type,
            "message": f"Successfully loaded dataset '{request.dataset_id}'{active_msg}",
            "loaded_datasets": datasets_info,
            "active_dataset": droma_state.active_dataset,
            "active_multidataset": droma_state.active_multidataset
        }
        
    except Exception as e:
        await ctx.error(f"Error loading dataset '{request.dataset_id}': {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to load dataset '{request.dataset_id}': {str(e)}"
        }


@dataset_management_mcp.tool()
async def list_loaded_datasets(
    ctx: Context,
    request: ListDatasetsModel
) -> Dict[str, Any]:
    """
    List all datasets currently loaded in memory.
    
    Shows which datasets are available for use by data loading functions.
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    try:
        datasets_info = droma_state.list_datasets()
        
        result = {
            "status": "success",
            "datasets": datasets_info["datasets"],
            "multidatasets": datasets_info["multidatasets"],
            "active_dataset": droma_state.active_dataset,
            "active_multidataset": droma_state.active_multidataset,
            "total_loaded": len(datasets_info["datasets"]) + len(datasets_info["multidatasets"])
        }
        
        if request.include_details:
            # Add detailed information about each dataset
            result["dataset_details"] = {}
            
            # For regular datasets
            for dataset_id in datasets_info["datasets"]:
                result["dataset_details"][dataset_id] = {
                    "type": "DromaSet",
                    "r_object_name": droma_state.datasets[dataset_id],
                    "is_active": dataset_id == droma_state.active_dataset
                }
            
            # For multidatasets
            for dataset_id in datasets_info["multidatasets"]:
                result["dataset_details"][dataset_id] = {
                    "type": "MultiDromaSet", 
                    "r_object_name": droma_state.multidatasets[dataset_id],
                    "is_active": dataset_id == droma_state.active_multidataset
                }
        
        message = f"Found {result['total_loaded']} loaded datasets"
        if result['total_loaded'] > 0:
            message += f" ({len(datasets_info['datasets'])} DromaSets, {len(datasets_info['multidatasets'])} MultiDromaSets)"
        
        result["message"] = message
        
        await ctx.info(message)
        return result
        
    except Exception as e:
        await ctx.error(f"Error listing datasets: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to list datasets: {str(e)}"
        }


@dataset_management_mcp.tool()
async def set_active_dataset(
    ctx: Context,
    request: SetActiveDatasetModel
) -> Dict[str, Any]:
    """
    Set the active dataset for subsequent operations.
    
    The active dataset is used by data loading functions when no specific dataset is provided.
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    try:
        # Set the active dataset
        droma_state.set_active_dataset(request.dataset_id, request.dataset_type)
        
        await ctx.info(f"Set '{request.dataset_id}' as active {request.dataset_type.lower()}")
        
        return {
            "status": "success",
            "dataset_id": request.dataset_id,
            "dataset_type": request.dataset_type,
            "message": f"Successfully set '{request.dataset_id}' as active {request.dataset_type.lower()}",
            "active_dataset": droma_state.active_dataset,
            "active_multidataset": droma_state.active_multidataset
        }
        
    except Exception as e:
        await ctx.error(f"Error setting active dataset: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to set active dataset '{request.dataset_id}': {str(e)}"
        }


@dataset_management_mcp.tool()
async def unload_dataset(
    ctx: Context,
    request: UnloadDatasetModel
) -> Dict[str, Any]:
    """
    Unload a dataset from memory.
    
    This removes the dataset from memory to free up resources.
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    try:
        # Check if dataset exists
        datasets_info = droma_state.list_datasets()
        
        if request.dataset_type == "DromaSet":
            if request.dataset_id not in datasets_info["datasets"]:
                return {
                    "status": "warning",
                    "message": f"Dataset '{request.dataset_id}' is not loaded"
                }
            
            # Remove from R environment if possible
            if droma_state.r is not None:
                r_object_name = droma_state.datasets[request.dataset_id]
                try:
                    droma_state.r(f"rm({r_object_name})")
                except:
                    pass  # Ignore R cleanup errors
            
            # Remove from state
            del droma_state.datasets[request.dataset_id]
            
            # Clear active dataset if this was it
            if droma_state.active_dataset == request.dataset_id:
                droma_state.active_dataset = None
                
        else:  # MultiDromaSet
            if request.dataset_id not in datasets_info["multidatasets"]:
                return {
                    "status": "warning",
                    "message": f"MultiDataset '{request.dataset_id}' is not loaded"
                }
            
            # Remove from R environment if possible
            if droma_state.r is not None:
                r_object_name = droma_state.multidatasets[request.dataset_id]
                try:
                    droma_state.r(f"rm({r_object_name})")
                except:
                    pass  # Ignore R cleanup errors
            
            # Remove from state
            del droma_state.multidatasets[request.dataset_id]
            
            # Clear active multidataset if this was it
            if droma_state.active_multidataset == request.dataset_id:
                droma_state.active_multidataset = None
        
        await ctx.info(f"Unloaded dataset '{request.dataset_id}'")
        
        # Get updated state
        updated_info = droma_state.list_datasets()
        
        return {
            "status": "success",
            "dataset_id": request.dataset_id,
            "dataset_type": request.dataset_type,
            "message": f"Successfully unloaded dataset '{request.dataset_id}'",
            "remaining_datasets": updated_info,
            "active_dataset": droma_state.active_dataset,
            "active_multidataset": droma_state.active_multidataset
        }
        
    except Exception as e:
        await ctx.error(f"Error unloading dataset: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to unload dataset '{request.dataset_id}': {str(e)}"
        } 