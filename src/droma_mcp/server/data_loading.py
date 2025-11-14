"""DROMA MCP server for data loading operations."""

from fastmcp import FastMCP, Context
from typing import Dict, Optional, Any, Union
import pandas as pd

from ..schema.data_loading import (
    LoadMolecularProfilesModel,
    LoadTreatmentResponseModel,
    MultiProjectMolecularProfilesModel,
    MultiProjectTreatmentResponseModel
)

# Create sub-MCP server for data loading
data_loading_mcp = FastMCP("DROMA-Data-Loading")


def _convert_r_to_python(r_result) -> Union[pd.DataFrame, Dict[str, Any], list]:
    """Convert R result to Python data structures."""
    try:
        from rpy2.robjects import pandas2ri, default_converter
        from rpy2.robjects.conversion import localconverter
        import numpy as np
        
        def _extract_r_names(r_obj):
            """Extract row and column names from R object."""
            try:
                rownames = list(r_obj.rownames) if hasattr(r_obj, 'rownames') and r_obj.rownames else None
                colnames = list(r_obj.colnames) if hasattr(r_obj, 'colnames') and r_obj.colnames else None
                return rownames, colnames
            except:
                return None, None
        
        # Use localconverter for pandas conversion
        with localconverter(default_converter + pandas2ri.converter):
            # Check if it's a list (multi-project case)
            if hasattr(r_result, 'rclass') and 'list' in r_result.rclass:
                # Handle list of data frames (multi-project results)
                result_list = []
                for i, item in enumerate(r_result):
                    if hasattr(item, 'rclass') and ('matrix' in item.rclass or 'data.frame' in item.rclass):
                        # Convert each data frame in the list
                        pandas_df = pandas2ri.rpy2py(item)
                        # Ensure it's always a DataFrame (not Series or numpy array)
                        if isinstance(pandas_df, pd.Series):
                            pandas_df = pandas_df.to_frame()
                        elif isinstance(pandas_df, np.ndarray):
                            # Extract names from R object before converting
                            rownames, colnames = _extract_r_names(item)
                            pandas_df = pd.DataFrame(pandas_df, index=rownames, columns=colnames)
                        result_list.append(pandas_df)
                    else:
                        # Keep non-dataframe items as is
                        result_list.append({"r_object": str(item), "type": str(type(item))})
                return result_list
                
            # Check if it's a single matrix or data.frame
            elif hasattr(r_result, 'rclass') and ('matrix' in r_result.rclass or 'data.frame' in r_result.rclass):
                # Convert R matrix or data.frame to pandas DataFrame
                pandas_df = pandas2ri.rpy2py(r_result)
                # Ensure it's always a DataFrame (not Series or numpy array)
                if isinstance(pandas_df, pd.Series):
                    pandas_df = pandas_df.to_frame()
                elif isinstance(pandas_df, np.ndarray):
                    # Extract names from R object before converting
                    rownames, colnames = _extract_r_names(r_result)
                    pandas_df = pd.DataFrame(pandas_df, index=rownames, columns=colnames)
                return pandas_df
            else:
                # Return as dictionary for other R objects
                return {"r_object": str(r_result), "type": str(type(r_result))}
            
    except Exception as e:
        print(f"Error converting R result: {e}")
        return {"error": str(e), "r_result": str(r_result)}


@data_loading_mcp.tool()
async def load_molecular_profiles(
    ctx: Context,
    request: LoadMolecularProfilesModel
) -> Dict[str, Any]:
    """
    Load molecular profiles with optional z-score normalization.
    
    Equivalent to R function: loadMolecularProfiles() or loadMultiProjectMolecularProfiles()
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    # Check if dataset exists (try both DromaSet and MultiDromaSet)
    dataset_r_name = droma_state.get_dataset(request.dataset_name)
    is_multi = False
    if not dataset_r_name:
        dataset_r_name = droma_state.get_multidataset(request.dataset_name)
        is_multi = True
        if not dataset_r_name:
            return {
                "status": "error",
                "message": f"Dataset {request.dataset_name} not found. Please load it first."
            }
    
    try:
        # Build R command
        features_str = "NULL"
        if request.select_features:
            features_str = 'c("' + '", "'.join(request.select_features) + '")'
        
        samples_str = "NULL"
        if request.samples:
            samples_str = 'c("' + '", "'.join(request.samples) + '")'
        
        # Use appropriate R function based on dataset type
        if is_multi:
            r_command = f'''
            result <- loadMultiProjectMolecularProfiles(
                {dataset_r_name},
                feature_type = "{request.feature_type.value}",
                select_features = {features_str},
                projects = NULL,
                overlap_only = FALSE,
                data_type = "{request.data_type.value}",
                tumor_type = "{request.tumor_type}",
                zscore = {str(request.zscore).upper()},
                format = "{request.format}"
            )
            '''
        else:
            r_command = f'''
            result <- loadMolecularProfiles(
                {dataset_r_name},
                feature_type = "{request.feature_type.value}",
                select_features = {features_str},
                samples = {samples_str},
                return_data = TRUE,
                data_type = "{request.data_type.value}",
                tumor_type = "{request.tumor_type}",
                zscore = {str(request.zscore).upper()},
                format = "{request.format}"
            )
            '''
        
        await ctx.info(f"Executing R command for molecular profiles: {request.feature_type.value}")
        
        # Execute R command
        droma_state.r(r_command)
        r_result = droma_state.r('result')
        
        # Convert result to Python
        python_result = _convert_r_to_python(r_result)
        
        # Cache the result
        cache_key = f"mol_profiles_{request.dataset_name}_{request.feature_type.value}"
        droma_state.cache_data(cache_key, python_result, {
            "feature_type": request.feature_type.value,
            "zscore_normalized": request.zscore,
            "select_features": request.select_features,
            "samples": request.samples,
            "data_type": request.data_type.value,
            "tumor_type": request.tumor_type,
            "format": request.format
        })
        
        # Get basic stats
        if isinstance(python_result, pd.DataFrame):
            stats = {
                "shape": python_result.shape,
                "features_count": len(python_result.index),
                "samples_count": len(python_result.columns),
                "has_missing_values": bool(python_result.isnull().any().any())
            }
        else:
            stats = {"result_type": "non_matrix"}
        
        await ctx.info(f"Successfully loaded molecular profiles: {stats}")
        
        return {
            "status": "success",
            "cache_key": cache_key,
            "feature_type": request.feature_type.value,
            "zscore_normalized": request.zscore,
            "stats": stats,
            "message": f"Loaded {request.feature_type.value} data for {request.dataset_name}"
        }
        
    except Exception as e:
        await ctx.error(f"Error loading molecular profiles: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to load molecular profiles: {str(e)}"
        }


@data_loading_mcp.tool()
async def load_treatment_response(
    ctx: Context,
    request: LoadTreatmentResponseModel
) -> Dict[str, Any]:
    """
    Load treatment response data with optional z-score normalization.
    
    Equivalent to R function: loadTreatmentResponse() or loadMultiProjectTreatmentResponse()
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    # Check if dataset exists (try both DromaSet and MultiDromaSet)
    dataset_r_name = droma_state.get_dataset(request.dataset_name)
    is_multi = False
    if not dataset_r_name:
        dataset_r_name = droma_state.get_multidataset(request.dataset_name)
        is_multi = True
        if not dataset_r_name:
            return {
                "status": "error",
                "message": f"Dataset {request.dataset_name} not found. Please load it first."
            }
    
    try:
        # Build R command
        drugs_str = "NULL"
        if request.select_drugs:
            drugs_str = 'c("' + '", "'.join(request.select_drugs) + '")'
        
        samples_str = "NULL"
        if request.samples:
            samples_str = 'c("' + '", "'.join(request.samples) + '")'
        
        # Use appropriate R function based on dataset type
        if is_multi:
            r_command = f'''
            result <- loadMultiProjectTreatmentResponse(
                {dataset_r_name},
                select_drugs = {drugs_str},
                projects = NULL,
                overlap_only = FALSE,
                data_type = "{request.data_type.value}",
                tumor_type = "{request.tumor_type}",
                zscore = {str(request.zscore).upper()}
            )
            '''
        else:
            r_command = f'''
            result <- loadTreatmentResponse(
                {dataset_r_name},
                select_drugs = {drugs_str},
                samples = {samples_str},
                return_data = TRUE,
                data_type = "{request.data_type.value}",
                tumor_type = "{request.tumor_type}",
                zscore = {str(request.zscore).upper()}
            )
            '''
        
        await ctx.info(f"Executing R command for treatment response data")
        
        # Execute R command
        droma_state.r(r_command)
        r_result = droma_state.r('result')
        
        # Convert result to Python
        python_result = _convert_r_to_python(r_result)
        
        # Cache the result
        cache_key = f"treatment_response_{request.dataset_name}"
        droma_state.cache_data(cache_key, python_result, {
            "select_drugs": request.select_drugs,
            "samples": request.samples,
            "zscore_normalized": request.zscore,
            "data_type": request.data_type.value,
            "tumor_type": request.tumor_type
        })
        
        # Get basic stats
        if isinstance(python_result, pd.DataFrame):
            stats = {
                "shape": python_result.shape,
                "drugs_count": len(python_result.index),
                "samples_count": len(python_result.columns),
                "has_missing_values": bool(python_result.isnull().any().any())
            }
        else:
            stats = {"result_type": "non_matrix"}
        
        await ctx.info(f"Successfully loaded treatment response data: {stats}")
        
        return {
            "status": "success",
            "cache_key": cache_key,
            "select_drugs": request.select_drugs,
            "zscore_normalized": request.zscore,
            "stats": stats,
            "message": f"Loaded treatment response data for {request.dataset_name}"
        }
        
    except Exception as e:
        await ctx.error(f"Error loading treatment response: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to load treatment response: {str(e)}"
        }


@data_loading_mcp.tool()
async def load_multi_project_molecular_profiles(
    ctx: Context,
    request: MultiProjectMolecularProfilesModel
) -> Dict[str, Any]:
    """
    Load multi-project molecular profiles with optional z-score normalization.
    
    Equivalent to R function: loadMultiProjectMolecularProfiles()
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    # Check if multidataset exists
    multidataset_r_name = droma_state.get_multidataset(request.multidromaset_id)
    if not multidataset_r_name:
        return {
            "status": "error",
            "message": f"MultiDataset {request.multidromaset_id} not found. Please load it first."
        }
    
    try:
        # Build R command
        features_str = "NULL"
        if request.select_features:
            features_str = 'c("' + '", "'.join(request.select_features) + '")'
        
        r_command = f'''
        result <- loadMultiProjectMolecularProfiles(
            {multidataset_r_name},
            feature_type = "{request.feature_type.value}",
            select_features = {features_str},
            projects = NULL,
            overlap_only = {str(request.overlap_only).upper()},
            data_type = "{request.data_type.value}",
            tumor_type = "{request.tumor_type}",
            zscore = {str(request.zscore).upper()},
            format = "{request.format}"
        )
        '''
        
        await ctx.info(f"Executing R command for multi-project molecular profiles")
        
        # Execute R command
        droma_state.r(r_command)
        r_result = droma_state.r('result')
        
        # Convert result to Python (should be a list of matrices)
        python_result = _convert_r_to_python(r_result)
        
        # Cache the result
        cache_key = f"multi_mol_profiles_{request.multidromaset_id}_{request.feature_type.value}"
        droma_state.cache_data(cache_key, python_result, {
            "feature_type": request.feature_type.value,
            "zscore_normalized": request.zscore,
            "select_features": request.select_features,
            "overlap_only": request.overlap_only,
            "data_type": request.data_type.value,
            "tumor_type": request.tumor_type,
            "format": request.format
        })
        
        # Get basic stats for multi-project data
        if isinstance(python_result, list):
            project_stats = {}
            for i, data in enumerate(python_result):
                if isinstance(data, pd.DataFrame):
                    project_name = f"project_{i+1}"  # or get actual project names if available
                    project_stats[project_name] = {
                        "shape": data.shape,
                        "features_count": len(data.index),
                        "samples_count": len(data.columns)
                    }
            stats = {"projects": project_stats, "total_projects": len(python_result)}
        else:
            stats = {"result_type": "unknown"}
        
        await ctx.info(f"Successfully loaded multi-project molecular profiles: {stats}")
        
        return {
            "status": "success",
            "cache_key": cache_key,
            "feature_type": request.feature_type.value,
            "zscore_normalized": request.zscore,
            "overlap_only": request.overlap_only,
            "stats": stats,
            "message": f"Loaded multi-project {request.feature_type.value} data"
        }
        
    except Exception as e:
        await ctx.error(f"Error loading multi-project molecular profiles: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to load multi-project molecular profiles: {str(e)}"
        }


@data_loading_mcp.tool()
async def load_multi_project_treatment_response(
    ctx: Context,
    request: MultiProjectTreatmentResponseModel
) -> Dict[str, Any]:
    """
    Load multi-project treatment response data with optional z-score normalization.
    
    Equivalent to R function: loadMultiProjectTreatmentResponse()
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    # Check if multidataset exists
    multidataset_r_name = droma_state.get_multidataset(request.multidromaset_id)
    if not multidataset_r_name:
        return {
            "status": "error",
            "message": f"MultiDataset {request.multidromaset_id} not found. Please load it first."
        }
    
    try:
        # Build R command
        drugs_str = "NULL"
        if request.select_drugs:
            drugs_str = 'c("' + '", "'.join(request.select_drugs) + '")'
        
        r_command = f'''
        result <- loadMultiProjectTreatmentResponse(
            {multidataset_r_name},
            select_drugs = {drugs_str},
            projects = NULL,
            overlap_only = {str(request.overlap_only).upper()},
            data_type = "{request.data_type.value}",
            tumor_type = "{request.tumor_type}",
            zscore = {str(request.zscore).upper()}
        )
        '''
        
        await ctx.info(f"Executing R command for multi-project treatment response")
        
        # Execute R command
        droma_state.r(r_command)
        r_result = droma_state.r('result')
        
        # Convert result to Python
        python_result = _convert_r_to_python(r_result)
        
        # Cache the result
        cache_key = f"multi_treatment_response_{request.multidromaset_id}"
        droma_state.cache_data(cache_key, python_result, {
            "select_drugs": request.select_drugs,
            "zscore_normalized": request.zscore,
            "overlap_only": request.overlap_only,
            "data_type": request.data_type.value,
            "tumor_type": request.tumor_type
        })
        
        # Get basic stats for multi-project data
        if isinstance(python_result, list):
            project_stats = {}
            for i, data in enumerate(python_result):
                if isinstance(data, pd.DataFrame):
                    project_name = f"project_{i+1}"  # or get actual project names if available
                    project_stats[project_name] = {
                        "shape": data.shape,
                        "drugs_count": len(data.index),
                        "samples_count": len(data.columns)
                    }
            stats = {"projects": project_stats, "total_projects": len(python_result)}
        else:
            stats = {"result_type": "unknown"}
        
        await ctx.info(f"Successfully loaded multi-project treatment response: {stats}")
        
        return {
            "status": "success",
            "cache_key": cache_key,
            "select_drugs": request.select_drugs,
            "zscore_normalized": request.zscore,
            "overlap_only": request.overlap_only,
            "stats": stats,
            "message": f"Loaded multi-project treatment response data"
        }
        
    except Exception as e:
        await ctx.error(f"Error loading multi-project treatment response: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to load multi-project treatment response: {str(e)}"
        }


@data_loading_mcp.tool()
async def get_cached_data_info(
    ctx: Context,
    cache_key: Optional[str] = None
) -> Dict[str, Any]:
    """Get information about cached data."""
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    if cache_key:
        # Get info for specific cached data
        cached_entry = droma_state.data_cache.get(cache_key)
        if not cached_entry:
            return {
                "status": "error",
                "message": f"No cached data found for key: {cache_key}"
            }
        
        data = cached_entry['data']
        metadata = cached_entry.get('metadata', {})
        timestamp = cached_entry.get('timestamp')
        
        data_info = {
            "cache_key": cache_key,
            "timestamp": str(timestamp),
            "metadata": metadata,
            "data_type": str(type(data)),
        }
        
        if isinstance(data, pd.DataFrame):
            data_info.update({
                "shape": data.shape,
                "columns": list(data.columns),
                "index_count": len(data.index)
            })
        
        return {
            "status": "success",
            "data_info": data_info
        }
    else:
        # List all cached data
        cache_summary = {}
        for key, entry in droma_state.data_cache.items():
            cache_summary[key] = {
                "timestamp": str(entry.get('timestamp')),
                "data_type": str(type(entry['data'])),
                "metadata": entry.get('metadata', {})
            }
        
        return {
            "status": "success",
            "cached_items": cache_summary,
            "total_items": len(cache_summary)
        }


@data_loading_mcp.tool()
async def view_cached_data(
    ctx: Context,
    cache_key: str,
    preview_size: int = 5,
    features: Optional[list[str]] = None,
    samples: Optional[list[str]] = None
) -> Dict[str, Any]:
    """
    Preview cached data with statistics and sample values.
    Always shows a small preview regardless of data size.
    
    Args:
        cache_key: The cache key to retrieve data
        preview_size: Number of rows/columns to preview (default: 5, max: 10)
        features: Optional list of specific features/rows to view
        samples: Optional list of specific samples/columns to view
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    # Get cached entry with metadata
    cached_entry = droma_state.data_cache.get(cache_key)
    if not cached_entry:
        return {
            "status": "error",
            "message": f"No cached data found for key: {cache_key}"
        }
    
    cached_data = cached_entry['data']
    metadata = cached_entry.get('metadata', {})
    
    try:
        import numpy as np
        
        # Convert numpy array to DataFrame if needed
        if isinstance(cached_data, np.ndarray):
            index_names = metadata.get('select_features')
            cached_data = pd.DataFrame(cached_data, index=index_names)
        
        if isinstance(cached_data, pd.DataFrame):
            original_shape = cached_data.shape
            
            # Apply specific feature/sample filtering if requested
            display_data = cached_data
            if features:
                available_features = [f for f in features if f in display_data.index]
                if available_features:
                    display_data = display_data.loc[available_features]
            
            if samples:
                available_samples = [s for s in samples if s in display_data.columns]
                if available_samples:
                    display_data = display_data[available_samples]
            
            # Limit preview size (max 10x10)
            preview_size = min(max(preview_size, 1), 10)
            preview_data = display_data.head(preview_size).iloc[:, :preview_size]
            preview_dict = preview_data.to_dict(orient='split')
            
            # Calculate statistics on full data
            stats = {
                "min": float(cached_data.min().min()) if not cached_data.empty else None,
                "max": float(cached_data.max().max()) if not cached_data.empty else None,
                "mean": float(cached_data.mean().mean()) if not cached_data.empty else None,
                "missing_count": int(cached_data.isnull().sum().sum()),
                "missing_rate": f"{(cached_data.isnull().sum().sum() / cached_data.size * 100):.2f}%" if cached_data.size > 0 else "0%"
            }
            
            # Build response
            result = {
                "status": "preview",
                "cache_key": cache_key,
                "full_shape": original_shape,
                "preview_shape": preview_data.shape,
                "preview_data": {
                    "features": preview_dict['index'],
                    "samples": preview_dict['columns'],
                    "values": preview_dict['data']
                },
                "statistics": stats,
                "all_features": list(cached_data.index[:20]),  # Show first 20
                "all_samples": list(cached_data.columns[:20]),  # Show first 20
                "metadata": metadata
            }
            
            # Add recommendation for large datasets
            if original_shape[0] > 50 or original_shape[1] > 50:
                result["recommendation"] = (
                    f"Dataset is large ({original_shape[0]}×{original_shape[1]}). "
                    f"Use export_cached_data(cache_key='{cache_key}') to save the full dataset to a file."
                )
                result["message"] = f"Showing preview of {preview_data.shape[0]}×{preview_data.shape[1]} from full dataset {original_shape[0]}×{original_shape[1]}"
            else:
                result["message"] = f"Showing {preview_data.shape[0]}×{preview_data.shape[1]} preview"
            
            await ctx.info(f"Previewed data: {original_shape[0]}×{original_shape[1]}")
            return result
            
        else:
            return {
                "status": "error",
                "message": f"Cached data is not a DataFrame (type: {type(cached_data)})"
            }
            
    except Exception as e:
        await ctx.error(f"Error viewing data: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to view data: {str(e)}"
        }


@data_loading_mcp.tool()
async def export_cached_data(
    ctx: Context,
    cache_key: str,
    file_format: str = "csv",
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """Export cached data to file."""
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    cached_data = droma_state.get_cached_data(cache_key)
    if cached_data is None:
        return {
            "status": "error",
            "message": f"No cached data found for key: {cache_key}"
        }
    
    try:
        # Use utility function for saving
        from ..util import save_analysis_result
        import numpy as np
        
        # Convert numpy array to DataFrame if needed
        if isinstance(cached_data, np.ndarray):
            cached_data = pd.DataFrame(cached_data)
        
        if isinstance(cached_data, pd.DataFrame):
            from ..util import EXPORTS
            export_id = save_analysis_result(cached_data, filename, file_format)
            
            # Get the actual file path
            file_path = EXPORTS.get(export_id, "")
            
            return {
                "status": "success",
                "export_id": export_id,
                "file_path": file_path,
                "filename": filename,
                "file_format": file_format,
                "data_shape": cached_data.shape,
                "message": f"Data exported successfully to: {file_path}"
            }
        else:
            return {
                "status": "error",
                "message": "Only pandas DataFrame or numpy array can be exported to structured files"
            }
            
    except Exception as e:
        await ctx.error(f"Error exporting data: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to export data: {str(e)}"
        } 