"""DROMA MCP server for data loading operations."""

from fastmcp import FastMCP, Context
from typing import Dict, Optional, Any, Union
import pandas as pd

from ..schema.data_loading import (
    LoadMolecularProfilesModel,
    LoadTreatmentResponseModel,
    MultiProjectMolecularProfilesModel,
    MultiProjectTreatmentResponseModel,
    ViewCachedDataModel,
    ExportCachedDataModel,
    ViewExportedDataModel
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
        # Check if this cache_key has been exported - recommend view_exported_data instead
        export_id = ctx.get_state(f"export:{cache_key}")
        if export_id:
            return {
                "status": "redirect",
                "message": f"This data was already exported as export_id '{export_id}'. Use view_exported_data(export_id='{export_id}') instead to avoid context pollution.",
                "export_id": export_id,
                "recommendation": f"view_exported_data(export_id='{export_id}', full_data=False)"
            }
        
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
    request: ViewCachedDataModel
) -> Dict[str, Any]:
    """
    Preview cached data with statistics and sample values.
    Always shows a small preview regardless of data size.
    """
    # Check if this cache_key has been exported - recommend view_exported_data instead
    export_id = ctx.get_state(f"export:{request.cache_key}")
    if export_id:
        return {
            "status": "redirect",
            "message": f"This data was already exported as export_id '{export_id}'. Use view_exported_data(export_id='{export_id}') instead to avoid context pollution.",
            "export_id": export_id,
            "recommendation": f"view_exported_data(export_id='{export_id}', full_data=True)"
        }
    
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    # Get cached entry with metadata
    cached_entry = droma_state.data_cache.get(request.cache_key)
    if not cached_entry:
        return {
            "status": "error",
            "message": f"No cached data found for key: {request.cache_key}"
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
            if request.features:
                available_features = [f for f in request.features if f in display_data.index]
                if available_features:
                    display_data = display_data.loc[available_features]
            
            if request.samples:
                available_samples = [s for s in request.samples if s in display_data.columns]
                if available_samples:
                    display_data = display_data[available_samples]
            
            # Limit preview size (max 10x10)
            preview_size = min(max(request.preview_size, 1), 10)
            preview_data = display_data.head(preview_size).iloc[:, :preview_size]
            preview_dict = preview_data.to_dict(orient='split')
            
            # Calculate statistics on display data (filtered by features if specified)
            stats = _calculate_feature_stats(display_data)
            
            # Build response
            result = {
                "status": "preview",
                "cache_key": request.cache_key,
                "full_shape": original_shape,
                "preview_shape": preview_data.shape,
                "preview_data": {
                    "features": preview_dict['index'],
                    "samples": preview_dict['columns'],
                    "values": preview_dict['data']
                },
                "statistics": stats,
                "all_features": list(cached_data.index[:10]),  # Show first 10
                "all_samples": list(cached_data.columns[:10]),  # Show first 10
                "metadata": metadata
            }
            
            # Add recommendation for large datasets
            if original_shape[0] > 50 or original_shape[1] > 50:
                result["recommendation"] = (
                    f"Dataset is large ({original_shape[0]}×{original_shape[1]}). "
                    f"Use export_cached_data(cache_key='{request.cache_key}') to save the full dataset to a file."
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
    request: ExportCachedDataModel
) -> Dict[str, Any]:
    """Export cached data to file with optional memory release and auto preview."""
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    cached_data = droma_state.get_cached_data(request.cache_key)
    if cached_data is None:
        return {
            "status": "error",
            "message": f"No cached data found for key: {request.cache_key}"
        }
    
    try:
        # Use utility function for saving
        from ..util import save_analysis_result, EXPORTS, format_data_size
        import numpy as np
        from pathlib import Path
        
        # Convert numpy array to DataFrame if needed
        if isinstance(cached_data, np.ndarray):
            cached_data = pd.DataFrame(cached_data)
        
        if isinstance(cached_data, pd.DataFrame):
            export_id = save_analysis_result(cached_data, request.filename, request.file_format)
            
            # Get the actual file path
            file_path = EXPORTS.get(export_id, "")
            
            # Release memory if requested
            memory_released = False
            if request.release_memory and request.cache_key in droma_state.data_cache:
                del droma_state.data_cache[request.cache_key]
                memory_released = True
                await ctx.info(f"Released memory for cache_key: {request.cache_key}")
            
            # Generate preview from exported file
            preview_data = _get_export_preview(file_path, request.file_format, preview_size=5)
            
            result = {
                "status": "success",
                "export_id": export_id,
                "file_path": file_path,
                "filename": request.filename or Path(file_path).name,
                "file_format": request.file_format,
                "data_shape": cached_data.shape,
                "memory_released": memory_released,
                "message": f"Data exported successfully to: {file_path}"
            }
            
            # Add preview if available
            if preview_data:
                result["preview"] = preview_data
                result["message"] += " (with preview)"
            
            # Track export in context state to help guide future queries
            ctx.set_state(f"export:{request.cache_key}", export_id)
            
            return result
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


def _calculate_feature_stats(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate statistics for DataFrame.
    If multiple features exist, calculate stats per feature.
    Otherwise, calculate overall stats.
    """
    if data.empty:
        return {"overall": {"min": None, "max": None, "mean": None, "missing_count": 0, "missing_rate": "0%"}}
    
    # Calculate overall missing values
    total_missing = int(data.isnull().sum().sum())
    total_elements = data.size
    missing_rate = f"{(total_missing / total_elements * 100):.2f}%" if total_elements > 0 else "0%"
    
    # If more than one feature, calculate per-feature stats
    if len(data.index) > 1:
        feature_stats = {}
        for feature in data.index:
            feature_data = data.loc[feature]
            feature_missing = int(feature_data.isnull().sum())
            feature_size = len(feature_data)
            feature_stats[str(feature)] = {
                "min": float(feature_data.min()) if not feature_data.isnull().all() else None,
                "max": float(feature_data.max()) if not feature_data.isnull().all() else None,
                "mean": float(feature_data.mean()) if not feature_data.isnull().all() else None,
                "missing_count": feature_missing,
                "missing_rate": f"{(feature_missing / feature_size * 100):.2f}%" if feature_size > 0 else "0%"
            }
        return {
            "by_feature": feature_stats,
            "overall_missing": {"missing_count": total_missing, "missing_rate": missing_rate}
        }
    else:
        # Single feature or overall stats
        return {
            "overall": {
                "min": float(data.min().min()),
                "max": float(data.max().max()),
                "mean": float(data.mean().mean()),
                "missing_count": total_missing,
                "missing_rate": missing_rate
            }
        }


def _get_export_preview(file_path: str, file_format: str, preview_size: int = 5) -> Optional[Dict[str, Any]]:
    """Helper function to generate preview from exported file."""
    try:
        from pathlib import Path
        filepath = Path(file_path)
        
        if not filepath.exists():
            return None
        
        # Read file based on format
        if file_format == "csv":
            data = pd.read_csv(filepath, index_col=0, nrows=preview_size + 5)
        elif file_format in ["xlsx", "xls", "excel"]:
            data = pd.read_excel(filepath, index_col=0, nrows=preview_size + 5)
        elif file_format == "json":
            data = pd.read_json(filepath, orient='split')
        else:
            return None
        
        # Limit preview
        preview_data = data.head(preview_size).iloc[:, :preview_size]
        preview_dict = preview_data.to_dict(orient='split')
        
        return {
            "shape": preview_data.shape,
            "features": preview_dict['index'][:preview_size],
            "samples": preview_dict['columns'][:preview_size],
            "values": preview_dict['data']
        }
    except Exception:
        return None


@data_loading_mcp.tool()
async def view_exported_data(
    ctx: Context,
    request: ViewExportedDataModel
) -> Dict[str, Any]:
    """
    Preview exported data file without loading into memory cache.
    Useful for verifying export results after memory has been released.
    """
    from ..util import EXPORTS, format_data_size
    from pathlib import Path
    import numpy as np
    
    # Check if export exists
    if request.export_id not in EXPORTS:
        return {
            "status": "error",
            "message": f"Export not found: {request.export_id}. Available exports: {list(EXPORTS.keys())}"
        }
    
    filepath = Path(EXPORTS[request.export_id])
    
    # Check if file exists
    if not filepath.exists():
        return {
            "status": "error",
            "message": f"Export file not found on disk: {filepath}"
        }
    
    try:
        # Determine file format
        file_format = filepath.suffix[1:]  # Remove dot
        
        # Read file - full or preview based on request
        if request.full_data:
            # Load complete data
            if file_format == "csv":
                data = pd.read_csv(filepath, index_col=0)
            elif file_format in ["xlsx", "xls"]:
                data = pd.read_excel(filepath, index_col=0)
            elif file_format == "json":
                data = pd.read_json(filepath, orient='split')
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported file format: {file_format}"
                }
            await ctx.info(f"Loading full data from {filepath.name}")
        else:
            # Load preview only (for efficiency)
            preview_rows = min(max(request.preview_size, 1), 50) + 20
            if file_format == "csv":
                data = pd.read_csv(filepath, index_col=0, nrows=preview_rows)
            elif file_format in ["xlsx", "xls"]:
                data = pd.read_excel(filepath, index_col=0, nrows=preview_rows)
            elif file_format == "json":
                data = pd.read_json(filepath, orient='split')
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported file format: {file_format}"
                }
        
        original_shape = data.shape
        
        # Apply filtering if requested
        display_data = data
        if request.features:
            available_features = [f for f in request.features if f in display_data.index]
            if available_features:
                display_data = display_data.loc[available_features]
        
        if request.samples:
            available_samples = [s for s in request.samples if s in display_data.columns]
            if available_samples:
                display_data = display_data[available_samples]
        
        # Limit display size only if not requesting full data
        if request.full_data:
            preview_data = display_data
        else:
            preview_size = min(max(request.preview_size, 1), 50)
            preview_data = display_data.head(preview_size).iloc[:, :preview_size]
        
        preview_dict = preview_data.to_dict(orient='split')
        
        # Calculate statistics on display data (filtered by features if specified)
        stats = _calculate_feature_stats(display_data)
        
        # Get file info
        file_stats = filepath.stat()
        
        result = {
            "status": "full_data" if request.full_data else "preview",
            "export_id": request.export_id,
            "file_path": str(filepath),
            "file_format": file_format,
            "file_size_bytes": file_stats.st_size,
            "file_size_readable": format_data_size(file_stats.st_size),
            "full_shape": original_shape,
            "display_shape": preview_data.shape,
            "data": {
                "features": preview_dict['index'],
                "samples": preview_dict['columns'],
                "values": preview_dict['data']
            },
            "statistics": stats,
            "is_complete": request.full_data,
            "message": f"Loaded complete data {preview_data.shape[0]}×{preview_data.shape[1]}" if request.full_data 
                      else f"Showing {preview_data.shape[0]}×{preview_data.shape[1]} preview (use full_data=True for complete data)"
        }
        
        await ctx.info(f"{'Loaded full' if request.full_data else 'Previewed'} exported data: {filepath.name}")
        return result
        
    except Exception as e:
        await ctx.error(f"Error viewing exported data: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to view exported data: {str(e)}"
        } 