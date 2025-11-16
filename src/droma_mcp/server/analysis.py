"""DROMA MCP server for analysis operations."""

from fastmcp import FastMCP, Context
from typing import Dict, Any, List, Optional, Union
import tempfile
from pathlib import Path

# Create sub-MCP server for analysis
analysis_mcp = FastMCP("DROMA-Analysis")


@analysis_mcp.tool()
async def analyze_drug_omic_pair(
    ctx: Context,
    dataset_name: str,
    feature_type: str,
    select_features: str,
    select_drugs: str,
    data_type: str = "all",
    tumor_type: str = "all",
    overlap_only: bool = False,
    merged_enabled: bool = True,
    meta_enabled: bool = True,
    zscore: bool = True,
    data_type_anno: Optional[str] = None,
    save_plots: bool = True,
    plot_width: float = 10.0,
    plot_height: float = 6.0,
    display_mode: str = "inline"
) -> Dict[str, Any]:
    """
    Analyze associations between a drug and an omic feature.
    
    Equivalent to R function: analyzeDrugOmicPair()
    
    This function analyzes the relationship between drug sensitivity and molecular features
    (e.g., gene expression, mutations, copy number variations). It supports both continuous
    features (mRNA, methylation, CNV, protein) and discrete features (mutations, fusions).
    
    For MultiDromaSet objects with multiple studies, it can:
    - Create individual plots for each study
    - Create a merged plot combining all studies (if merged_enabled=True)
    - Perform meta-analysis across studies (if meta_enabled=True)
    
    Args:
        ctx: FastMCP context
        dataset_name: Dataset name (e.g., 'CCLE', 'gCSI') for DromaSet or MultiDromaSet
        feature_type: Type of omics data ('mRNA', 'cnv', 'meth', 'proteinrppa', 'proteinms', 
                     'mutation_gene', 'mutation_site', 'fusion')
        select_features: Name of the specific omics feature (e.g., 'ABCB1', 'TP53')
        select_drugs: Name of the drug to analyze (e.g., 'Paclitaxel')
        data_type: Filter by data type ('all', 'CellLine', 'PDC', 'PDO', 'PDX')
        tumor_type: Filter by tumor type ('all' or specific tumor types)
        overlap_only: For MultiDromaSet, use only overlapping samples
        merged_enabled: Whether to create a merged dataset from all studies
        meta_enabled: Whether to perform meta-analysis (requires multiple studies)
        zscore: Whether to apply z-score normalization (recommended for merging)
        data_type_anno: Optional annotation for plot titles (e.g., 'Cell lines')
        save_plots: Whether to save plots to files
        plot_width: Plot width in inches
        plot_height: Plot height in inches
        display_mode: How to display plots: 'inline' (show in chat, default) or 'link' (provide download link only)
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - message: Description of the result
        - analysis_type: Type of analysis performed ("continuous" or "discrete")
        - plots: Information about individual study plots (if applicable)
        - merged_plot: Information about merged plot (if merged_enabled=True)
        - meta_analysis: Meta-analysis results (if meta_enabled=True and multiple studies)
        - statistics: Summary statistics for the analysis
        - data_cache_key: Key for accessing cached analysis data
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    # Check if dataset exists (try both DromaSet and MultiDromaSet)
    dataset_r_name = droma_state.get_dataset(dataset_name)
    is_multi = False
    if not dataset_r_name:
        dataset_r_name = droma_state.get_multidataset(dataset_name)
        is_multi = True
        if not dataset_r_name:
            return {
                "status": "error",
                "message": f"Dataset {dataset_name} not found. Please load it first using load_dataset tool."
            }
    
    try:
        # Build R command parameters
        data_type_anno_str = "NULL"
        if data_type_anno:
            data_type_anno_str = f'"{data_type_anno}"'
        
        # Execute analysis
        r_command = f'''
        # Perform drug-omic pair analysis
        analysis_result <- analyzeDrugOmicPair(
            dromaset_object = {dataset_r_name},
            feature_type = "{feature_type}",
            select_features = "{select_features}",
            select_drugs = "{select_drugs}",
            data_type = "{data_type}",
            tumor_type = "{tumor_type}",
            overlap_only = {str(overlap_only).upper()},
            merged_enabled = {str(merged_enabled).upper()},
            meta_enabled = {str(meta_enabled).upper()},
            zscore = {str(zscore).upper()},
            data_type_anno = {data_type_anno_str}
        )
        '''
        
        await ctx.info(f"Analyzing {feature_type}:{select_features} vs {select_drugs}")
        
        # Execute R command
        droma_state.r(r_command)
        
        # Initialize response
        response = {
            "status": "success",
            "message": f"Analysis completed for {select_features} vs {select_drugs}",
            "analysis_type": "continuous" if feature_type in ["mRNA", "cnv", "meth", "proteinrppa", "proteinms"] else "discrete",
            "parameters": {
                "dataset": dataset_name,
                "dataset_type": "MultiDromaSet" if is_multi else "DromaSet",
                "feature_type": feature_type,
                "feature": select_features,
                "drug": select_drugs,
                "data_type": data_type,
                "tumor_type": tumor_type,
                "zscore_normalized": zscore,
                "merged_enabled": merged_enabled,
                "meta_enabled": meta_enabled
            }
        }
        
        # Save plots if requested
        if save_plots:
            temp_dir = Path(tempfile.gettempdir()) / "droma_mcp_figures"
            temp_dir.mkdir(exist_ok=True)
            
            saved_plots = []
            inline_images = []
            
            # Save individual study plot(s)
            if_plot_exists = droma_state.r('!is.null(analysis_result$plot)')[0]
            if if_plot_exists:
                plot_name = f"drug_omic_pair_{select_features}_{select_drugs}_{dataset_name}_individual.png"
                plot_path = temp_dir / plot_name
                
                save_cmd = f'''
                library(ggplot2)
                ggsave(
                    filename = "{str(plot_path)}",
                    plot = analysis_result$plot,
                    width = {plot_width},
                    height = {plot_height},
                    dpi = 300
                )
                '''
                droma_state.r(save_cmd)
                
                if plot_path.exists():
                    from ..util import save_figure
                    fig_id = save_figure(plot_path, plot_name)

                    # Get server URL and generate figure URL if in HTTP mode
                    from ..util import get_server_url, get_figure_url
                    server_url = get_server_url()
                    figure_url = get_figure_url(fig_id) if server_url else None
                    figure_display_path = figure_url if figure_url else str(plot_path)

                    plot_info = {
                        "type": "individual",
                        "figure_id": fig_id,
                        "figure_path": str(plot_path),  # Keep local path for reference
                        "figure_url": figure_url,  # Add URL for HTTP mode
                        "figure_display_path": figure_display_path,  # Path/URL for display
                        "resource_uri": f"figure://{fig_id}"
                    }
                    
                    # Add to inline images list for inline mode
                    if display_mode == "inline":
                        inline_images.append({
                            "title": "Individual Study Plot",
                            "path": figure_display_path,
                            "url": figure_url
                        })
                    
                    saved_plots.append(plot_info)
            
            # Save merged plot if it exists
            if_merged_plot_exists = droma_state.r('!is.null(analysis_result$merged_plot)')[0]
            if if_merged_plot_exists:
                merged_plot_name = f"drug_omic_pair_{select_features}_{select_drugs}_{dataset_name}_merged.png"
                merged_plot_path = temp_dir / merged_plot_name
                
                save_merged_cmd = f'''
                library(ggplot2)
                ggsave(
                    filename = "{str(merged_plot_path)}",
                    plot = analysis_result$merged_plot,
                    width = {plot_width},
                    height = {plot_height},
                    dpi = 300
                )
                '''
                droma_state.r(save_merged_cmd)
                
                if merged_plot_path.exists():
                    from ..util import save_figure
                    merged_fig_id = save_figure(merged_plot_path, merged_plot_name)

                    # Get server URL and generate figure URL if in HTTP mode
                    from ..util import get_server_url, get_figure_url
                    server_url = get_server_url()
                    merged_figure_url = get_figure_url(merged_fig_id) if server_url else None
                    merged_figure_display_path = merged_figure_url if merged_figure_url else str(merged_plot_path)

                    plot_info = {
                        "type": "merged",
                        "figure_id": merged_fig_id,
                        "figure_path": str(merged_plot_path),  # Keep local path for reference
                        "figure_url": merged_figure_url,  # Add URL for HTTP mode
                        "figure_display_path": merged_figure_display_path,  # Path/URL for display
                        "resource_uri": f"figure://{merged_fig_id}"
                    }
                    
                    # Add to inline images list for inline mode
                    if display_mode == "inline":
                        inline_images.append({
                            "title": "Merged Studies Plot",
                            "path": merged_figure_display_path,
                            "url": merged_figure_url
                        })
                    
                    saved_plots.append(plot_info)
            
            if saved_plots:
                response["plots"] = saved_plots
                response["display_mode"] = display_mode
                
                # For inline display mode, format message with images
                if display_mode == "inline" and inline_images:
                    # Format message that AI will output in conversation
                    plot_message = f"{response['message']}\n\n**Generated Plots:**\n\n"
                    for img in inline_images:
                        plot_message += f"**{img['title']}:**\n\n![{img['title']}]({img['path']})\n\n"
                    response["message"] = plot_message
                else:
                    # For link mode, add resource URIs and URLs to message
                    if saved_plots:
                        response["message"] += f"\n\n**Resource URIs (use MCP client to access):**\n"
                        for plot in saved_plots:
                            response["message"] += f"- {plot['type'].title()} Plot: `{plot['resource_uri']}`\n"

                        # Add download URLs if in HTTP mode
                        from ..util import get_server_url
                        server_url = get_server_url()
                        if server_url:
                            response["message"] += f"\n\n**Download URLs:**\n"
                            for plot in saved_plots:
                                if plot.get('figure_url'):
                                    response["message"] += f"- {plot['type'].title()} Plot: {plot['figure_url']}\n"

                        response["message"] += f"\n**Local Paths:**\n"
                        for plot in saved_plots:
                            response["message"] += f"- {plot['type'].title()} Plot: {plot['figure_path']}\n"
                    
                await ctx.info(f"Saved {len(saved_plots)} plot(s) (display mode: {display_mode})")
        
        # Extract meta-analysis results if available
        if_meta_exists = droma_state.r('!is.null(analysis_result$meta)')[0]
        if if_meta_exists:
            try:
                # Extract meta-analysis statistics
                meta_r = droma_state.r('analysis_result$meta')
                
                # Convert to Python dict
                from ..server.data_loading import _convert_r_to_python
                meta_data = _convert_r_to_python(meta_r)
                
                # Try to extract key statistics
                meta_stats = {}
                if hasattr(meta_r, 'names'):
                    meta_names = list(meta_r.names)
                    for name in meta_names:
                        try:
                            value = droma_state.r(f'analysis_result$meta${name}')
                            if value is not None:
                                # Convert to Python type
                                if hasattr(value, '__len__') and len(value) == 1:
                                    meta_stats[name] = float(value[0]) if isinstance(value[0], (int, float)) else str(value[0])
                                else:
                                    meta_stats[name] = str(value)
                        except:
                            pass
                
                response["meta_analysis"] = {
                    "available": True,
                    "statistics": meta_stats
                }
                await ctx.info("Meta-analysis results extracted")
            except Exception as e:
                await ctx.warning(f"Could not extract detailed meta-analysis results: {e}")
                response["meta_analysis"] = {
                    "available": True,
                    "note": "Meta-analysis completed but detailed extraction failed"
                }
        
        # Cache the full result for later retrieval
        cache_key = f"analysis_{dataset_name}_{feature_type}_{select_features}_{select_drugs}"
        droma_state.cache_data(cache_key, {
            "r_object_name": "analysis_result",
            "timestamp": str(droma_state.r('Sys.time()')[0])
        }, {
            "feature_type": feature_type,
            "feature": select_features,
            "drug": select_drugs,
            "dataset": dataset_name
        })
        
        response["data_cache_key"] = cache_key
        response["note"] = "Full R analysis object is cached and can be accessed for further processing"
        
        await ctx.info("Analysis completed successfully")
        return response
        
    except Exception as e:
        error_msg = str(e)
        await ctx.error(f"Error in analysis: {error_msg}")
        return {
            "status": "error",
            "message": f"Failed to analyze drug-omic pair: {error_msg}",
            "parameters": {
                "feature_type": feature_type,
                "feature": select_features,
                "drug": select_drugs,
                "dataset": dataset_name
            }
        }


__all__ = ["analysis_mcp", "analyze_drug_omic_pair"]

