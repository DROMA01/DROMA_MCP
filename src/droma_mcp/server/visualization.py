"""DROMA MCP server for visualization operations."""

from fastmcp import FastMCP, Context
from typing import Dict, Any, Union
import tempfile
from pathlib import Path

# Create sub-MCP server for visualization
visualization_mcp = FastMCP("DROMA-Visualization")


@visualization_mcp.tool()
async def plot_drug_sensitivity_rank(
    ctx: Context,
    dataset_name: str,
    select_drugs: str,
    data_type: str = "all",
    tumor_type: str = "all",
    overlap_only: bool = False,
    highlight: Union[int, str, list, None] = None,
    color: Union[str, None] = None,
    zscore: bool = False,
    merge: bool = False,
    point_size: float = 2.0,
    highlight_alpha: float = 0.6,
    sample_annotations: Union[str, None] = None,
    db_path: Union[str, None] = None,
    output_file: Union[str, None] = None,
    width: float = 10.0,
    height: float = 6.0,
    display_mode: str = "inline"
) -> Dict[str, Any]:
    """
    Create drug sensitivity rank plot showing samples ordered by sensitivity values.
    
    Equivalent to R function: plotDrugSensitivityRank()
    
    For AUC data, higher values indicate lower sensitivity, so samples are ranked 
    with highest values on the left. Supports highlighting specific samples and 
    coloring by categorical variables.
    
    Args:
        ctx: FastMCP context
        dataset_name: Dataset name (e.g., 'CCLE', 'gCSI') for DromaSet or MultiDromaSet
        select_drugs: Drug name to plot
        data_type: Filter by data type: "all" (default), "CellLine", "PDC", "PDO", "PDX"
        tumor_type: Filter by tumor type: "all" (default) or specific tumor type
        overlap_only: For MultiDromaSet, whether to use only overlapping samples
        highlight: Samples to highlight (int for top N, str for type/tumor, or list of IDs)
        color: Variable to use for coloring points (None, "data_type", "tumor_type", or annotation column)
        zscore: Whether to use z-score normalized values
        merge: For MultiDromaSet with zscore=True, merge data from multiple projects
        point_size: Size of points in the plot
        highlight_alpha: Alpha transparency for non-highlighted points
        sample_annotations: Optional dataframe identifier containing sample annotations
        db_path: Optional path to SQLite database for loading sample annotations
        output_file: Optional output file path for saving the plot
        width: Plot width in inches
        height: Plot height in inches
        display_mode: How to display plots: 'inline' (show in chat, default) or 'link' (provide download link only)
    
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - message: Description of the result
        - figure_id: Identifier for retrieving the saved figure
        - figure_path: Path to the saved figure file
        - plot_info: Information about the plot (dimensions, samples, etc.)
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
        highlight_str = "NULL"
        if highlight is not None:
            if isinstance(highlight, int):
                highlight_str = str(highlight)
            elif isinstance(highlight, str):
                highlight_str = f'"{highlight}"'
            elif isinstance(highlight, list):
                highlight_str = 'c("' + '", "'.join(highlight) + '")'
        
        color_str = "NULL"
        if color:
            color_str = f'"{color}"'
        
        sample_annotations_str = "NULL"
        if sample_annotations:
            sample_annotations_str = sample_annotations
        
        db_path_str = "NULL"
        if db_path:
            db_path_str = f'"{db_path}"'
        
        # Generate temporary file for plot output
        temp_dir = Path(tempfile.gettempdir()) / "droma_mcp_figures"
        temp_dir.mkdir(exist_ok=True)
        
        # Determine file extension
        if output_file:
            file_ext = Path(output_file).suffix
            if not file_ext:
                file_ext = ".png"
            output_name = output_file
        else:
            file_ext = ".png"
            output_name = f"drug_sensitivity_rank_{select_drugs}_{dataset_name}{file_ext}"
        
        output_path = temp_dir / output_name
        
        # Build R command for plotting
        r_command = f'''
        library(ggplot2)
        
        # Create the plot
        plot_result <- plotDrugSensitivityRank(
            dromaset_object = {dataset_r_name},
            select_drugs = "{select_drugs}",
            data_type = "{data_type}",
            tumor_type = "{tumor_type}",
            overlap_only = {str(overlap_only).upper()},
            highlight = {highlight_str},
            color = {color_str},
            zscore = {str(zscore).upper()},
            merge = {str(merge).upper()},
            point_size = {point_size},
            highlight_alpha = {highlight_alpha},
            sample_annotations = {sample_annotations_str},
            db_path = {db_path_str}
        )
        
        # Save the plot
        ggsave(
            filename = "{str(output_path)}",
            plot = plot_result,
            width = {width},
            height = {height},
            dpi = 300
        )
        '''
        
        await ctx.info(f"Generating drug sensitivity rank plot for {select_drugs}")
        
        # Execute R command
        droma_state.r(r_command)
        
        # Verify plot was created
        if not output_path.exists():
            return {
                "status": "error",
                "message": "Plot file was not created. Check if the drug exists in the dataset."
            }
        
        # Save figure using util function
        from ..util import save_figure
        figure_id = save_figure(output_path, output_name)
        
        await ctx.info(f"Plot saved successfully: {figure_id} (display mode: {display_mode})")
        
        # Prepare response
        response = {
            "status": "success",
            "message": f"Drug sensitivity rank plot created successfully for {select_drugs}",
            "figure_id": figure_id,
            "figure_path": str(output_path),
            "display_mode": display_mode,
            "plot_info": {
                "drug": select_drugs,
                "dataset": dataset_name,
                "dataset_type": "MultiDromaSet" if is_multi else "DromaSet",
                "data_type": data_type,
                "tumor_type": tumor_type,
                "zscore_normalized": zscore,
                "merged": merge if is_multi else False,
                "dimensions": {
                    "width": width,
                    "height": height
                }
            },
            "resource_uri": f"figure://{figure_id}"
        }
        
        # Format message based on display mode
        if display_mode == "inline":
            # Add markdown formatted image to message using file path
            plot_message = f"{response['message']}\n\n**Drug Sensitivity Rank Plot:**\n\n![Drug Sensitivity Rank for {select_drugs}]({str(output_path)})\n\n"
            response["message"] = plot_message
        else:
            # For link mode, add resource URI and local path to message
            response["message"] += f"\n\n**Resource URI (use MCP client to access):** `{response['resource_uri']}`"
            response["message"] += f"\n\n**Local Path:** {str(output_path)}"
        
        return response
        
    except Exception as e:
        error_msg = str(e)
        await ctx.error(f"Error creating plot: {error_msg}")
        return {
            "status": "error",
            "message": f"Failed to create plot: {error_msg}"
        }


__all__ = ["visualization_mcp", "plot_drug_sensitivity_rank"]

