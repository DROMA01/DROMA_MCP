"""DROMA MCP server for database query and exploration operations."""

from fastmcp import FastMCP, Context
from typing import Dict, Any
import sqlite3
from pathlib import Path

from ..schema.database_query import (
    GetAnnotationModel,
    ListSamplesModel,
    ListFeaturesModel,
    ListProjectsModel
)

# Create sub-MCP server for database queries
database_query_mcp = FastMCP("DROMA-Database-Query")


def _get_database_connection(droma_state) -> sqlite3.Connection:
    """Get database connection from DROMA state."""
    # Check if we have a database path in environment or state
    import os
    db_path = os.environ.get('DROMA_DB_PATH')
    
    if not db_path:
        raise RuntimeError("No database path configured. Set DROMA_DB_PATH environment variable or use --db-path")
    
    if not Path(db_path).exists():
        raise RuntimeError(f"Database file not found: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        # Enable row factory for easier data access
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        raise RuntimeError(f"Failed to connect to database: {e}")


@database_query_mcp.tool()
async def get_droma_annotation(
    ctx: Context,
    request: GetAnnotationModel
) -> Dict[str, Any]:
    """
    Retrieve annotation data from either sample_anno or drug_anno tables.
    
    Equivalent to R function: getDROMAAnnotation()
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    try:
        # Get database connection
        conn = _get_database_connection(droma_state)
        
        # Determine table name and ID column
        if request.anno_type == "sample":
            table_name = "sample_anno"
            id_column = "SampleID"
            project_column = "ProjectID"
        else:
            table_name = "drug_anno"  
            id_column = "DrugName"
            project_column = "ProjectID"
        
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            conn.close()
            return {
                "status": "error",
                "message": f"Annotation table '{table_name}' not found in database"
            }
        
        # Build query
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        params = []
        
        # Add project filter
        if request.project_name:
            query += f" AND {project_column} = ?"
            params.append(request.project_name)
        
        # Add ID filter
        if request.ids and len(request.ids) > 0:
            placeholders = ",".join(["?" for _ in request.ids])
            query += f" AND {id_column} IN ({placeholders})"
            params.extend(request.ids)
        
        # Add sample-specific filters
        if request.anno_type == "sample":
            if request.data_type.value != "all":
                query += " AND DataType = ?"
                params.append(request.data_type.value)
            
            if request.tumor_type != "all":
                query += " AND TumorType = ?"
                params.append(request.tumor_type)
        
        # Add ordering
        query += f" ORDER BY {id_column}"
        
        # Add limit if specified
        if request.limit and request.limit > 0:
            query += " LIMIT ?"
            params.append(request.limit)
        
        await ctx.info(f"Executing query for {request.anno_type} annotations")
        
        # Execute query
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        result_data = []
        if rows:
            columns = [description[0] for description in cursor.description]
            for row in rows:
                result_data.append(dict(zip(columns, row)))
        
        # Get total count for summary
        total_query = f"SELECT COUNT(*) FROM {table_name}"
        cursor.execute(total_query)
        total_records = cursor.fetchone()[0]
        
        conn.close()
        
        # Prepare filter description for logging
        filters = []
        if request.project_name:
            filters.append(f"project='{request.project_name}'")
        if request.ids:
            filters.append(f"specific IDs ({len(request.ids)} requested)")
        if request.anno_type == "sample":
            if request.data_type.value != "all":
                filters.append(f"data_type='{request.data_type.value}'")
            if request.tumor_type != "all":
                filters.append(f"tumor_type='{request.tumor_type}'")
        
        filter_desc = f" (filtered by {', '.join(filters)})" if filters else ""
        
        if request.limit:
            message = f"Retrieved first {len(result_data)} {request.anno_type} annotations out of {total_records} total records{filter_desc}"
        else:
            message = f"Retrieved {len(result_data)} {request.anno_type} annotations{filter_desc}"
        
        await ctx.info(message)
        
        return {
            "status": "success",
            "annotation_type": request.anno_type,
            "data": result_data,
            "total_records": len(result_data),
            "total_in_database": total_records,
            "message": message
        }
        
    except Exception as e:
        await ctx.error(f"Error retrieving {request.anno_type} annotations: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to retrieve {request.anno_type} annotations: {str(e)}"
        }


@database_query_mcp.tool()
async def list_droma_samples(
    ctx: Context,
    request: ListSamplesModel
) -> Dict[str, Any]:
    """
    List all available samples for a specific project with optional filters.
    
    Equivalent to R function: listDROMASamples()
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    try:
        # Get database connection
        conn = _get_database_connection(droma_state)
        cursor = conn.cursor()
        
        # Check if sample_anno table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sample_anno'")
        if not cursor.fetchone():
            conn.close()
            return {
                "status": "error",
                "message": "Sample annotation table 'sample_anno' not found in database"
            }
        
        # Get samples with data_sources filter if specified
        filtered_samples_by_data = None
        if request.data_sources != "all":
            data_table_name = f"{request.project_name}_{request.data_sources}"
            
            # Check if the data source table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (data_table_name,))
            if not cursor.fetchone():
                # Get available tables for this project
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", 
                             (f"{request.project_name}_%",))
                available_tables = [row[0] for row in cursor.fetchall()]
                
                conn.close()
                return {
                    "status": "error",
                    "message": f"Data source table '{data_table_name}' not found. Available tables: {', '.join(available_tables)}"
                }
            
            # Get samples that have data in this data source
            if request.data_sources in ["mRNA", "cnv", "meth", "proteinrppa", "proteinms", "drug", "drug_raw"]:
                # For continuous data, get column names (excluding feature_id)
                cursor.execute(f"PRAGMA table_info({data_table_name})")
                columns_info = cursor.fetchall()
                filtered_samples_by_data = [row[1] for row in columns_info if row[1] != "feature_id"]
            elif request.data_sources in ["mutation_gene", "mutation_site", "fusion"]:
                # For discrete data, get unique values from samples column
                cursor.execute(f"SELECT DISTINCT samples FROM {data_table_name} WHERE samples IS NOT NULL")
                discrete_result = cursor.fetchall()
                filtered_samples_by_data = [row[0] for row in discrete_result]
            else:
                # Try to detect automatically
                cursor.execute(f"PRAGMA table_info({data_table_name})")
                columns_info = cursor.fetchall()
                column_names = [row[1] for row in columns_info]
                
                if "samples" in column_names:
                    # Discrete data
                    cursor.execute(f"SELECT DISTINCT samples FROM {data_table_name} WHERE samples IS NOT NULL")
                    discrete_result = cursor.fetchall()
                    filtered_samples_by_data = [row[0] for row in discrete_result]
                else:
                    # Continuous data
                    filtered_samples_by_data = [name for name in column_names if name != "feature_id"]
            
            if len(filtered_samples_by_data) == 0:
                conn.close()
                return {
                    "status": "warning",
                    "message": f"No samples found with data in '{request.data_sources}' for project '{request.project_name}'",
                    "samples": []
                }
        
        # Construct main query
        query = "SELECT DISTINCT SampleID FROM sample_anno WHERE ProjectID = ?"
        params = [request.project_name]
        
        # Add data type filter
        if request.data_type.value != "all":
            query += " AND DataType = ?"
            params.append(request.data_type.value)
        
        # Add tumor type filter
        if request.tumor_type != "all":
            query += " AND TumorType = ?"
            params.append(request.tumor_type)
        
        # Add data sources filter
        if filtered_samples_by_data is not None:
            if len(filtered_samples_by_data) > 0:
                placeholders = ",".join(["?" for _ in filtered_samples_by_data])
                query += f" AND SampleID IN ({placeholders})"
                params.extend(filtered_samples_by_data)
            else:
                conn.close()
                return {
                    "status": "warning",
                    "message": "No samples with data in the specified data source",
                    "samples": []
                }
        
        # Add pattern filter if specified
        if request.pattern:
            # Convert basic regex patterns to SQL LIKE patterns
            like_pattern = request.pattern
            if like_pattern.startswith("^"):
                like_pattern = like_pattern[1:] + "%"
            elif like_pattern.endswith("$"):
                like_pattern = "%" + like_pattern[:-1]
            elif like_pattern.startswith("^") and like_pattern.endswith("$"):
                like_pattern = like_pattern[1:-1]
            else:
                like_pattern = f"%{like_pattern}%"
            
            # Replace regex wildcards with SQL wildcards
            like_pattern = like_pattern.replace("*", "%").replace(".", "_")
            
            query += " AND SampleID LIKE ?"
            params.append(like_pattern)
        
        # Add ordering
        query += " ORDER BY SampleID"
        
        # Add limit if specified
        if request.limit and request.limit > 0:
            query += " LIMIT ?"
            params.append(request.limit)
        
        await ctx.info(f"Executing query for samples in project {request.project_name}")
        
        # Execute query
        cursor.execute(query, params)
        sample_rows = cursor.fetchall()
        samples = [row[0] for row in sample_rows]
        
        # Get total count
        total_query = "SELECT COUNT(DISTINCT SampleID) FROM sample_anno WHERE ProjectID = ?"
        cursor.execute(total_query, [request.project_name])
        total_samples = cursor.fetchone()[0]
        
        conn.close()
        
        # Prepare filter description
        filters = []
        if request.data_type.value != "all":
            filters.append(f"data_type='{request.data_type.value}'")
        if request.tumor_type != "all":
            filters.append(f"tumor_type='{request.tumor_type}'")
        if request.data_sources != "all":
            filters.append(f"data_sources='{request.data_sources}'")
        if request.pattern:
            filters.append(f"pattern='{request.pattern}'")
        
        filter_desc = f" (filtered by {', '.join(filters)})" if filters else ""
        
        if request.limit:
            message = f"Showing first {len(samples)} samples out of {total_samples} total samples for project '{request.project_name}'{filter_desc}"
        else:
            if filters or request.data_sources != "all":
                message = f"Found {len(samples)} samples out of {total_samples} total samples for project '{request.project_name}'{filter_desc}"
            else:
                message = f"Found {len(samples)} samples for project '{request.project_name}'{filter_desc}"
        
        await ctx.info(message)
        
        return {
            "status": "success",
            "project_name": request.project_name,
            "samples": samples,
            "total_found": len(samples),
            "total_in_project": total_samples,
            "message": message
        }
        
    except Exception as e:
        await ctx.error(f"Error listing samples for project {request.project_name}: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to list samples for project {request.project_name}: {str(e)}"
        }


@database_query_mcp.tool()
async def list_droma_features(
    ctx: Context,
    request: ListFeaturesModel
) -> Dict[str, Any]:
    """
    List all available features (genes, drugs, etc.) for a specific project and data type.
    
    Equivalent to R function: listDROMAFeatures()
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    try:
        # Get database connection
        conn = _get_database_connection(droma_state)
        cursor = conn.cursor()
        
        # Construct table name
        table_name = f"{request.project_name}_{request.data_sources}"
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if not cursor.fetchone():
            # Get available tables for this project
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", 
                         (f"{request.project_name}_%",))
            available_tables = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return {
                "status": "error",
                "message": f"Table '{table_name}' not found. Available tables: {', '.join(available_tables)}"
            }
        
        # Determine the feature column name based on data type
        if request.data_sources in ["mRNA", "cnv", "meth", "proteinrppa", "proteinms", "drug", "drug_raw"]:
            feature_column = "feature_id"
        elif request.data_sources in ["mutation_gene", "mutation_site", "fusion"]:
            feature_column = "features"
        else:
            # Try to detect automatically
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            column_names = [row[1] for row in columns_info]
            
            if "feature_id" in column_names:
                feature_column = "feature_id"
            elif "features" in column_names:
                feature_column = "features"
            else:
                conn.close()
                return {
                    "status": "error",
                    "message": f"Cannot determine feature column for data type '{request.data_sources}'. Available columns: {', '.join(column_names)}"
                }
        
        # Handle sample filtering for continuous data types
        # Note: This is conceptual since all features exist, but we might want to filter based on sample availability
        
        # Construct query to get distinct features
        query = f"SELECT DISTINCT {feature_column} FROM {table_name} WHERE {feature_column} IS NOT NULL"
        params = []
        
        # Add pattern filter if specified
        if request.pattern:
            # Convert regex pattern to SQL LIKE pattern
            like_pattern = request.pattern
            if like_pattern.startswith("^"):
                like_pattern = like_pattern[1:] + "%"
            elif like_pattern.endswith("$"):
                like_pattern = "%" + like_pattern[:-1]
            elif like_pattern.startswith("^") and like_pattern.endswith("$"):
                like_pattern = like_pattern[1:-1]
            else:
                like_pattern = f"%{like_pattern}%"
            
            # Replace regex wildcards with SQL wildcards
            like_pattern = like_pattern.replace("*", "%").replace(".", "_")
            
            query += f" AND {feature_column} LIKE ?"
            params.append(like_pattern)
        
        # Add ordering
        query += f" ORDER BY {feature_column}"
        
        # Add limit if specified
        if request.limit and request.limit > 0:
            query += " LIMIT ?"
            params.append(request.limit)
        
        await ctx.info(f"Executing query for features in {table_name}")
        
        # Execute query
        cursor.execute(query, params)
        feature_rows = cursor.fetchall()
        features = [row[0] for row in feature_rows]
        
        # Get total count
        total_query = f"SELECT COUNT(DISTINCT {feature_column}) FROM {table_name} WHERE {feature_column} IS NOT NULL"
        cursor.execute(total_query)
        total_features = cursor.fetchone()[0]
        
        conn.close()
        
        # Prepare filter description
        filters = []
        if request.pattern:
            filters.append(f"pattern='{request.pattern}'")
        if request.data_type.value != "all":
            filters.append(f"data_type='{request.data_type.value}'")
        if request.tumor_type != "all":
            filters.append(f"tumor_type='{request.tumor_type}'")
        
        filter_desc = f" (filtered by {', '.join(filters)})" if filters else ""
        
        if request.limit:
            message = f"Showing first {len(features)} features out of {total_features} total features in {table_name}{filter_desc}"
        elif request.pattern:
            message = f"Found {len(features)} features matching pattern '{request.pattern}' out of {total_features} total features in {table_name}{filter_desc}"
        else:
            message = f"Found {len(features)} features in {table_name}{filter_desc}"
        
        await ctx.info(message)
        
        return {
            "status": "success",
            "project_name": request.project_name,
            "data_sources": request.data_sources,
            "features": features,
            "total_found": len(features),
            "total_in_table": total_features,
            "message": message
        }
        
    except Exception as e:
        await ctx.error(f"Error listing features from {request.project_name}_{request.data_sources}: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to list features from {request.project_name}_{request.data_sources}: {str(e)}"
        }


@database_query_mcp.tool()
async def list_droma_projects(
    ctx: Context,
    request: ListProjectsModel
) -> Dict[str, Any]:
    """
    List all projects available in the DROMA database.
    
    Equivalent to R function: listDROMAProjects()
    """
    # Get DROMA state
    droma_state = ctx.request_context.lifespan_context
    
    try:
        # Get database connection
        conn = _get_database_connection(droma_state)
        cursor = conn.cursor()
        
        # Check if projects table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='projects'")
        has_projects_table = cursor.fetchone() is not None
        
        if has_projects_table:
            # Use the projects table
            cursor.execute("SELECT * FROM projects")
            project_rows = cursor.fetchall()
            
            if request.project_data_types:
                # Return data types for a specific project
                for row in project_rows:
                    if row[0] == request.project_data_types:  # Assuming project_name is first column
                        # Find data_types column
                        cursor.execute("PRAGMA table_info(projects)")
                        columns_info = cursor.fetchall()
                        column_names = [col[1] for col in columns_info]
                        
                        if "data_types" in column_names:
                            data_types_idx = column_names.index("data_types")
                            data_types_str = row[data_types_idx]
                            data_types = data_types_str.split(",") if data_types_str else []
                            
                            conn.close()
                            return {
                                "status": "success",
                                "project_name": request.project_data_types,
                                "data_types": data_types,
                                "message": f"Found {len(data_types)} data types for project '{request.project_data_types}'"
                            }
                
                conn.close()
                return {
                    "status": "warning",
                    "message": f"Project '{request.project_data_types}' not found",
                    "data_types": []
                }
            
            if request.show_names_only:
                # Return only project names
                project_names = [row[0] for row in project_rows]  # Assuming project_name is first column
                conn.close()
                return {
                    "status": "success",
                    "project_names": project_names,
                    "message": f"Found {len(project_names)} projects"
                }
            
            # Return full project information
            cursor.execute("PRAGMA table_info(projects)")
            columns_info = cursor.fetchall()
            column_names = [col[1] for col in columns_info]
            
            projects_data = []
            for row in project_rows:
                project_dict = dict(zip(column_names, row))
                projects_data.append(project_dict)
            
            conn.close()
            return {
                "status": "success",
                "projects": projects_data,
                "total_projects": len(projects_data),
                "message": f"Found {len(projects_data)} projects in database"
            }
        
        else:
            # Infer projects from table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            all_tables = [row[0] for row in cursor.fetchall()]
            
            # Extract project names from table prefixes
            project_names = set()
            for table in all_tables:
                if table not in ["sample_anno", "drug_anno", "droma_metadata", "search_vectors"]:
                    parts = table.split("_")
                    if len(parts) >= 2:
                        project_names.add(parts[0])
            
            project_names = sorted(list(project_names))
            
            if len(project_names) == 0:
                conn.close()
                return {
                    "status": "warning",
                    "message": "No projects found in database",
                    "projects": [] if not request.show_names_only else [],
                    "project_names": [] if request.show_names_only else None
                }
            
            if request.project_data_types:
                # Return data types for a specific project
                if request.project_data_types in project_names:
                    project_tables = [t for t in all_tables if t.startswith(f"{request.project_data_types}_")]
                    data_types = sorted(list(set([t.replace(f"{request.project_data_types}_", "") for t in project_tables])))
                    
                    conn.close()
                    return {
                        "status": "success",
                        "project_name": request.project_data_types,
                        "data_types": data_types,
                        "message": f"Found {len(data_types)} data types for project '{request.project_data_types}'"
                    }
                else:
                    conn.close()
                    return {
                        "status": "warning",
                        "message": f"Project '{request.project_data_types}' not found",
                        "data_types": []
                    }
            
            if request.show_names_only:
                conn.close()
                return {
                    "status": "success",
                    "project_names": project_names,
                    "message": f"Found {len(project_names)} projects"
                }
            
            # Create basic project information
            projects_data = []
            for project_name in project_names:
                projects_data.append({
                    "project_name": project_name,
                    "source": "inferred_from_tables"
                })
            
            conn.close()
            return {
                "status": "success",
                "projects": projects_data,
                "total_projects": len(projects_data),
                "message": f"Found {len(projects_data)} projects (inferred from table names)"
            }
        
    except Exception as e:
        await ctx.error(f"Error listing projects: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to list projects: {str(e)}"
        } 