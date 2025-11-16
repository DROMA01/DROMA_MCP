"""Pydantic schemas for DROMA MCP data validation."""

# Data loading schemas
from .data_loading import (
    MolecularType,
    LoadMolecularProfilesModel,
    LoadTreatmentResponseModel,
    MultiProjectMolecularProfilesModel,
    MultiProjectTreatmentResponseModel,
    ViewCachedDataModel,
    ExportCachedDataModel
)

# Database query schemas
from .database_query import (
    DataType,
    GetAnnotationModel,
    ListSamplesModel,
    ListFeaturesModel,
    ListProjectsModel
)

# Dataset management schemas
from .dataset_management import (
    LoadDatasetModel,
    ListDatasetsModel,
    SetActiveDatasetModel,
    UnloadDatasetModel
)

# Visualization schemas
# Note: visualization tools use direct parameters, not request models
# from .visualization import (
#     PlotDrugSensitivityRankModel
# )

# Analysis schemas  
# Note: analysis tools use direct parameters, not request models
# from .analysis import (
#     AnalyzeDrugOmicPairModel
# )

__all__ = [
    # Enums
    "MolecularType",
    "DataType",
    # Data loading models
    "LoadMolecularProfilesModel",
    "LoadTreatmentResponseModel", 
    "MultiProjectMolecularProfilesModel",
    "MultiProjectTreatmentResponseModel",
    "ViewCachedDataModel",
    "ExportCachedDataModel",
    # Database query models
    "GetAnnotationModel",
    "ListSamplesModel", 
    "ListFeaturesModel",
    "ListProjectsModel",
    # Dataset management models
    "LoadDatasetModel",
    "ListDatasetsModel",
    "SetActiveDatasetModel",
    "UnloadDatasetModel",
    # Visualization models - currently not used (direct params)
    # "PlotDrugSensitivityRankModel",
    # Analysis models - currently not used (direct params)
    # "AnalyzeDrugOmicPairModel"
] 