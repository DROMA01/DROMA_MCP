"""Pydantic schemas for DROMA dataset management operations."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal


class LoadDatasetModel(BaseModel):
    """Schema for loading DROMA datasets."""
    
    dataset_id: str = Field(
        description="Dataset identifier (e.g., 'CCLE', 'gCSI', or comma-separated for MultiDromaSet like 'CCLE,gCSI')"
    )
    dataset_type: Literal["DromaSet", "MultiDromaSet"] = Field(
        default="DromaSet",
        description="Type of dataset to load: 'DromaSet' for single datasets or 'MultiDromaSet' for multiple projects"
    )
    db_path: Optional[str] = Field(
        default=None,
        description="Path to DROMA database file (if not set, uses DROMA_DB_PATH environment variable)"
    )
    set_active: bool = Field(
        default=True,
        description="Whether to set this dataset as the active dataset after loading"
    )
    
    @field_validator('dataset_id')
    @classmethod
    def normalize_dataset_id(cls, v: str) -> str:
        """Normalize dataset_id by removing spaces around commas."""
        # Remove spaces around commas for MultiDromaSet IDs
        return ','.join([part.strip() for part in v.split(',')])


class ListDatasetsModel(BaseModel):
    """Schema for listing loaded datasets."""
    
    include_details: bool = Field(
        default=False,
        description="Whether to include detailed information about each dataset"
    )


class SetActiveDatasetModel(BaseModel):
    """Schema for setting the active dataset."""
    
    dataset_id: str = Field(
        description="Dataset identifier to set as active"
    )
    dataset_type: Literal["DromaSet", "MultiDromaSet"] = Field(
        default="DromaSet",
        description="Type of dataset: 'DromaSet' or 'MultiDromaSet'"
    )


class UnloadDatasetModel(BaseModel):
    """Schema for unloading datasets from memory."""
    
    dataset_id: str = Field(
        description="Dataset identifier to unload"
    )
    dataset_type: Literal["DromaSet", "MultiDromaSet"] = Field(
        default="DromaSet",
        description="Type of dataset: 'DromaSet' or 'MultiDromaSet'"
    ) 