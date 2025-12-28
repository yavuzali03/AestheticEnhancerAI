"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Optional


class EnhanceRequest(BaseModel):
    """Request model for image enhancement."""
    
    denoise: bool = Field(
        default=False,
        description="Apply Gaussian blur denoising to remove noise/grain"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "denoise": False
            }
        }


class EnhanceResponse(BaseModel):
    """Response model for enhanced image."""
    
    success: bool = Field(description="Whether processing was successful")
    
    enhanced_image: Optional[str] = Field(
        default=None,
        description="Base64 encoded enhanced image (2x upscaled, JPEG format)"
    )
    
    segmentation_map: Optional[str] = Field(
        default=None,
        description="Base64 encoded segmentation visualization map (JPEG format)"
    )
    
    original_size: Optional[dict] = Field(
        default=None,
        description="Original image dimensions {width, height}"
    )
    
    output_size: Optional[dict] = Field(
        default=None,
        description="Output image dimensions {width, height} (2x upscaled)"
    )
    
    processing_time: Optional[float] = Field(
        default=None,
        description="Processing time in seconds"
    )
    
    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "enhanced_image": "base64_encoded_string_here...",
                "segmentation_map": "base64_encoded_string_here...",
                "original_size": {"width": 800, "height": 600},
                "output_size": {"width": 1600, "height": 1200},
                "processing_time": 45.3,
                "error": None
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(description="Service status")
    device: str = Field(description="Processing device (cpu/cuda)")
    models_loaded: bool = Field(description="Whether AI models are ready")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "device": "cuda",
                "models_loaded": True
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    success: bool = Field(default=False)
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Invalid image format",
                "detail": "Only JPEG, PNG, BMP formats are supported"
            }
        }
