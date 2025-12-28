"""API routes for image enhancement endpoints."""

import time
import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

from core import ImageProcessor, pil_to_base64, validate_image, flush_memory
from .models import EnhanceResponse, HealthResponse, ErrorResponse

router = APIRouter()

# Global processor instance (lazy loaded)
_processor: ImageProcessor = None


def get_processor() -> ImageProcessor:
    """Get or create global ImageProcessor instance."""
    global _processor
    if _processor is None:
        _processor = ImageProcessor()
    return _processor


@router.post(
    "/enhance",
    response_model=EnhanceResponse,
    summary="Enhance Image with AI",
    description="Upload an image to enhance with AI. Automatically upscales to 2x resolution."
)
async def enhance_image(
    image: UploadFile = File(..., description="Image file to enhance (JPEG, PNG, BMP)"),
    denoise: bool = Form(False, description="Apply denoising to remove grain/noise"),
    shadow_recovery: bool = Form(False, description="Apply shadow recovery (brighten dark areas)")
) -> EnhanceResponse:
    """
    Enhance uploaded image with AI processing.
    
    - **image**: Image file (max 10MB, JPEG/PNG/BMP)
    - **denoise**: Optional denoising (default: false)
    - **shadow_recovery**: Optional shadow recovery (default: false)
    
    Returns 2x upscaled enhanced image and segmentation map as base64 strings.
    """
    start_time = time.time()
    
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image file."
            )
        
        # Read and validate image
        contents = await image.read()
        
        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(contents) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {max_size / (1024 * 1024):.0f}MB"
            )
        
        # Open image
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        original_size = {"width": pil_image.size[0], "height": pil_image.size[1]}
        
        # Process image
        processor = get_processor()
        enhanced_img, seg_map = processor.process_image(pil_image, denoise=denoise, shadow_recovery=shadow_recovery)
        
        # Convert to base64
        enhanced_b64 = pil_to_base64(enhanced_img, format="JPEG", quality=95)
        seg_map_b64 = pil_to_base64(seg_map, format="JPEG", quality=90)
        
        output_size = {"width": enhanced_img.size[0], "height": enhanced_img.size[1]}
        processing_time = time.time() - start_time
        
        # Clean up
        flush_memory()
        
        return EnhanceResponse(
            success=True,
            enhanced_image=enhanced_b64,
            segmentation_map=seg_map_b64,
            original_size=original_size,
            output_size=output_size,
            processing_time=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        flush_memory()
        return EnhanceResponse(
            success=False,
            error=str(e)
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the service is running and models are loaded"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns service status and available processing device.
    """
    try:
        processor = get_processor()
        return HealthResponse(
            status="healthy",
            device=processor.device,
            models_loaded=True
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            device="unknown",
            models_loaded=False
        )
