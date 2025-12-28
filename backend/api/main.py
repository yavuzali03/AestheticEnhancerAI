"""FastAPI application entry point."""

import os
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routes import router
from .models import ErrorResponse

# Suppress warnings
warnings.filterwarnings("ignore")

# Fix basicsr compatibility
def fix_basicsr_compatibility():
    """Fix basicsr torchvision compatibility issue."""
    try:
        import basicsr
        package_path = os.path.dirname(basicsr.__file__)
        target_file = os.path.join(package_path, 'data', 'degradations.py')
        
        if os.path.exists(target_file):
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            old_imp = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
            new_imp = "from torchvision.transforms.functional import rgb_to_grayscale"
            
            if old_imp in content:
                print("🔧 Basicsr uyumluluk yaması uygulanıyor...")
                content = content.replace(old_imp, new_imp)
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(content)
    except Exception:
        pass


# Apply fix before importing heavy libraries
fix_basicsr_compatibility()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    print("🚀 AestheticEnhancerAI API başlatılıyor...")
    print("📥 Model dosyaları kontrol ediliyor...")
    
    # Download models if needed
    from core.processor import ImageProcessor
    
    # Check model files
    model_files = {
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "GFPGANv1.3.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    }
    
    for filename, url in model_files.items():
        if not os.path.exists(filename):
            print(f"⚠️ Model dosyası bulunamadı: {filename}")
            print(f"   Lütfen şu URL'den indirin: {url}")
    
    print("✅ API hazır!")
    yield
    print("👋 API kapatılıyor...")


# Create FastAPI app
app = FastAPI(
    title="AestheticEnhancerAI API",
    description="AI-powered image enhancement API with face restoration and super-resolution",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["Image Enhancement"])


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """API root endpoint."""
    return {
        "message": "AestheticEnhancerAI API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return ErrorResponse(
        success=False,
        error="Internal server error",
        detail=str(exc)
    )
