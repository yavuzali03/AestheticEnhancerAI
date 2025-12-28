"""
Image Processor - Core AI processing pipeline.
Refactored from main.py for API usage.
"""

import os
import cv2
import numpy as np
import torch
import warnings
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from typing import Tuple, Optional

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    SegformerImageProcessor,
    AutoModelForSemanticSegmentation
)

from .utils import flush_memory

# Suppress warnings
warnings.filterwarnings("ignore")


class ImageProcessor:
    """Main image processing class with AI enhancement capabilities."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize ImageProcessor.
        
        Args:
            device: 'cuda', 'mps', or 'cpu'. Auto-detect if None.
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        print(f"   ℹ️ İşlemci Birimi: {self.device.upper()}")
        
        # Model instances (lazy loaded)
        self._upsampler = None
        self._face_enhancer = None
        self._depth_processor = None
        self._depth_model = None
        self._seg_processor = None
        self._seg_model = None
    
    def process_image(
        self,
        input_image: Image.Image,
        denoise: bool = False,
        shadow_recovery: bool = False,
        shadow_strength: float = 0.5
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Complete processing pipeline.
        
        Args:
            input_image: Input PIL Image (RGB)
            denoise: Apply Gaussian blur denoising
            shadow_recovery: Apply Lightroom-style shadow recovery
            shadow_strength: Shadow recovery strength (0.0-1.0)
        
        Returns:
            Tuple of (final_enhanced_image, labeled_segmentation_map)
        """
        print(f"\n🚀 İşleme Başladı...")
        print(f"   ℹ️ Orijinal Boyut: {input_image.size[0]}x{input_image.size[1]}")
        print(f"   ℹ️ Temizlik: {'Evet' if denoise else 'Hayır'}")
        print(f"   ℹ️ Gölge Kurtarma: {'Evet' if shadow_recovery else 'Hayır'}")
        
        try:
            # Step 0: Optional denoising
            if denoise:
                print(f"⚙️ 1. Aşama: Temizlik...")
                cleaned_input = self._feature_halftone_fix_restore(input_image)
            else:
                cleaned_input = input_image
            
            # Step 1: AI Restoration
            print(f"⚙️ 2. Aşama: AI Restorasyon...")
            natural_img, sharp_img = self._restore_dual_mode(cleaned_input)
            
            # Step 2: Get analysis maps
            print(f"📏 3. Aşama: Analiz...")
            depth_map, seg_np, labels_dict = self._get_maps(sharp_img)
            
            # Step 3: Create visualization
            labeled_seg_vis = self._create_labeled_segmentation_visual(
                seg_np, labels_dict, sharp_img
            )
            
            # Step 4: Composite
            print(f"⚗️ 4. Aşama: Montaj...")
            fg_mask = depth_map.point(lambda p: 255 if p > 80 else 0).convert("L").filter(
                ImageFilter.GaussianBlur(10)
            )
            person_mask = Image.fromarray(
                ((seg_np == 12) * 255).astype(np.uint8)
            ).convert("L").filter(ImageFilter.GaussianBlur(10))
            
            comp = Image.composite(sharp_img, natural_img, fg_mask)
            final_image = Image.composite(natural_img, comp, person_mask)
            
            # Step 5: Effects
            print(f"✨ 5. Aşama: Son Dokunuşlar...")
            final_image = self._feature_master_curve(final_image, bend_factor=-0.12)
            
            # Adaptive texture: stronger if denoised (to compensate for softness)
            texture_intensity = 0.40 if denoise else 0.25
            final_image = self._feature_lightroom_texture(final_image, intensity=texture_intensity)
            
            # Step 5.5: Optional Shadow Recovery
            if shadow_recovery:
                print(f"🌓 5.5 Aşama: Gölge Kurtarma...")
                final_image = self._feature_true_shadow_recovery(final_image, shadow_strength)
            
            # Step 6: Upscale to 2x (ALWAYS)
            print(f"📏 6. Aşama: 2x Büyütme...")
            orig_w, orig_h = input_image.size
            target_w, target_h = orig_w * 2, orig_h * 2
            final_image = final_image.resize((target_w, target_h), Image.LANCZOS)
            
            print(f"✅ İşleme Tamamlandı! ({target_w}x{target_h})")
            return final_image, labeled_seg_vis
            
        except Exception as e:
            print(f"❌ İşleme Hatası: {e}")
            raise
        finally:
            flush_memory()
    
    # ============================================================
    # PRIVATE HELPER METHODS (from main.py)
    # ============================================================
    
    def _feature_halftone_fix_restore(self, pil_image: Image.Image) -> Image.Image:
        """Downscale + Gaussian Blur + Upscale for noise reduction."""
        print(f"      -> 📉💧📈 [GAUSSIAN BLUR] Tram eritiliyor...")
        
        img_np = np.array(pil_image)
        orig_h, orig_w = img_np.shape[:2]
        
        downscale_factor = 2.0
        new_w = int(orig_w / downscale_factor)
        new_h = int(orig_h / downscale_factor)
        
        small_img = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        small_img = cv2.GaussianBlur(small_img, (3, 3), 0)
        restored_img = cv2.resize(small_img, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        
        return Image.fromarray(restored_img)
    
    def _restore_dual_mode(self, pil_image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """AI-based restoration using GFPGAN and RealESRGAN."""
        flush_memory()
        original_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Build models (lazy)
        if self._upsampler is None:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self._upsampler = RealESRGANer(
                scale=4,
                model_path='../RealESRGAN_x4plus.pth',
                model=model,
                tile=200,
                tile_pad=10,
                pre_pad=0,
                half=(self.device == 'cuda'),
                device=self.device
            )
        
        if self._face_enhancer is None:
            self._face_enhancer = GFPGANer(
                model_path='../GFPGANv1.3.pth',
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=self._upsampler,
                device=self.device
            )
        
        _, _, high_res_cv = self._face_enhancer.enhance(
            original_cv, has_aligned=False, only_center_face=False, paste_back=True
        )
        
        ai_base = high_res_cv
        orig_base = cv2.resize(
            original_cv, (ai_base.shape[1], ai_base.shape[0]), interpolation=cv2.INTER_LANCZOS4
        )
        
        natural_cv = cv2.addWeighted(ai_base, 0.60, orig_base, 0.40, 0)
        sharp_cv = ai_base
        
        return (
            Image.fromarray(cv2.cvtColor(natural_cv, cv2.COLOR_BGR2RGB)),
            Image.fromarray(cv2.cvtColor(sharp_cv, cv2.COLOR_BGR2RGB))
        )
    
    def _get_maps(self, pil_image: Image.Image) -> Tuple[Image.Image, np.ndarray, dict]:
        """Get depth and segmentation maps."""
        flush_memory()
        
        # Depth map
        if self._depth_processor is None:
            self._depth_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
            self._depth_model = AutoModelForDepthEstimation.from_pretrained(
                "LiheYoung/depth-anything-small-hf"
            ).to(self.device)
        
        d_inputs = self._depth_processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            d_out = self._depth_model(**d_inputs).predicted_depth
        
        d_interp = torch.nn.functional.interpolate(
            d_out.unsqueeze(1), size=pil_image.size[::-1], mode="bicubic", align_corners=False
        )
        depth_np = (d_interp.squeeze().cpu().numpy() * 255 / np.max(d_interp.squeeze().cpu().numpy())).astype("uint8")
        
        # Segmentation map
        if self._seg_processor is None:
            self._seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
            self._seg_model = AutoModelForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            ).to(self.device)
        
        s_inputs = self._seg_processor(images=pil_image, return_tensors="pt").to(self.device)
        s_out = self._seg_model(**s_inputs).logits.cpu()
        s_interp = torch.nn.functional.interpolate(
            s_out, size=pil_image.size[::-1], mode="bilinear", align_corners=False
        )
        seg_np = s_interp.argmax(dim=1)[0].numpy().astype(np.uint8)
        
        labels_dict = self._seg_model.config.id2label
        
        return Image.fromarray(depth_np), seg_np, labels_dict
    
    def _create_labeled_segmentation_visual(
        self, seg_np: np.ndarray, id2label: dict, original_pil: Image.Image
    ) -> Image.Image:
        """Create labeled segmentation visualization."""
        print("   🗺️ Etiketli Harita Hazırlanıyor...")
        
        h, w = seg_np.shape
        colored_map = np.zeros((h, w, 3), dtype=np.uint8)
        detected_ids = np.unique(seg_np)
        
        np.random.seed(42)
        id_to_color = {}
        for category_id in detected_ids:
            color = tuple(np.random.randint(50, 255, 3).tolist())
            id_to_color[category_id] = color
            colored_map[seg_np == category_id] = color
        
        colored_pil = Image.fromarray(colored_map, 'RGB')
        blended_pil = Image.blend(original_pil.convert("RGB"), colored_pil, 0.6)
        draw = ImageDraw.Draw(blended_pil)
        
        try:
            font = ImageFont.truetype("arial.ttf", 20)
            box_height = 30
        except:
            font = ImageFont.load_default()
            box_height = 20
        
        x_offset, y_offset = 20, 20
        legend_width = 300
        legend_height = len(detected_ids) * box_height + 20
        
        draw.rectangle(
            [(x_offset - 10, y_offset - 10), (x_offset + legend_width, y_offset + legend_height)],
            fill=(0, 0, 0, 180)
        )
        
        for category_id in detected_ids:
            label_name = id2label.get(category_id, id2label.get(str(category_id), f"Unknown ID: {category_id}"))
            color = id_to_color[category_id]
            draw.rectangle(
                [(x_offset, y_offset), (x_offset + 20, y_offset + 20)],
                fill=color, outline=(255, 255, 255)
            )
            draw.text(
                (x_offset + 30, y_offset),
                f"{str(label_name).upper()} ({category_id})",
                fill=(255, 255, 255),
                font=font
            )
            y_offset += box_height
        
        return blended_pil
    
    def _feature_master_curve(self, pil_image: Image.Image, bend_factor: float = -0.15) -> Image.Image:
        """Apply master curve for light/shadow adjustment."""
        img_np = np.array(pil_image)
        x = np.arange(256, dtype=np.uint8)
        y = (x / 255.0 + bend_factor * np.sin(x / 255.0 * np.pi)) * 255
        lut = np.clip(y, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.LUT(img_np, lut))
    
    def _feature_lightroom_texture(self, pil_image: Image.Image, intensity: float = 0.3) -> Image.Image:
        """Apply Lightroom-style texture enhancement."""
        if intensity <= 0.0:
            return pil_image
        
        img_np = np.array(pil_image)
        blurred = cv2.GaussianBlur(img_np, (0, 0), 3.0)
        sharpened = cv2.addWeighted(img_np, 1.0 + intensity, blurred, -intensity, 0)
        return Image.fromarray(sharpened)
    
    def _feature_true_shadow_recovery(self, pil_image: Image.Image, shadow_strength: float = 0.5) -> Image.Image:
        """
        Lightroom 'Shadows' Slider - Professional shadow recovery using tone mapping.
        
        Args:
            pil_image: Input PIL Image
            shadow_strength: Recovery strength (0.0-1.0), default 0.5
        
        Returns:
            PIL Image with recovered shadows
        """
        print(f"      -> 🌓 [PRO SHADOWS] Karanlık alanlar Tone Mapping ile işleniyor...")
        
        img = np.array(pil_image)
        
        # Convert to float32 for precise math
        img_float = img.astype(np.float32) / 255.0
        
        # 1. Extract Luminance Map
        # Weighted by human eye perception: Green > Red > Blue
        luminance = 0.299 * img_float[:,:,0] + 0.587 * img_float[:,:,1] + 0.114 * img_float[:,:,2]
        
        # 2. Create Shadow Mask with threshold
        # Only affect truly dark areas (luminance < 0.35)
        # Power of 5 creates very sharp falloff, protecting mid-tones
        dark_threshold = 0.35
        shadow_mask = np.where(
            luminance < dark_threshold,
            ((1.0 - (luminance / dark_threshold)) ** 5.0),
            0.0
        )
        
        # 3. Calculate Brightening Factor
        # Only brighten where mask exists (dark areas)
        brightening_factor = 1.0 + (shadow_mask * shadow_strength)
        
        # 4. Apply Mask to Image (Brighten)
        # Multiply each color channel equally to preserve color
        result_float = np.zeros_like(img_float)
        for i in range(3):
            result_float[:,:,i] = np.clip(img_float[:,:,i] * brightening_factor, 0, 1.0)
        
        # 5. Saturation Boost (only in affected areas)
        # Recovered shadows can look washed out, boost color by 10%
        result_uint8 = (result_float * 255).astype(np.uint8)
        hsv = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Only boost saturation where we actually brightened (shadow_mask > 0)
        saturation_boost = 1.0 + (shadow_mask * 0.15)  # Up to 15% boost in darkest areas
        hsv[:,:,1] = hsv[:,:,1] * saturation_boost
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        final_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return Image.fromarray(final_img)
