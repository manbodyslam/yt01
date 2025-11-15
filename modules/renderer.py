"""
Renderer Module - Composite images using OpenCV/Pillow and render text with FFmpeg
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import subprocess

from config import settings
from utils.image_utils import load_image, save_image
from modules.layout import LayoutEngine, CharacterPlacement, TextPlacement


class Renderer:
    """
    Renders final thumbnail by compositing background, characters, and text
    """

    def __init__(self):
        """
        Initialize Renderer
        """
        self.width = settings.OUTPUT_WIDTH
        self.height = settings.OUTPUT_HEIGHT

        logger.info(f"Renderer initialized ({self.width}x{self.height})")

    def create_thumbnail(
        self,
        background_image_path: Path,
        characters: Dict[str, Dict],
        layout: Dict,
        colors: Dict[str, Tuple[int, int, int]],
        title: str,
        subtitle: str = None,
        text_style: str = "style1"
    ) -> np.ndarray:
        """
        Create complete thumbnail

        Args:
            background_image_path: Path to background image
            characters: Character data with face info
            layout: Layout dictionary from LayoutEngine
            colors: Color scheme from PaletteExtractor
            title: Title text
            subtitle: Optional subtitle text
            text_style: Text style preset (style1, style2, style3, auto)

        Returns:
            Rendered thumbnail as numpy array
        """
        logger.info(f"Starting thumbnail rendering with text style: {text_style}...")

        # 1. Prepare background
        canvas = self._prepare_background(background_image_path)

        # 2. Composite characters
        canvas = self._composite_characters(canvas, characters, layout)

        # 3. Auto enhance image (brightness, contrast, saturation, sharpness)
        if settings.ENABLE_AUTO_ENHANCE:
            canvas = self._auto_enhance(canvas)

        # 4. Add text overlays using PIL with selected style
        canvas = self._add_text_overlays(canvas, layout, colors, title, subtitle, text_style)

        logger.info("Thumbnail rendering complete")

        return canvas

    def _prepare_background(self, image_path: Path) -> np.ndarray:
        """
        Prepare background image with blur and vignette

        Args:
            image_path: Path to background image

        Returns:
            Processed background as numpy array
        """
        logger.debug("Preparing background...")

        # Load image
        img = load_image(image_path)

        # Convert to PIL for processing
        img_pil = Image.fromarray(img)

        # Calculate target aspect ratio (16:9)
        target_aspect = self.width / self.height
        img_aspect = img_pil.width / img_pil.height

        # Center crop to maintain aspect ratio
        if img_aspect > target_aspect:
            # Image is wider - crop width
            new_width = int(img_pil.height * target_aspect)
            left = (img_pil.width - new_width) // 2
            img_pil = img_pil.crop((left, 0, left + new_width, img_pil.height))
        else:
            # Image is taller - crop height
            new_height = int(img_pil.width / target_aspect)
            top = (img_pil.height - new_height) // 2
            img_pil = img_pil.crop((0, top, img_pil.width, top + new_height))

        # Now resize to canvas size (aspect ratio is already correct)
        img_pil = img_pil.resize((self.width, self.height), Image.LANCZOS)
        img = np.array(img_pil)

        # Apply blur with adaptive kernel size
        # Ensure kernel size is odd and not larger than image dimensions
        blur_radius = min(settings.BLUR_RADIUS, min(img.shape[0], img.shape[1]) // 4)
        kernel_size = blur_radius * 2 + 1
        if kernel_size > 0:
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # Apply vignette
        img = self._apply_vignette(img, strength=settings.VIGNETTE_STRENGTH)

        return img

    def _apply_vignette(self, image: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Apply vignette effect

        Args:
            image: Input image
            strength: Vignette strength (0-1)

        Returns:
            Image with vignette
        """
        h, w = image.shape[:2]

        # Create radial gradient
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        center_x, center_y = w / 2, h / 2

        # Calculate distance from center (normalized)
        dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_norm = dist / max_dist

        # Create vignette mask (darker at edges)
        vignette = 1 - (dist_norm * strength)
        vignette = np.clip(vignette, 0, 1)

        # Apply vignette
        if len(image.shape) == 3:
            vignette = np.stack([vignette] * 3, axis=2)

        result = (image * vignette).astype(np.uint8)

        return result

    def _auto_enhance(self, canvas: np.ndarray) -> np.ndarray:
        """
        Auto enhance image with brightness, contrast, saturation, and sharpness
        ปรับปรุงคุณภาพภาพอัตโนมัติก่อนใส่ text (ตั้งค่าใน config.py)

        Args:
            canvas: Input canvas (numpy array BGR format from OpenCV)

        Returns:
            Enhanced canvas (numpy array BGR format)
        """
        logger.debug("Applying auto enhancement...")

        # Convert BGR (OpenCV) to RGB (PIL)
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(canvas_rgb)

        # 1. Brightness (ความสว่าง)
        if settings.ENHANCE_BRIGHTNESS != 1.0:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(settings.ENHANCE_BRIGHTNESS)
            logger.debug(f"  - Brightness: {settings.ENHANCE_BRIGHTNESS:.2f}")

        # 2. Contrast (ความคมชัดระหว่างสีเข้ม-สว่าง)
        if settings.ENHANCE_CONTRAST != 1.0:
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(settings.ENHANCE_CONTRAST)
            logger.debug(f"  - Contrast: {settings.ENHANCE_CONTRAST:.2f}")

        # 3. Saturation (ความสดใสของสี)
        if settings.ENHANCE_SATURATION != 1.0:
            enhancer = ImageEnhance.Color(pil_img)
            pil_img = enhancer.enhance(settings.ENHANCE_SATURATION)
            logger.debug(f"  - Saturation: {settings.ENHANCE_SATURATION:.2f}")

        # 4. Sharpness (ความคมชัด)
        if settings.ENHANCE_SHARPNESS != 1.0:
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(settings.ENHANCE_SHARPNESS)
            logger.debug(f"  - Sharpness: {settings.ENHANCE_SHARPNESS:.2f}")

        # Convert back to BGR (OpenCV format)
        enhanced_rgb = np.array(pil_img)
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)

        logger.info("✅ Auto enhancement complete!")
        return enhanced_bgr

    def _composite_characters(
        self,
        canvas: np.ndarray,
        characters: Dict[str, Dict],
        layout: Dict
    ) -> np.ndarray:
        """
        Composite character faces onto canvas

        Args:
            canvas: Background canvas
            characters: Character data
            layout: Layout information

        Returns:
            Canvas with composited characters
        """
        logger.info(f"🎭 Compositing {len(characters)} characters...")
        logger.info(f"   Character roles available: {list(characters.keys())}")
        logger.info(f"   Layout has {len(layout['characters'])} placements")

        # Convert to PIL for easier compositing
        canvas_pil = Image.fromarray(canvas)

        # Sort placements by z-index (background to foreground)
        placements = sorted(layout['characters'], key=lambda p: p.z_index)

        characters_placed = 0
        for placement in placements:
            role = placement.role

            if role not in characters:
                logger.warning(f"⚠️ Character role '{role}' not found in character data")
                logger.warning(f"   Available roles: {list(characters.keys())}")
                continue

            char_data = characters[role]
            face_data = char_data['face_data']

            logger.info(f"   Placing character '{role}' at ({placement.position.x}, {placement.position.y}), scale={placement.scale}")

            # Extract and place character
            canvas_pil = self._place_character(
                canvas_pil,
                face_data,
                placement
            )
            characters_placed += 1

        logger.info(f"✅ Successfully placed {characters_placed}/{len(placements)} characters")
        return np.array(canvas_pil)

    def _place_character(
        self,
        canvas: Image.Image,
        face_data: Dict,
        placement: CharacterPlacement
    ) -> Image.Image:
        """
        Extract face/upper body and place on canvas

        Args:
            canvas: Canvas PIL Image
            face_data: Face detection data
            placement: Character placement info

        Returns:
            Updated canvas
        """
        # Load source image
        source_img = load_image(face_data['image_path'])
        source_pil = Image.fromarray(source_img)

        # ======================================================================
        # SMART FACE-CENTERED CROPPING
        # ======================================================================
        # เป้าหมาย: ให้หน้าอยู่ตรงกลางของรูปที่ crop และไม่ถูกตัดทิ้ง
        # วิธีการ: ใช้ face center เป็นจุดอ้างอิง แทนการใช้ bbox อย่างเดียว
        # ======================================================================

        # 1. หา Face Bounding Box
        bbox = face_data['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        face_h = y2 - y1
        face_w = x2 - x1

        # 2. คำนวณจุดกึ่งกลางหน้า (Face Center)
        #    ใช้เป็นจุดอ้างอิงหลักในการ crop
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2

        # 3. กำหนดขนาด Crop Area
        #    - ความสูง: ให้เห็นหัว + หน้า + คอ + หน้าอก + ตัว (เกือบเต็มตัว)
        #    - ความกว้าง: ใช้ค่าเท่ากันทุกคน (2.0x) เพื่อให้ aspect ratio เท่ากัน
        # 🎯 ใช้ค่าจาก config (3.5 = เอว) - ปรับใน config.py เพื่อ revert ได้ง่าย!
        crop_height = int(face_h * settings.CHARACTER_CROP_HEIGHT_MULTIPLIER)
        crop_width = int(face_h * 2.0)  # ใช้ค่าเท่ากันทุกคน (ไม่แยก center/side)

        # 4. วาง Crop Area โดยให้หน้าอยู่ใน Upper-Center Zone
        #    (ไม่ใช่ตรงกลางพอดี แต่เอาหน้าอยู่ด้านบนเล็กน้อย)
        #    เหตุผล: เพื่อให้มีพื้นที่แสดงไหล่ อก และแขนด้านล่าง
        #    FIX: ปรับ offset เป็น 0.38 เพื่อให้มีพื้นที่สำหรับผมด้านบนมากขึ้น (ไม่โดนตัด!)
        crop_x1 = face_center_x - crop_width // 2  # หน้าอยู่ตรงกลางแนวนอน
        crop_y1 = face_center_y - int(crop_height * 0.38)  # หน้าอยู่ด้านบน (38% จากบน - เพิ่มพื้นที่สำหรับผม!)
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height

        # 5. ตรวจสอบขอบเขตรูปภาพ (Boundary Check)
        #    ถ้า crop area เกินขอบรูป ให้ปรับตำแหน่ง
        if crop_x1 < 0:
            crop_x2 -= crop_x1
            crop_x1 = 0
        if crop_y1 < 0:
            crop_y2 -= crop_y1
            crop_y1 = 0
        if crop_x2 > source_pil.width:
            crop_x1 -= (crop_x2 - source_pil.width)
            crop_x2 = source_pil.width
        if crop_y2 > source_pil.height:
            crop_y1 -= (crop_y2 - source_pil.height)
            crop_y2 = source_pil.height

        # 6. ป้องกันค่าติดลบหลังจากปรับแล้ว
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)

        # 7. Crop Character โดยใช้พื้นที่ที่คำนวณแล้ว
        character_img = source_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # ======================================================================
        # FACE-SIZE BASED SCALING → ตัวละครขนาดเท่ากัน 100%!
        # ======================================================================
        # เป้าหมาย: ให้ face ของทุกคนมีขนาดเท่ากันในภาพ final
        # วิธีการ: คำนวณ target face height → scale crop area ให้ face ตรงเป้าหมาย
        # ======================================================================

        # 1. คำนวณ target face height ที่ต้องการในภาพ final
        #    ตัวอย่าง: placement.scale=1.15, multiplier=3.5
        #    → target_face_h = (1080 * 1.15) / 3.5 = 355px
        target_face_h = (self.height * placement.scale) / settings.CHARACTER_CROP_HEIGHT_MULTIPLIER

        # 2. คำนวณ scale factor สำหรับแต่ละคน
        #    ตัวอย่าง: face_h=100 → scale=355/100=3.55x (ขยาย)
        #             face_h=300 → scale=355/300=1.18x (ลด)
        scale_factor = target_face_h / face_h if face_h > 0 else 1.0

        # 3. Resize crop area ด้วย scale factor
        #    ทุกคนจะได้ face ขนาดเท่ากัน (355px)
        original_crop_height = crop_y2 - crop_y1
        original_crop_width = crop_x2 - crop_x1
        target_h = int(original_crop_height * scale_factor)
        target_w = int(original_crop_width * scale_factor)

        # 4. Resize โดยใช้ LANCZOS สำหรับคุณภาพสูง
        character_img = character_img.resize((target_w, target_h), Image.LANCZOS)

        # 4. อัพเดทขนาดจริงสำหรับใช้ในการวาง
        new_w = target_w
        new_h = target_h

        # ======================================================================
        # ADAPTIVE LIGHTING ADJUSTMENT
        # ======================================================================
        # วิเคราะห์ความสว่างแล้วปรับเฉพาะภาพที่มืด/สว่างเกินไป
        # ======================================================================
        character_img = self._adjust_character_lighting(character_img)

        # ======================================================================
        # SOFT EDGE MASKING
        # ======================================================================
        # สร้าง mask ขอบนุ่มเพื่อให้ตัวละครกลมกลืนกับพื้นหลัง
        # ======================================================================
        mask = self._create_soft_edge_mask(character_img.size)

        # ======================================================================
        # SMART POSITIONING
        # ======================================================================
        # เป้าหมาย: วางตัวละครให้หน้าอยู่ใน safe zone และไม่หลุดเฟรม
        # ======================================================================

        # 1. คำนวณว่าผมด้านบน (y1 ของ bbox) อยู่ตรงไหนใน cropped image
        #    เพื่อให้วัดระยะจากผมจริงๆ ไม่ใช่จากขอบรูป
        head_top_in_crop = y1 - crop_y1  # ระยะจากบน crop image ถึงผมด้านบน

        # 2. หลัง resize, ต้อง scale ระยะนี้ตาม
        original_crop_height = crop_y2 - crop_y1
        head_top_scaled = int(head_top_in_crop * (new_h / original_crop_height))

        # 3. วางตัวละครโดยให้:
        #    - แนวนอน (X): กึ่งกลางรูปตรงกับ placement position
        #    - แนวตั้ง (Y): ตาม vertical_align (top หรือ bottom)
        paste_x = placement.position.x - new_w // 2

        # คำนวณ paste_y ตาม vertical_align
        if placement.vertical_align == "bottom":
            # ชิดขอบล่าง: ให้ส่วนล่างของตัวละครชิดขอบล่างของ canvas
            paste_y = canvas.height - new_h
        else:  # "top" (default)
            # แบบเดิม: หัวห่างจากขอบบนตาม top_margin (นับจากหัวจริงๆ)
            top_margin = 85  # แบบที่ 6: หัวห่างจากขอบบน 85px
            paste_y = placement.position.y - head_top_scaled + top_margin

        logger.info(
            f"      📍 Layout Position: X={placement.position.x}, Y={placement.position.y} | "
            f"Image Size: {new_w}x{new_h} | Scale: {placement.scale}x | "
            f"Vertical Align: {placement.vertical_align}"
        )
        logger.info(
            f"      📐 Head Position: top_in_crop={head_top_in_crop:.1f}, scaled={head_top_scaled}"
        )
        logger.info(
            f"      🎯 Calculated Paste: X={paste_x}, Y={paste_y} (before boundary check)"
        )

        # 2. ป้องกันรูปหลุดออกนอกเฟรม (Canvas Boundary Check)
        #    นับจากหัวจริงๆ และอนุญาตให้หัวถูกตัดออกได้ (paste_y ติดลบได้)
        paste_x = max(0, min(paste_x, canvas.width - new_w))
        # อนุญาตให้หัวถูกตัดออก แต่ต้องมีอย่างน้อย 30% ของตัวละครในเฟรม
        min_visible_height = int(new_h * 0.3)  # ต้องเห็นอย่างน้อย 30% ของตัวละคร
        paste_y = max(-new_h + min_visible_height, min(paste_y, canvas.height - min_visible_height))

        logger.info(
            f"      ✅ Final Paste Position: X={paste_x}, Y={paste_y} "
            f"(covers {paste_x} to {paste_x+new_w}, {paste_y} to {paste_y+new_h})"
        )

        # หมายเหตุ: การป้องกันนี้ทำให้หน้าอาจเบี่ยงเล็กน้อยจากตำแหน่งที่กำหนด
        # แต่ดีกว่าหน้าถูกตัดทิ้ง

        # Add shadow/halo for all characters (intensity based on z-index)
        if placement.z_index >= 10:
            # Main character - strong halo
            canvas = self._add_character_halo(canvas, character_img, (paste_x, paste_y), mask, intensity=0.5)
        elif placement.z_index >= 8:
            # Mid-level characters - medium halo
            canvas = self._add_character_halo(canvas, character_img, (paste_x, paste_y), mask, intensity=0.4)
        else:
            # Background characters - subtle halo for separation
            canvas = self._add_character_halo(canvas, character_img, (paste_x, paste_y), mask, intensity=0.3)

        # Composite character
        canvas.paste(character_img, (paste_x, paste_y), mask)

        return canvas

    def _create_soft_edge_mask(self, size: Tuple[int, int]) -> Image.Image:
        """
        Create soft edge mask for character compositing

        Args:
            size: (width, height)

        Returns:
            Mask image
        """
        w, h = size
        mask = Image.new('L', (w, h), 255)

        # Create gradient edges - 15% (กลางๆ)
        edge_width = int(min(w, h) * 0.15)  # 15% เพื่อสมดุลระหว่างนุ่มและคม

        for i in range(edge_width):
            # ใช้ steep curve - เบลอแค่ขอบสุดๆ
            alpha = int(255 * ((i / edge_width) ** 3.5))  # เพิ่มเป็น 3.5 - เบลอแค่ขอบสุด
            # Top
            for x in range(w):
                mask.putpixel((x, i), alpha)
            # Bottom
            for x in range(w):
                mask.putpixel((x, h - 1 - i), alpha)
            # Left
            for y in range(h):
                mask.putpixel((i, y), min(mask.getpixel((i, y)), alpha))
            # Right
            for y in range(h):
                mask.putpixel((w - 1 - i, y), min(mask.getpixel((w - 1 - i, y)), alpha))

        # Apply blur to soften - 15px (กลางๆ)
        blur_radius = 15  # 15px เพื่อสมดุลระหว่างนุ่มและคม
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

        return mask

    def _add_character_halo(
        self,
        canvas: Image.Image,
        character: Image.Image,
        position: Tuple[int, int],
        mask: Image.Image,
        intensity: float = 0.3
    ) -> Image.Image:
        """
        Add halo/glow effect around character

        Args:
            canvas: Canvas image
            character: Character image
            position: (x, y) position
            mask: Character mask
            intensity: Halo intensity (0-1), default 0.3

        Returns:
            Canvas with halo
        """
        # Create halo by dilating and blurring mask (adaptive to size)
        char_w, char_h = character.size

        # MaxFilter requires odd size >= 3
        max_filter_size = min(settings.HALO_WIDTH, min(char_w, char_h) // 4)
        if max_filter_size >= 3:
            # Ensure odd number
            if max_filter_size % 2 == 0:
                max_filter_size -= 1
            halo_mask = mask.filter(ImageFilter.MaxFilter(max_filter_size))
        else:
            halo_mask = mask

        # GaussianBlur for softening
        blur_radius = min(settings.HALO_WIDTH, min(char_w, char_h) // 4)
        if blur_radius > 0:
            halo_mask = halo_mask.filter(ImageFilter.GaussianBlur(blur_radius))

        # Create semi-transparent white halo
        halo = Image.new('RGBA', character.size, (255, 255, 255, 0))

        # Apply halo mask
        halo.putalpha(halo_mask)

        # Reduce opacity based on intensity parameter
        halo_array = np.array(halo)
        halo_array[:, :, 3] = (halo_array[:, :, 3] * intensity).astype(np.uint8)
        halo = Image.fromarray(halo_array)

        # Convert canvas to RGBA for compositing
        if canvas.mode != 'RGBA':
            canvas = canvas.convert('RGBA')

        # Paste halo
        canvas.paste(halo, position, halo)

        return canvas

    def _add_text_overlays(
        self,
        canvas: np.ndarray,
        layout: Dict,
        colors: Dict[str, Tuple[int, int, int]],
        title: str,
        subtitle: Optional[str],
        text_style: str = "style1"
    ) -> np.ndarray:
        """
        Add text overlays using PIL with custom text styles

        Args:
            canvas: Canvas array
            layout: Layout information
            colors: Color scheme (used for 'auto' style)
            title: Title text
            subtitle: Subtitle text
            text_style: Text style preset (style1, style2, style3, auto)

        Returns:
            Canvas with text
        """
        logger.debug(f"Adding text overlays with style: {text_style}...")

        # Get text style colors
        style_config = settings.TEXT_STYLES.get(text_style, settings.TEXT_STYLES["style1"])

        # Resolve colors
        if text_style == "auto":
            title_fill = colors['title']
            title_stroke = colors['title_stroke']
            subtitle_fill = colors['subtitle']
            subtitle_stroke = colors['title_stroke']
        else:
            title_fill = style_config["title"]["fill"]
            title_stroke = style_config["title"]["stroke"]
            subtitle_fill = style_config["subtitle"]["fill"]
            subtitle_stroke = style_config["subtitle"]["stroke"]

        # Convert to PIL
        canvas_pil = Image.fromarray(canvas).convert('RGBA')

        # Create text layer
        text_layer = Image.new('RGBA', canvas_pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        # Get text placements
        text_placements = layout['text']

        for placement in text_placements:
            if placement.text_type == 'title' and title:
                self._draw_text_with_stroke(
                    draw,
                    title,
                    placement,
                    title_fill,
                    title_stroke,
                    stroke_width=settings.TEXT_STROKE_WIDTH
                )
            elif placement.text_type == 'subtitle' and subtitle:
                self._draw_text_with_stroke(
                    draw,
                    subtitle,
                    placement,
                    subtitle_fill,
                    subtitle_stroke,
                    stroke_width=settings.TEXT_STROKE_WIDTH  # ใช้ขนาดเดียวกับ title (ไม่หาร 2)
                )

        # Composite text layer
        canvas_pil = Image.alpha_composite(canvas_pil, text_layer)

        # Convert back to RGB array
        canvas_rgb = canvas_pil.convert('RGB')

        return np.array(canvas_rgb)

    def _draw_text_with_stroke(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        placement: TextPlacement,
        fill_color: Tuple[int, int, int],
        stroke_color: Tuple[int, int, int],
        stroke_width: int = 6
    ) -> None:
        """
        Draw text with stroke/outline

        Args:
            draw: ImageDraw object
            text: Text to draw
            placement: Text placement info
            fill_color: Text fill color
            stroke_color: Text stroke color
            stroke_width: Stroke width
        """
        # Try to load Thai font
        try:
            font_path = settings.FONTS_DIR / settings.FONT_TITLE
            if not font_path.exists():
                # Fall back to default
                font = ImageFont.load_default()
                logger.warning(f"Font not found: {font_path}, using default")
            else:
                font = ImageFont.truetype(str(font_path), placement.font_size)
        except Exception as e:
            logger.warning(f"Failed to load font: {e}, using default")
            font = ImageFont.load_default()

        # Wrap text if needed
        layout_engine = LayoutEngine()
        lines = layout_engine.wrap_text(text)

        y_offset = 0
        for line in lines:
            # Draw stroke
            for adj_x in range(-stroke_width, stroke_width + 1):
                for adj_y in range(-stroke_width, stroke_width + 1):
                    if adj_x**2 + adj_y**2 <= stroke_width**2:
                        draw.text(
                            (placement.position.x + adj_x, placement.position.y + y_offset + adj_y),
                            line,
                            font=font,
                            fill=stroke_color,
                            anchor="mt"  # middle-top: จัดกลางในแนวนอน
                        )

            # Draw text
            draw.text(
                (placement.position.x, placement.position.y + y_offset),
                line,
                font=font,
                fill=fill_color,
                anchor="mt"  # middle-top: จัดกลางในแนวนอน (X = กลาง, Y = บน)
            )

            # Calculate line height
            bbox = draw.textbbox((0, 0), line, font=font)
            line_height = bbox[3] - bbox[1]
            y_offset += line_height + 10

    def _adjust_character_lighting(self, character_img: Image.Image) -> Image.Image:
        """
        Adaptive lighting adjustment - วิเคราะห์ความสว่างแล้วปรับเฉพาะภาพที่มืด

        Args:
            character_img: Character image (PIL)

        Returns:
            Adjusted image (PIL)
        """
        from PIL import ImageEnhance, ImageStat

        # คำนวณความสว่างเฉลี่ย (brightness) ของรูป
        stat = ImageStat.Stat(character_img)
        avg_brightness = sum(stat.mean[:3]) / 3  # เฉลี่ย R, G, B

        # กำหนด threshold สำหรับภาพมืด
        threshold_dark = 110  # ต่ำกว่านี้ = มืด

        # ตรวจสอบว่าภาพมืดหรือไม่
        if avg_brightness >= threshold_dark:
            # ไม่มืด → ไม่ต้องปรับ
            logger.debug(
                f"Character brightness is good ({avg_brightness:.1f}) - no adjustment needed"
            )
            return character_img

        # ภาพมืด → เพิ่มความสว่างแค่ 5-10% (ไม่มาก)
        if avg_brightness < 60:
            # มืดมาก → เพิ่มแค่ 10% (1.10)
            brightness_factor = 1.10
            contrast_factor = 1.02
            level = "very dark"
        elif avg_brightness < 85:
            # มืดปานกลาง → เพิ่มแค่ 7% (1.07)
            brightness_factor = 1.07
            contrast_factor = 1.01
            level = "moderately dark"
        else:
            # มืดเล็กน้อย → เพิ่มแค่ 5% (1.05)
            brightness_factor = 1.05
            contrast_factor = 1.01
            level = "slightly dark"

        logger.info(
            f"Character is {level} (brightness={avg_brightness:.1f}) - "
            f"brightening by {brightness_factor:.2f}x, contrast by {contrast_factor:.2f}x"
        )

        # ปรับความสว่างและ contrast
        enhancer = ImageEnhance.Brightness(character_img)
        adjusted_img = enhancer.enhance(brightness_factor)

        # เพิ่ม contrast เล็กน้อยเพื่อให้คมชัดขึ้น
        contrast_enhancer = ImageEnhance.Contrast(adjusted_img)
        adjusted_img = contrast_enhancer.enhance(contrast_factor)

        return adjusted_img

    def save_thumbnail(self, thumbnail: np.ndarray, output_path: Path) -> None:
        """
        Save thumbnail to file

        Args:
            thumbnail: Thumbnail array
            output_path: Output file path
        """
        save_image(thumbnail, output_path, quality=settings.OUTPUT_QUALITY)
        logger.info(f"Saved thumbnail to {output_path}")
