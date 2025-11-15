"""
Layout Engine - Determine layout positions for characters and text
"""

import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw
from loguru import logger
from dataclasses import dataclass

from config import settings


@dataclass
class Position:
    """Position with x, y coordinates"""
    x: int
    y: int


@dataclass
class CharacterPlacement:
    """Character placement information"""
    role: str
    position: Position
    scale: float
    z_index: int
    vertical_align: str = "top"  # "top" or "bottom"


@dataclass
class TextPlacement:
    """Text placement information"""
    text_type: str  # 'title' or 'subtitle'
    position: Position
    max_width: int
    font_size: int
    alignment: str  # 'left', 'center', 'right'


class LayoutEngine:
    """
    Determines layout based on number of characters
    Supports: DuoFocus (2), TriHero (3), QuadLineup (4)
    """

    def __init__(self):
        """
        Initialize Layout Engine
        """
        self.width = settings.OUTPUT_WIDTH
        self.height = settings.OUTPUT_HEIGHT
        self.margin = settings.TEXT_SAFE_AREA_MARGIN

        self.main_scale = settings.MAIN_CHARACTER_SCALE
        self.secondary_scale = settings.SECONDARY_CHARACTER_SCALE

        logger.info(f"LayoutEngine initialized ({self.width}x{self.height})")

    def _calculate_optimal_font_size(
        self,
        text: str,
        default_size: int,
        max_width: int,
        font_path: Path
    ) -> int:
        """
        Calculate optimal font size for text to fit within max_width

        Args:
            text: Text to measure
            default_size: Default font size (from config)
            max_width: Maximum allowed width in pixels
            font_path: Path to font file

        Returns:
            Optimal font size (minimum 50% of default)
        """
        try:
            # Load font at default size
            font = ImageFont.truetype(str(font_path), default_size)

            # Create a temporary image to measure text
            temp_img = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(temp_img)

            # Measure text width at default size
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]

            # If text fits, return default size
            if text_width <= max_width:
                return default_size

            # Calculate scale factor needed to fit text
            scale_factor = max_width / text_width

            # Apply scale factor with 5% safety margin
            new_size = int(default_size * scale_factor * 0.95)

            # Enforce minimum size (50% of default)
            min_size = int(default_size * 0.5)
            optimal_size = max(new_size, min_size)

            logger.info(
                f"📏 Auto-resize: '{text[:30]}...' "
                f"{default_size}px → {optimal_size}px "
                f"(width: {text_width}px > {max_width}px)"
            )

            return optimal_size

        except Exception as e:
            logger.warning(f"⚠️ Font resize calculation failed: {e}, using default size")
            return default_size

    def _add_position_variation(self, x: int, y: int, variation_percent: float = 0.03) -> Position:
        """
        Add slight random variation to position for variety

        Args:
            x: Original x coordinate
            y: Original y coordinate
            variation_percent: Max variation as percentage of image dimensions (default 3%)

        Returns:
            Position with random variation applied
        """
        # Calculate max variation in pixels
        max_x_variation = int(self.width * variation_percent)
        max_y_variation = int(self.height * variation_percent)

        # Add random variation (±variation)
        x_offset = random.randint(-max_x_variation, max_x_variation)
        y_offset = random.randint(-max_y_variation, max_y_variation)

        new_x = max(0, min(self.width, x + x_offset))
        new_y = max(0, min(self.height, y + y_offset))

        return Position(x=new_x, y=new_y)

    def select_layout(self, num_characters: int, layout_type: str = None) -> str:
        """
        Select layout based on number of characters

        Args:
            num_characters: Number of characters (1-4)
            layout_type: Optional explicit layout type

        Returns:
            Layout name that matches num_characters
        """
        # Define all available layouts by character count
        LAYOUTS_BY_COUNT = {
            1: ["solo_focus"],
            2: ["duo_focus", "duo_diagonal"],
            3: ["tri_hero", "tri_pyramid", "tri_staggered"],
            4: ["quad_lineup"]
        }

        # Layout requirements mapping (for validation)
        LAYOUT_REQUIREMENTS = {
            'solo_focus': 1,
            'duo_focus': 2,
            'duo_diagonal': 2,
            'tri_hero': 3,
            'tri_pyramid': 3,
            'tri_staggered': 3,
            'quad_lineup': 4
        }

        # If explicit layout provided, validate it matches num_characters
        if layout_type:
            required_count = LAYOUT_REQUIREMENTS.get(layout_type)
            if required_count == num_characters:
                logger.info(f"🎯 Using specified layout: {layout_type} (for {num_characters} character(s))")
                return layout_type
            else:
                logger.warning(
                    f"⚠️ Layout mismatch: '{layout_type}' requires {required_count} chars "
                    f"but have {num_characters} chars. Will select appropriate layout."
                )

        # Select random layout from available layouts for this character count
        available_layouts = LAYOUTS_BY_COUNT.get(num_characters, ["solo_focus"])
        selected = random.choice(available_layouts)
        logger.info(f"🎲 Random layout selection: {selected} (for {num_characters} character(s))")
        return selected

    def create_layout(
        self,
        characters: Dict[str, Dict],
        layout_type: str = None,
        custom_positions: Optional[List[Dict]] = None,
        vertical_align: str = "top",
        title: str = "",
        subtitle: str = ""
    ) -> Dict:
        """
        Create complete layout for characters and text

        Args:
            characters: Character data from FaceService
            layout_type: Optional explicit layout type (from AI suggestion)
            custom_positions: Optional custom positions [{"x": 100, "y": 50, "scale": 2.0}, ...]
            vertical_align: Vertical alignment for characters
            title: Title text (for auto font resize calculation)
            subtitle: Subtitle text (for auto font resize calculation)

        Returns:
            Layout dictionary with character and text placements
        """
        num_chars = len(characters)
        layout_type = self.select_layout(num_chars, layout_type)

        logger.info(f"Creating '{layout_type}' layout for {num_chars} character(s)")

        # If custom positions provided, use them
        if custom_positions:
            logger.info(f"🎨 Using custom positions provided by user")
            char_placements = self._layout_custom(characters, custom_positions)
        else:
            # Generate character placements using preset layouts
            if layout_type == "solo_focus":
                char_placements = self._layout_solo_focus(characters)
            elif layout_type == "duo_focus":
                char_placements = self._layout_duo_focus(characters)
            elif layout_type == "duo_diagonal":
                char_placements = self._layout_duo_diagonal(characters)
            elif layout_type == "tri_hero":
                char_placements = self._layout_tri_hero(characters)
            elif layout_type == "tri_pyramid":
                char_placements = self._layout_tri_pyramid(characters)
            elif layout_type == "tri_staggered":
                char_placements = self._layout_tri_staggered(characters)
            elif layout_type == "quad_lineup":
                char_placements = self._layout_quad_lineup(characters)
            else:
                char_placements = []

        # Set vertical_align for all character placements
        for placement in char_placements:
            placement.vertical_align = vertical_align

        # Generate text placements with auto font resize
        text_placements = self._create_text_layout(char_placements, title, subtitle)

        layout = {
            'type': layout_type,
            'characters': char_placements,
            'text': text_placements,
        }

        return layout

    def _layout_solo_focus(self, characters: Dict[str, Dict]) -> List[CharacterPlacement]:
        """
        Solo Focus Layout - Single character centered or slightly off-center

        Args:
            characters: Character data

        Returns:
            List of character placements
        """
        main_char = list(characters.values())[0]

        # Base position with random variation for variety
        base_x = int(self.width * 0.65)
        base_y = int(self.height * 0.15)
        position = self._add_position_variation(base_x, base_y)

        placements = [
            CharacterPlacement(
                role="main",
                position=position,
                scale=0.82,  # ลด scale เพิ่มเพื่อไม่ให้ขาโดนตัด (จาก 0.90 → 0.82)
                z_index=10
            )
        ]

        return placements

    def _layout_duo_focus(self, characters: Dict[str, Dict]) -> List[CharacterPlacement]:
        """
        🔥 Duo Focus Layout - 2 คนใหญ่เหมือนแบบ 3 คน + ภาพกว้างตรงกลาง (Large + Wide Shot)

        จุดเด่น:
        - 2 คนใหญ่มาก (scale 1.15) เหมือนแบบ 3 คน - ดูโดดเด่น
        - วางซ้าย-ขวา ห่างกันพอดี เว้นพื้นที่กลางสำหรับภาพกว้าง
        - ชิดบนสุด (y=0) เหมือนแบบ 3 คน - ดูมีพลัง
        - ⚠️ แยกจากแบบ 3 คน โดยสิ้นเชิง - ไม่ยุ่งกัน!

        Args:
            characters: Character data

        Returns:
            List of character placements
        """
        chars_list = list(characters.items())

        placements = [
            # ตัวละครซ้าย - ใหญ่เหมือนแบบ 3 คน (1.15x)
            CharacterPlacement(
                role=chars_list[0][0],
                position=Position(
                    x=int(self.width * 0.25),  # ซ้าย 25% (เว้นพื้นที่กลาง)
                    y=0  # ชิดบนสุดเหมือนแบบ 3 คน
                ),
                scale=1.15,  # 🔥 ใหญ่เหมือนแบบ 3 คน! (เพิ่มจาก 0.85 → 1.15)
                z_index=10
            ),
            # ตัวละครขวา - ใหญ่เหมือนแบบ 3 คน (1.15x)
            CharacterPlacement(
                role=chars_list[1][0],
                position=Position(
                    x=int(self.width * 0.75),  # ขวา 75% (เว้นพื้นที่กลาง)
                    y=0  # ชิดบนสุดเหมือนแบบ 3 คน
                ),
                scale=1.15,  # 🔥 ใหญ่เหมือนแบบ 3 คน! (เพิ่มจาก 0.85 → 1.15)
                z_index=10
            )
        ]

        # 💡 หมายเหตุ: พื้นที่กลาง (25%-75%) เหลือว่างสำหรับภาพกว้าง (wide shot)
        # ภาพพื้นหลังที่เบลอจะแสดงเหตุการณ์น่าสนใจตรงกลาง

        return placements

    def _layout_duo_diagonal(self, characters: Dict[str, Dict]) -> List[CharacterPlacement]:
        """
        Duo Diagonal Layout - 2 คนวางเฉียง ไดนามิก (Dynamic Diagonal)

        จุดเด่น:
        - วางเป็นแนวทแยง บน-ซ้าย กับ ล่าง-ขวา
        - ตัวบนใหญ่กว่า (scale 1.2) เป็นจุดสนใจ
        - ตัวล่างเล็กกว่า (scale 1.0) รองรับ
        - สร้าง visual flow ดูมีพลังและเคลื่อนไหว
        - เหมาะกับเนื้อหาแอคชั่น/ตื่นเต้น

        Args:
            characters: Character data

        Returns:
            List of character placements
        """
        chars_list = list(characters.items())

        placements = [
            # ตัวละครบน-ซ้าย (ใหญ่กว่า)
            CharacterPlacement(
                role=chars_list[0][0],
                position=Position(
                    x=int(self.width * 0.28),  # ซ้าย
                    y=int(self.height * 0.12)  # ขยับหัวสูงขึ้นอีก +10% (จาก 0.22 → 0.12)
                ),
                scale=0.95,  # ลด scale เพิ่มเพื่อไม่ให้ขาโดนตัด (จาก 1.05 → 0.95)
                z_index=11
            ),
            # ตัวละครล่าง-ขวา (เล็กกว่า)
            CharacterPlacement(
                role=chars_list[1][0],
                position=Position(
                    x=int(self.width * 0.72),  # ขวา
                    y=int(self.height * 0.25)  # ขยับหัวสูงขึ้นอีก +10% (จาก 0.35 → 0.25)
                ),
                scale=0.75,  # ลด scale เพิ่มเพื่อไม่ให้ขาโดนตัด (จาก 0.85 → 0.75)
                z_index=9
            )
        ]

        return placements

    def _layout_tri_hero(self, characters: Dict[str, Dict]) -> List[CharacterPlacement]:
        """
        Tri-Hero Layout - 3 คนเรียงแถว แบบสมดุล (Balanced Lineup)

        จุดเด่น:
        - ตัวละคร 3 คนขนาดเท่ากัน แต่ไม่เท่ากันทุกคน (มีความหลากหลาย)
        - ตัวกลางใหญ่ที่สุด (scale 1.2) เป็นจุดสนใจหลัก
        - ซ้าย-ขวา ขนาดกลาง (scale 1.0) รองรับตัวกลาง
        - เว้นระยะพอดี ไม่ทับกัน ดูเป็นทีม

        Args:
            characters: Character data

        Returns:
            List of character placements
        """
        chars_list = list(characters.items())

        placements = [
            # ตัวละครซ้าย - 1.15x (ลดขนาดเพื่อไม่ทับกัน) หัวชิดบนสุด
            CharacterPlacement(
                role=chars_list[0][0],
                position=Position(
                    x=int(self.width * 0.21),  # ซ้าย 21% (ห่างจากขอบซ้ายมากขึ้นเพื่อไม่ทับกัน)
                    y=0  # ชิดบนสุด
                ),
                scale=1.15,  # ลดจาก 1.25 เป็น 1.15 (เพื่อไม่ให้ทับกัน)
                z_index=9
            ),
            # ตัวละครกลาง - 1.15x (ลดขนาดเพื่อไม่ทับกัน) หัวชิดบนสุด
            CharacterPlacement(
                role=chars_list[1][0],
                position=Position(
                    x=int(self.width * 0.50),  # ตรงกลาง 50%
                    y=0  # ชิดบนสุด
                ),
                scale=1.15,  # ลดจาก 1.25 เป็น 1.15 (เพื่อไม่ให้ทับกัน)
                z_index=10
            ),
            # ตัวละครขวา - 1.15x (ลดขนาดเพื่อไม่ทับกัน) หัวชิดบนสุด
            CharacterPlacement(
                role=chars_list[2][0],
                position=Position(
                    x=int(self.width * 0.79),  # ขวา 79% (ห่างจากขอบขวามากขึ้นเพื่อไม่ทับกัน)
                    y=0  # ชิดบนสุด
                ),
                scale=1.15,  # ลดจาก 1.25 เป็น 1.15 (เพื่อไม่ให้ทับกัน)
                z_index=9
            )
        ]

        return placements

    def _layout_tri_pyramid(self, characters: Dict[str, Dict]) -> List[CharacterPlacement]:
        """
        Tri-Pyramid Layout - โฟกัสตัวหลัก มีตัวรอง 2 คนด้านหลัง (Hero Focus)

        จุดเด่น:
        - ตัวหลักใหญ่มาก (scale 1.3) อยู่ด้านหน้า - เป็น HERO
        - ตัวรอง 2 คน เล็กกว่า (scale 0.75) อยู่ด้านหลังสูงขึ้น
        - สร้างความลึก (depth) ด้วย z-index
        - เหมาะกับเนื้อหาที่มีตัวเอกชัดเจน

        Args:
            characters: Character data

        Returns:
            List of character placements
        """
        chars_list = list(characters.items())

        placements = [
            # ตัวหลัก - HERO (ด้านหน้า กลางจอ ใหญ่มาก)
            CharacterPlacement(
                role=chars_list[0][0],
                position=Position(
                    x=int(self.width * 0.50),  # ตรงกลางพอดี
                    y=int(self.height * 0.28)  # ปรับให้หัวสูง 10% จากบน (ลดจาก 0.55 - เพราะ scale ใหญ่ที่สุด)
                ),
                scale=1.3,  # ใหญ่มาก - เป็นตัวเอก
                z_index=12  # อยู่หน้าสุด
            ),
            # ตัวรองซ้าย (ด้านหลัง สูงขึ้น เล็กกว่า)
            CharacterPlacement(
                role=chars_list[1][0],
                position=Position(
                    x=int(self.width * 0.22),  # ซ้าย
                    y=int(self.height * 0.25)  # ปรับให้หัวสูง 10% จากบน (ลดจาก 0.35)
                ),
                scale=0.75,  # เล็กกว่าตัวหลัก
                z_index=8   # อยู่หลังตัวหลัก
            ),
            # ตัวรองขวา (ด้านหลัง สูงขึ้น เล็กกว่า)
            CharacterPlacement(
                role=chars_list[2][0],
                position=Position(
                    x=int(self.width * 0.78),  # ขวา
                    y=int(self.height * 0.25)  # ปรับให้หัวสูง 10% จากบน (เท่ากับซ้าย)
                ),
                scale=0.75,  # เล็กกว่าตัวหลัก
                z_index=8   # อยู่หลังตัวหลัก
            )
        ]

        return placements

    def _layout_tri_staggered(self, characters: Dict[str, Dict]) -> List[CharacterPlacement]:
        """
        Tri-Staggered Layout - 3 คนวางเป็นชั้นๆ ไดนามิก (Dynamic Diagonal)

        จุดเด่น:
        - วางเป็นลำดับชั้น สูง-กลาง-ต่ำ
        - ขนาดแตกต่างกัน สร้างจังหวะ
        - ดูมีพลัง เคลื่อนไหว เหมาะกับเนื้อหาแอคชั่น/ตื่นเต้น
        - มี depth และ visual flow

        Args:
            characters: Character data

        Returns:
            List of character placements
        """
        chars_list = list(characters.items())

        placements = [
            # ตัวละครซ้าย - สูงสุด ขนาดกลาง
            CharacterPlacement(
                role=chars_list[0][0],
                position=Position(
                    x=int(self.width * 0.18),  # ซ้ายสุด
                    y=int(self.height * 0.30)  # ปรับให้หัวสูง 10% จากบน (ลดจาก 0.38)
                ),
                scale=0.95,  # ขนาดกลาง
                z_index=9
            ),
            # ตัวละครกลาง - กลางจอ ใหญ่ที่สุด
            CharacterPlacement(
                role=chars_list[1][0],
                position=Position(
                    x=int(self.width * 0.50),  # ตรงกลาง
                    y=int(self.height * 0.32)  # ปรับให้หัวสูง 10% จากบน (ลดจาก 0.48)
                ),
                scale=1.15,  # ใหญ่ที่สุด - เป็นจุดสนใจ
                z_index=11
            ),
            # ตัวละครขวา - ต่ำสุด ขนาดกลาง
            CharacterPlacement(
                role=chars_list[2][0],
                position=Position(
                    x=int(self.width * 0.82),  # ขวาสุด
                    y=int(self.height * 0.40)  # ปรับให้หัวสูงขึ้น (ลดจาก 0.58)
                ),
                scale=0.90,  # เล็กกว่ากลาง
                z_index=10
            )
        ]

        return placements

    def _layout_quad_lineup(self, characters: Dict[str, Dict]) -> List[CharacterPlacement]:
        """
        Quad Lineup Layout - 4 คนเรียงแถว เท่าเทียมกัน (YouTube Thumbnail Style)

        จุดเด่น:
        - ตัวละคร 4 คนขนาดใหญ่ (scale 0.95) เพื่อให้เห็นหน้าชัด
        - วางต่ำลงมา (y = 0.52) เพื่อให้มีพื้นที่ข้อความด้านล่าง
        - เว้นระยะเท่าๆ กัน (13%, 36%, 64%, 87%) เพื่อไม่ให้ทับกัน
        - ขนาดเล็กกว่า tri_hero เล็กน้อย (0.95 vs 1.1) เพราะต้องใส่ 4 คน

        Args:
            characters: Character data

        Returns:
            List of character placements
        """
        chars_list = list(characters.items())

        # การจัดวางแบบ YouTube Thumbnail (4 คน):
        # - ตัวละครต้องใหญ่และชัดเจน แต่ไม่ให้ทับกัน
        # - หน้าต้องอยู่ใน safe zone (ไม่ติดขอบบนหรือล่าง)
        # - เว้นระยะพอให้มีพื้นที่ข้อความด้านล่าง

        # Evenly spaced positions: 13%, 36%, 64%, 87%
        # (ความห่างระหว่างตัว ≈ 23-28% เพื่อป้องกันการทับกัน)
        # ปรับ y จาก 0.52 เป็น 0.35 เพื่อให้หัวนักแสดงสูง 10% จากบน
        positions = [
            # Character 1 (far left)
            (0.13, 0.35, 0.95, 10),
            # Character 2 (center-left)
            (0.36, 0.35, 0.95, 10),
            # Character 3 (center-right)
            (0.64, 0.35, 0.95, 10),
            # Character 4 (far right)
            (0.87, 0.35, 0.95, 10),
        ]

        placements = []
        for i, (role, char_data) in enumerate(chars_list[:4]):
            x_ratio, y_ratio, scale, z = positions[i]

            placements.append(
                CharacterPlacement(
                    role=role,
                    position=Position(
                        x=int(self.width * x_ratio),
                        y=int(self.height * y_ratio)
                    ),
                    scale=scale,
                    z_index=z
                )
            )

        return placements

    def _layout_custom(self, characters: Dict[str, Dict], custom_positions: List[Dict]) -> List[CharacterPlacement]:
        """
        Custom Layout - User-defined positions

        Args:
            characters: Character data
            custom_positions: List of {"x": int, "y": int, "scale": float} for each character

        Returns:
            List of character placements
        """
        chars_list = list(characters.items())
        placements = []

        # Match custom positions to characters
        for i, (role, char_data) in enumerate(chars_list):
            if i < len(custom_positions):
                pos = custom_positions[i]
                x = pos.get('x', self.width // 2)
                y = pos.get('y', self.height // 2)
                scale = pos.get('scale', 1.0)
                z_index = pos.get('z_index', 10)

                placements.append(
                    CharacterPlacement(
                        role=role,
                        position=Position(x=x, y=y),
                        scale=scale,
                        z_index=z_index
                    )
                )
                logger.info(f"Custom position for {role}: x={x}, y={y}, scale={scale}")
            else:
                # If not enough custom positions, use defaults
                logger.warning(f"Not enough custom positions for {role}, using default")
                placements.append(
                    CharacterPlacement(
                        role=role,
                        position=Position(x=self.width // 2, y=self.height // 2),
                        scale=1.0,
                        z_index=10
                    )
                )

        return placements

    def _create_text_layout(
        self,
        char_placements: List[CharacterPlacement],
        title: str = "",
        subtitle: str = ""
    ) -> List[TextPlacement]:
        """
        Create text layout that doesn't overlap with characters
        Auto-resize font if text is too long

        Args:
            char_placements: Character placements
            title: Title text (for auto font resize)
            subtitle: Subtitle text (for auto font resize)

        Returns:
            List of text placements
        """
        # NEW: Place text at BOTTOM of image
        # Title at bottom, subtitle below title

        # Calculate positions from bottom
        bottom_margin = self.margin + 40

        # Move text up by 0.75% from bottom (1080 * 0.0075 = 8.1 ≈ 8 pixels) - ลดอีก 50%
        move_up_offset = int(self.height * 0.0075)  # 8 pixels (ลดจาก 16 pixels)

        # Calculate max width for text
        max_text_width = self.width - (2 * self.margin)

        # Get font path
        font_path = settings.FONTS_DIR / settings.FONT_TITLE

        # 🔥 AUTO FONT RESIZE: Calculate optimal font sizes
        title_font_size = self._calculate_optimal_font_size(
            text=title or "Sample Title",
            default_size=settings.TITLE_FONT_SIZE,
            max_width=max_text_width,
            font_path=font_path
        )

        subtitle_font_size = self._calculate_optimal_font_size(
            text=subtitle or "Sample Subtitle",
            default_size=settings.SUBTITLE_FONT_SIZE,
            max_width=max_text_width,
            font_path=font_path
        )

        # Subtitle is at the very bottom (moved down by 20px from original)
        # Use actual subtitle font size for positioning
        subtitle_y = self.height - bottom_margin - subtitle_font_size - move_up_offset + 70

        # Title is above subtitle (same spacing as before, will move down 20px automatically)
        # Use actual title font size for positioning
        title_y = subtitle_y - title_font_size + 10

        title_placement = TextPlacement(
            text_type="title",
            position=Position(
                x=self.width // 2,  # Center X
                y=title_y
            ),
            max_width=max_text_width,
            font_size=title_font_size,  # 🔥 Use calculated optimal size
            alignment="center"
        )

        subtitle_placement = TextPlacement(
            text_type="subtitle",
            position=Position(
                x=self.width // 2,  # Center X
                y=subtitle_y
            ),
            max_width=max_text_width,
            font_size=subtitle_font_size,  # 🔥 Use calculated optimal size
            alignment="center"
        )

        return [title_placement, subtitle_placement]

    def calculate_safe_zones(self, char_placements: List[CharacterPlacement]) -> List[Tuple[int, int, int, int]]:
        """
        Calculate zones where text should not be placed

        Args:
            char_placements: Character placements

        Returns:
            List of rectangles (x, y, w, h) representing unsafe zones
        """
        unsafe_zones = []

        for placement in char_placements:
            # Estimate character bounding box
            # Assume character face is roughly 200x300 pixels at scale 1.0
            base_w = 200
            base_h = 300

            w = int(base_w * placement.scale)
            h = int(base_h * placement.scale)

            x = placement.position.x - w // 2
            y = placement.position.y - h // 2

            # Add padding
            padding = 50
            unsafe_zones.append((
                x - padding,
                y - padding,
                w + 2 * padding,
                h + 2 * padding
            ))

        return unsafe_zones

    def wrap_text(self, text: str, max_words_per_line: int = None) -> List[str]:
        """
        Wrap text into multiple lines

        Args:
            text: Input text
            max_words_per_line: Maximum words per line

        Returns:
            List of text lines
        """
        max_words = max_words_per_line or settings.TEXT_MAX_WORDS_PER_LINE

        words = text.split()
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)

            if len(current_line) >= max_words:
                lines.append(" ".join(current_line))
                current_line = []

        # Add remaining words
        if current_line:
            lines.append(" ".join(current_line))

        return lines
