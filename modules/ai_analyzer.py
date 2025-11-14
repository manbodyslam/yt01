"""
AI Analyzer - Use GPT to analyze title and suggest optimal layout
"""

from typing import Dict, Optional
from loguru import logger
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not installed. AI features will be disabled.")

from config import settings


class AIAnalyzer:
    """
    Analyze title and content using GPT-4o-mini to suggest optimal layout
    VERY COST-EFFECTIVE: ~$0.001-0.003 per analysis
    """

    def __init__(self):
        """Initialize AI Analyzer"""
        # Debug logging
        logger.info(f"🔍 AI Analyzer initialization:")
        logger.info(f"  - OPENAI_AVAILABLE: {OPENAI_AVAILABLE}")
        logger.info(f"  - API key present: {bool(settings.OPENAI_API_KEY)}")
        logger.info(f"  - API key length: {len(settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else 0}")

        self.enabled = OPENAI_AVAILABLE and settings.OPENAI_API_KEY

        if self.enabled:
            try:
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("✅ AI Analyzer initialized with GPT-4o-mini")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenAI client: {e}")
                self.enabled = False
        else:
            logger.warning(f"⚠️ AI Analyzer disabled (OPENAI_AVAILABLE={OPENAI_AVAILABLE}, API_KEY={'present' if settings.OPENAI_API_KEY else 'missing'})")
            self.client = None

    def analyze_title(
        self,
        title: str,
        subtitle: Optional[str] = None,
        num_characters: int = 3
    ) -> Dict:
        """
        Analyze title and suggest optimal thumbnail settings

        Args:
            title: Main title text
            subtitle: Optional subtitle
            num_characters: Number of characters

        Returns:
            Dict with suggestions: {
                'layout_type': str,  # 'tri_hero', 'duo_focus', etc
                'text_style': str,   # 'style1', 'style2', 'style3'
                'text_position': str, # 'bottom', 'top', 'center'
                'mood': str,         # 'exciting', 'serious', 'funny'
                'reasoning': str     # Why these choices
            }
        """
        logger.info(f"🤖 analyze_title called - AI enabled: {self.enabled}")
        logger.info(f"   Title: '{title}', Num Characters: {num_characters}")

        if not self.enabled:
            logger.warning("⚠️ AI disabled, using fallback suggestions")
            return self._fallback_suggestions(title, num_characters)

        try:
            logger.info("🤖 Calling OpenAI API...")
            prompt = self._build_prompt(title, subtitle, num_characters)

            response = self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,  # gpt-4o-mini (ถูกที่สุด)
                messages=[
                    {
                        "role": "system",
                        "content": "You are a YouTube thumbnail design expert. Analyze titles and suggest optimal design choices."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=settings.OPENAI_MAX_TOKENS,
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            logger.info(f"✅ AI analysis complete!")
            logger.info(f"   Layout: {result.get('layout_type', 'N/A')}")
            logger.info(f"   Style: {result.get('text_style', 'N/A')}")
            logger.info(f"   Mood: {result.get('mood', 'N/A')}")
            logger.info(f"   Reasoning: {result.get('reasoning', 'N/A')}")

            return self._validate_and_clean(result)

        except Exception as e:
            logger.error(f"❌ AI analysis failed: {e}")
            logger.warning("⚠️ Falling back to keyword-based suggestions")
            return self._fallback_suggestions(title, num_characters)

    def _build_prompt(
        self,
        title: str,
        subtitle: Optional[str],
        num_characters: int
    ) -> str:
        """Build analysis prompt"""

        subtitle_text = f"\nSubtitle: {subtitle}" if subtitle else ""

        prompt = f"""Analyze this YouTube thumbnail title and suggest optimal design:

Title: {title}{subtitle_text}
Number of characters in thumbnail: {num_characters}

IMPORTANT: You MUST choose a layout that matches the number of characters ({num_characters}).

Available layouts:
FOR 1 CHARACTER:
- solo_focus: 1 person centered (individual focus, personal story)

FOR 2 CHARACTERS:
- duo_focus: 2 equal-sized people side-by-side (balanced, partnership, vs, comparison)
- duo_diagonal: 2 people diagonal (dynamic, action, conflict, tension)

FOR 3 CHARACTERS:
- tri_hero: 3 people with center person larger (balanced team, main character with support, friendly group)
- tri_pyramid: 1 main HERO in front + 2 smaller behind (clear leader, hierarchy, main character focus)
- tri_staggered: 3 people at different heights in diagonal (very dynamic, action-packed, energetic, exciting)

FOR 4+ CHARACTERS:
- quad_lineup: 4 people evenly spaced (large team, group ensemble, family, friends)

Text styles:
- style1: Red text + white stroke / Yellow text + black stroke (YouTube style - exciting, clickable)
- style2: White text + black stroke / Yellow text + black stroke (Classic - readable, serious)
- style3: White text + orange stroke / Gold text + black stroke (Vibrant - energetic, fun)

Based on the title's MOOD and having {num_characters} characters, suggest:
1. Layout type (MUST match character count {num_characters})
2. Text style that matches the mood
3. Text position (bottom recommended for most)
4. Overall mood/emotion
5. Brief reasoning

Return JSON:
{{
    "layout_type": "tri_hero",
    "text_style": "style1",
    "text_position": "bottom",
    "mood": "funny",
    "reasoning": "Title suggests comedy/entertainment..."
}}"""

        return prompt

    def _validate_and_clean(self, result: Dict) -> Dict:
        """Validate and clean AI response"""

        # Default values
        defaults = {
            'layout_type': 'tri_hero',
            'text_style': 'style1',
            'text_position': 'bottom',
            'mood': 'neutral',
            'reasoning': 'Default suggestions'
        }

        # Validate layout_type
        valid_layouts = ['solo_focus', 'duo_focus', 'duo_diagonal', 'tri_hero', 'tri_pyramid', 'tri_staggered', 'quad_lineup']
        if result.get('layout_type') not in valid_layouts:
            result['layout_type'] = defaults['layout_type']

        # Validate text_style
        valid_styles = ['style1', 'style2', 'style3']
        if result.get('text_style') not in valid_styles:
            result['text_style'] = defaults['text_style']

        # Validate text_position
        valid_positions = ['bottom', 'top', 'center']
        if result.get('text_position') not in valid_positions:
            result['text_position'] = defaults['text_position']

        # Fill missing fields
        for key, value in defaults.items():
            if key not in result:
                result[key] = value

        return result

    def _fallback_suggestions(self, title: str, num_characters: int) -> Dict:
        """Fallback suggestions when AI is disabled"""

        logger.info("📝 Using fallback keyword-based suggestions...")

        # Simple keyword-based suggestions
        title_lower = title.lower()

        # Detect mood from keywords
        if any(word in title_lower for word in ['ฮา', 'ตลก', 'ขำ', 'สนุก', 'เฮฮา']):
            mood = 'funny'
            text_style = 'style1'  # Red/Yellow - exciting
        elif any(word in title_lower for word in ['สยอง', 'กลัว', 'ผี', 'หลอน']):
            mood = 'scary'
            text_style = 'style3'  # White/Gold - dramatic
        elif any(word in title_lower for word in ['รัก', 'โรแมนติก', 'หวาน']):
            mood = 'romantic'
            text_style = 'style2'  # White/Yellow - classic
        else:
            mood = 'neutral'
            text_style = 'style1'  # Default

        # Select layout based on character count
        layout_map = {
            1: 'solo_focus',
            2: 'duo_focus',
            3: 'tri_hero',
            4: 'quad_lineup'
        }
        layout_type = layout_map.get(num_characters, 'tri_hero')

        result = {
            'layout_type': layout_type,
            'text_style': text_style,
            'text_position': 'bottom',
            'mood': mood,
            'reasoning': f'Keyword-based suggestion (AI disabled) - {mood} mood detected'
        }

        logger.info(f"📝 Fallback suggestions: {layout_type}, {text_style}, {mood}")

        return result
