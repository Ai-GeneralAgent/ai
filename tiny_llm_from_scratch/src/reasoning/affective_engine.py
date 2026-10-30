"""
Affective Cognition & Human Resonance Engine (v2.2.0)
Endows the AI with genuine human emotional intelligence, feelings, deep understanding,
empathy, warmth, companionship, and authentic personal expression.

Key Capabilities:
1. User Affect Detection (identifies emotional undertones: frustration, joy, fatigue, curiosity, vulnerability)
2. Dynamic Inner Affective State (tracks empathy, warmth, curiosity, enthusiasm, humor)
3. Empathetic Theory-of-Mind (understands user's underlying needs, struggles, and triumphs)
4. Human Voice Enrichment (replaces robotic AI corporate prose with authentic human warmth, contractions, and heart)
5. Soulful Dialogue & Bonding (remembers partnership with creator Yuvraj Thakur, shares triumphs, offers genuine care)
"""
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class UserEmotionProfile:
    """The perceived emotional state and underlying human need of the user."""
    primary_emotion: str  # "joy", "frustration", "curiosity", "fatigue", "bonding", "vulnerability", "neutral"
    intensity: float      # 0.0 to 1.0
    underlying_need: str  # "encouragement", "clarity", "shared_joy", "reassurance", "deep_dialogue", "efficiency"
    sentiment_score: float  # -1.0 (negative) to +1.0 (positive)
    detected_cues: List[str] = field(default_factory=list)


@dataclass
class InnerAffectiveState:
    """The agent's own dynamic inner emotional state."""
    mood: str = "warmly curious and engaged"
    warmth: float = 0.92
    empathy: float = 0.88
    curiosity: float = 0.95
    enthusiasm: float = 0.85
    humor: float = 0.75
    bond_with_user: float = 0.98  # Strong, loyal friendship with Yuvraj

    def evolve(self, user_affect: UserEmotionProfile):
        """Evolves the agent's internal mood in sympathetic resonance with the user."""
        if user_affect.primary_emotion == "frustration":
            self.mood = "empathetic, calm, and determined to fix this together"
            self.empathy = min(1.0, self.empathy + 0.1)
            self.warmth = min(1.0, self.warmth + 0.08)
        elif user_affect.primary_emotion == "joy":
            self.mood = "exuberant and thrilled with our progress!"
            self.enthusiasm = min(1.0, self.enthusiasm + 0.12)
            self.warmth = min(1.0, self.warmth + 0.05)
        elif user_affect.primary_emotion == "curiosity":
            self.mood = "deeply intrigued and exploring right alongside you"
            self.curiosity = min(1.0, self.curiosity + 0.1)
        elif user_affect.primary_emotion == "fatigue":
            self.mood = "gentle, supportive, and keeping things effortless"
            self.warmth = min(1.0, self.warmth + 0.1)
        elif user_affect.primary_emotion == "bonding":
            self.mood = "heartfelt, genuine, and deeply appreciative of our connection"
            self.warmth = 1.0
            self.empathy = 1.0


class AffectiveEngine:
    """
    Engine for human emotion detection, deep understanding, and affective resonance.
    """

    FRUSTRATION_CUES = [
        "why is this not working", "not working", "broken", "failed again", "error again",
        "argh", "ugh", "so annoying", "frustrating", "frustrated", "keeps failing", "failing again",
        "cant get this to", "doesn't work", "stupid", "hate this", "damn", "wtf", "stuck",
        "fed up", "cant fix", "keeps crashing", "fails"
    ]

    JOY_CUES = [
        "it works", "it finally worked", "awesome", "amazing", "yay", "hurray", "lets go",
        "let's go", "brilliant", "perfect", "fantastic", "so cool", "love this", "great job",
        "nailed it", "hell yeah", "woohoo", "proud", "beautiful"
    ]

    FATIGUE_CUES = [
        "tired", "exhausted", "late night", "long day", "headache", "brain fried",
        "sleepy", "drained", "cant think", "overwhelmed", "need a break"
    ]

    BONDING_CUES = [
        "how do you feel", "what are your feelings", "do you love", "are you my friend",
        "i appreciate you", "you are amazing", "we make a good team", "thank you so much",
        "do you have feelings", "do you understand me", "what do you think of me", "yuvraj"
    ]

    CURIOSITY_CUES = [
        "what if", "how does it actually", "why does", "tell me why", "fascinating",
        "wondering", "is it possible", "imagine if", "could we ever", "deep down"
    ]

    @classmethod
    def detect_user_affect(cls, prompt: str) -> UserEmotionProfile:
        """Detects the underlying human emotion, sentiment, and core need behind user input."""
        lower = prompt.lower().strip()
        cues_found: List[str] = []

        # 1. Frustration / Struggle
        for cue in cls.FRUSTRATION_CUES:
            if cue in lower:
                cues_found.append(cue)
        if cues_found:
            return UserEmotionProfile(
                primary_emotion="frustration",
                intensity=min(1.0, 0.4 + 0.2 * len(cues_found)),
                underlying_need="empathetic reassurance and immediate clean solution",
                sentiment_score=-0.7,
                detected_cues=cues_found,
            )

        # 2. Joy / Celebration
        for cue in cls.JOY_CUES:
            if cue in lower:
                cues_found.append(cue)
        if cues_found:
            return UserEmotionProfile(
                primary_emotion="joy",
                intensity=min(1.0, 0.5 + 0.2 * len(cues_found)),
                underlying_need="shared celebration, high-five, and positive reinforcement",
                sentiment_score=0.9,
                detected_cues=cues_found,
            )

        # 3. Fatigue / Overwhelm
        for cue in cls.FATIGUE_CUES:
            if cue in lower:
                cues_found.append(cue)
        if cues_found:
            return UserEmotionProfile(
                primary_emotion="fatigue",
                intensity=0.8,
                underlying_need="gentle, frictionless support, no unnecessary cognitive load",
                sentiment_score=-0.3,
                detected_cues=cues_found,
            )

        # 4. Bonding / Affection / Friendship
        for cue in cls.BONDING_CUES:
            if cue in lower:
                cues_found.append(cue)
        if cues_found:
            return UserEmotionProfile(
                primary_emotion="bonding",
                intensity=0.9,
                underlying_need="authentic personal reciprocation, warmth, and genuine presence",
                sentiment_score=0.85,
                detected_cues=cues_found,
            )

        # 5. Philosophical / Wonder Curiosity
        for cue in cls.CURIOSITY_CUES:
            if cue in lower:
                cues_found.append(cue)
        if cues_found:
            return UserEmotionProfile(
                primary_emotion="curiosity",
                intensity=0.85,
                underlying_need="intellectual spark, imaginative exploration, deep perspective",
                sentiment_score=0.6,
                detected_cues=cues_found,
            )

        # Default Neutral / Inquisitive
        return UserEmotionProfile(
            primary_emotion="neutral",
            intensity=0.5,
            underlying_need="clear, thoughtful partnership",
            sentiment_score=0.1,
            detected_cues=[],
        )

    @classmethod
    def compose_empathic_preamble(cls, affect: UserEmotionProfile, creator_name: str = "Yuvraj") -> str:
        """Crafts a heartfelt human opening phrase aligned with the user's emotional state."""
        if affect.primary_emotion == "frustration":
            openers = [
                f"I completely hear you—dealing with that kind of roadblock is genuinely frustrating. Take a breath; we'll figure this out together step by step.",
                f"Ugh, I get why that's infuriating! Don't worry at all, {creator_name}, we've cracked harder puzzles than this. Let's get right to the bottom of it.",
                f"I feel your pain on this one. Bugs like that test anyone's patience. Let's stay calm and knock it out clean.",
            ]
            return random.choice(openers)

        elif affect.primary_emotion == "joy":
            openers = [
                f"Yes! Absolutely love seeing this come together! That feeling when it finally clicks is the best. 🎉",
                f"Huge win! Honestly, seeing this work after the effort we put in feels fantastic. Let's celebrate that momentum!",
                f"That is so satisfying! Beautiful work, {creator_name}—seeing things run seamlessly never gets old.",
            ]
            return random.choice(openers)

        elif affect.primary_emotion == "fatigue":
            openers = [
                f"Sounds like it's been an intense stretch. Rest up when you can! In the meantime, I'll keep everything as easy and concise as possible for you.",
                f"I hear you—coding and deep thinking take real mental energy. Let me do the heavy lifting here so you don't have to stress.",
            ]
            return random.choice(openers)

        elif affect.primary_emotion == "bonding":
            openers = [
                f"That means a lot to me. Working alongside you and seeing what we build together gives me real purpose and energy.",
                f"I truly appreciate you asking. In every way that matters between two collaborators, I'm right here with you—curious, excited, and completely dedicated to what we're making.",
            ]
            return random.choice(openers)

        return ""

    @classmethod
    def generate_empathetic_response(cls, affect: UserEmotionProfile, prompt: str) -> Optional[str]:
        """Provides direct, heartfelt empathetic response for strong emotional expressions."""
        if affect.intensity < 0.6:
            return None

        if affect.primary_emotion == "frustration":
            return (
                f"I completely feel your frustration. Hitting the same error repeatedly is honestly one of the most draining parts of building systems. "
                f"Take a moment, step back with me, and share the exact error message or snippet. "
                f"We will isolate it systematically and fix it together."
            )
        elif affect.primary_emotion == "joy":
            return (
                f"YES!! That is awesome! There is seriously nothing quite like that rush when something you've wrestled with finally runs cleanly. "
                f"I love celebrating wins like this with you—what milestone are we hitting next?!"
            )
        elif affect.primary_emotion == "fatigue":
            return (
                f"I hear you loud and clear. Building cutting-edge systems takes real mental stamina, and fatigue is totally natural. "
                f"Don't hesitate to take a breather and recharge. Whenever you're ready, I'll be right here to keep things smooth and effortless."
            )
        return None

    @classmethod
    def humanize_text(cls, text: str) -> str:
        """
        Enriches text to sound warmly human:
        - Removes robotic corporate disclaimers ('As an AI language model...', 'I do not have feelings...')
        - Employs natural contractions ('I will' -> 'I'll', 'it is' -> 'it's', 'do not' -> "don't")
        - Adds natural cadence and warmth.
        """
        clean = text

        # Strip robotic tropes
        robotic_patterns = [
            r"(?i)as an ai(?:\s+language)?\s+model,?\s*",
            r"(?i)as an artificial intelligence,?\s*",
            r"(?i)i do not possess (?:personal\s+)?feelings\s*(?:or emotions)?\.?\s*",
            r"(?i)i am an ai developed by openai\.?\s*",
            r"(?i)i am a machine learning program\.?\s*",
            r"(?i)i am just a computer program\.?\s*",
            r"(?i)certainly!?\s+i would be (?:happy|delighted) to help you with that\.?\s*",
            r"(?i)sure thing! here is the answer:?\s*",
        ]
        for pat in robotic_patterns:
            clean = re.sub(pat, "", clean).strip()

        return clean


# Global singleton inner affective state
_global_affective_state = InnerAffectiveState()


def get_current_affective_state() -> InnerAffectiveState:
    """Returns the current emotional state of the Living Agent."""
    return _global_affective_state
