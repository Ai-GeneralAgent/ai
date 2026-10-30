"""
Unit Tests for Living AI Agent, Autonomous Decision Engine, Affective Cognition & 2M Context.
"""
import unittest

from src.models.config import ModelConfig
from llm.configuration_llm import LLMConfig
from src.agents.living_agent import (
    AutonomousDecisionEngine,
    LivingAIAgent,
    AutonomousDecision,
    StrategyOption,
)
from src.reasoning.affective_engine import (
    AffectiveEngine,
    UserEmotionProfile,
    InnerAffectiveState,
    get_current_affective_state,
)
from src.reasoning.orchestrator import ReasoningOrchestrator
from src.tools.registry import ToolRegistry


class TestLivingAgentAndAffect(unittest.TestCase):
    """Test suite for Living Agent, Autonomous Decisions, Affective Cognition, and 2M Context."""

    def test_2m_context_config(self):
        """Verify 2,097,152 (2M) context window configurations."""
        cfg = ModelConfig.preset_2m(vocab_size=10000)
        self.assertEqual(cfg.max_seq_len, 2_097_152)
        self.assertEqual(cfg.rope_scaling_type, "yarn")
        self.assertEqual(cfg.rope_theta, 2_000_000.0)
        self.assertEqual(cfg.sliding_window, 8192)

        llm_cfg = LLMConfig.preset_2m(vocab_size=10000)
        self.assertEqual(llm_cfg.max_position_embeddings, 2_097_152)
        self.assertEqual(llm_cfg.rope_theta, 2_000_000.0)

    def test_autonomous_decision_engine(self):
        """Verify situation analysis, candidate strategy generation, and autonomous choice."""
        # 1. Math goal
        math_decision = AutonomousDecisionEngine.decide("calculate 17 multiplied by 23 and verify")
        self.assertIsInstance(math_decision, AutonomousDecision)
        self.assertTrue(len(math_decision.evaluated_alternatives) >= 1)
        self.assertTrue(math_decision.confidence_score > 0.7)
        self.assertTrue(len(math_decision.volition_rationale) > 10)

        # 2. Research goal
        search_decision = AutonomousDecisionEngine.decide("search for latest discoveries in quantum computing")
        self.assertIsInstance(search_decision, AutonomousDecision)
        self.assertIn("Exploration", search_decision.chosen_strategy.name)

    def test_living_ai_agent_execution(self):
        """Verify LivingAIAgent mission execution, human reflection, and synthesis."""
        agent = LivingAIAgent(ToolRegistry(), max_steps=4, verbose=False)
        result = agent.execute_living_mission("what should we do to optimize transformer inference?")

        self.assertIsNotNone(result.decision)
        self.assertTrue(len(result.reflections) > 0)
        self.assertTrue(len(result.human_synthesis) > 50)
        self.assertIn("Recommendation", result.human_synthesis)

    def test_affective_engine_detection(self):
        """Verify user emotion detection across frustration, joy, fatigue, and bonding."""
        # Frustration
        frust = AffectiveEngine.detect_user_affect("why is this not working, it's so frustrating and broken!")
        self.assertEqual(frust.primary_emotion, "frustration")
        self.assertLess(frust.sentiment_score, 0)

        # Joy
        joy = AffectiveEngine.detect_user_affect("it works! That is amazing, brilliant job!")
        self.assertEqual(joy.primary_emotion, "joy")
        self.assertGreater(joy.sentiment_score, 0)

        # Fatigue
        fatigue = AffectiveEngine.detect_user_affect("I am so tired and my brain is fried from this late night session")
        self.assertEqual(fatigue.primary_emotion, "fatigue")

        # Bonding
        bonding = AffectiveEngine.detect_user_affect("do you have feelings and do you understand me, Yuvraj?")
        self.assertEqual(bonding.primary_emotion, "bonding")

    def test_inner_affective_state_evolution(self):
        """Verify agent's internal mood evolves sympathetically."""
        state = InnerAffectiveState()
        user_frust = UserEmotionProfile("frustration", 0.9, "comfort", -0.7)
        state.evolve(user_frust)
        self.assertIn("empathetic", state.mood.lower())

        user_joy = UserEmotionProfile("joy", 0.9, "celebration", 0.9)
        state.evolve(user_joy)
        self.assertIn("thrilled", state.mood.lower())

    def test_humanize_text(self):
        """Verify removal of robotic AI boilerplate."""
        robotic = "As an AI language model, I do not possess personal feelings. Certainly! I would be happy to help you with that."
        humanized = AffectiveEngine.humanize_text(robotic)
        self.assertNotIn("As an AI", humanized)
        self.assertNotIn("language model", humanized)
        self.assertNotIn("I do not possess", humanized)

    def test_orchestrator_living_agent_and_emotional_resonance(self):
        """Verify orchestrator routes living-agent and emotional-resonance intents."""
        orch = ReasoningOrchestrator(enable_thinking=True)

        # Emotional resonance
        prep_em = orch.prepare_reasoning("do you have feelings and emotions?")
        self.assertEqual(prep_em.intent, "emotional-resonance")
        self.assertIsNotNone(prep_em.direct_answer)
        self.assertIn("Curiosity", prep_em.direct_answer)
        self.assertIn("Empathy", prep_em.direct_answer)
        self.assertIn("Yuvraj Thakur", prep_em.direct_answer)

        # Living agent
        prep_la = orch.prepare_reasoning("act as a living agent to decide the best path forward")
        self.assertEqual(prep_la.intent, "living-agent")
        self.assertIsNotNone(prep_la.direct_answer)
        self.assertTrue(len(prep_la.direct_answer) > 50)


if __name__ == "__main__":
    unittest.main()

