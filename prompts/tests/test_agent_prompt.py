# prompts/tests/test_agent_prompt.py
from unittest import TestCase

from prompts.agent_prompt import build_agent_system_prompt


class TestBuildAgentSystemPrompt(TestCase):

    def test_includes_max_retrieved_results(self):
        config = {
            "max_retrieved_results": 100,
        }

        prompt = build_agent_system_prompt(config)

        self.assertIn(
            "Maximum total customers that may be retrieved: 100",
            prompt,
        )
