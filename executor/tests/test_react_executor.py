# executor/tests/test_react_executor.py
from unittest import TestCase
from unittest.mock import Mock

from executor.react_executor import ReActExecutor
from llm.react_llm import ReActLLM
from state.agent_state import AgentState
from config.settings import config
from tests.utils.client_responses import make_react_client_response
from tool_registry import build_llm_tools, tool_registry

tools = build_llm_tools(
    tool_registry=tool_registry,
    config=config,
)


class TestReActExecutor(TestCase):
    def test_final_assistant_response(self):
        mock_react_client = Mock()
        mock_react_client.chat.completions.create.return_value = (
            make_react_client_response(
                content="Customer found",
                tool_calls=[],
            )
        )

        state = AgentState()

        react_executor = ReActExecutor(
            llm_call=ReActLLM(
                client=mock_react_client,
                model="test-model",
                tools=tools,
            ),
            config=config,
        )

        react_executor.execute(
            objective="Find customer",
            state=state,
        )

        self.assertEqual(
            state.messages[-1]["role"],
            "assistant",
        )

        self.assertEqual(
            state.messages[-1]["content"],
            "Customer found",
        )
