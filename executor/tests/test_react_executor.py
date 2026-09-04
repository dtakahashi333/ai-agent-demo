# executor/tests/test_react_executor.py
from unittest import TestCase
from unittest.mock import Mock

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from executor.react_executor import ReActExecutor
from llm.react_llm import ReActLLM
from prompts.agent_prompt import build_agent_system_prompt
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

    def test_starts_new_conversation_for_each_execution(self):
        state = AgentState()

        mock_react_client = Mock()
        mock_react_client.chat.completions.create.side_effect = [
            make_react_client_response(content="Customer found", tool_calls=[]),
            make_react_client_response(content="Orders found", tool_calls=[]),
        ]

        react_executor = ReActExecutor(
            llm_call=ReActLLM(
                client=mock_react_client,
                model="test-model",
                tools=tools,
            ),
            config=config,
        )

        react_executor.execute(objective="Find customer", state=state)
        react_executor.execute(objective="Find orders", state=state)

        self.assertEqual(
            state.messages,
            [
                {"role": "system", "content": build_agent_system_prompt(config=config)},
                {"role": "user", "content": "Find orders"},
                {"role": "assistant", "content": "Orders found"},
            ],
        )

    def test_resets_iteration_for_each_execution(self):
        state = AgentState()

        state.iteration = 2

        mock_react_client = Mock()
        mock_react_client.chat.completions.create.return_value = (
            make_react_client_response(content="Customer found", tool_calls=[])
        )

        react_executor = ReActExecutor(
            llm_call=ReActLLM(
                client=mock_react_client,
                model="test-model",
                tools=tools,
            ),
            config=config,
        )

        react_executor.execute(objective="Find customer", state=state)

        self.assertEqual(state.iteration, 0)

    def test_preserves_semantic_state_across_executions(self):
        state = AgentState(retrieved_count=3)

        mock_react_client = Mock()
        mock_react_client.chat.completions.create.return_value = (
            make_react_client_response(
                content="Customer found",
                tool_calls=[],
            )
        )

        react_executor = ReActExecutor(
            llm_call=ReActLLM(
                client=mock_react_client,
                model="test-model",
                tools=tools,
            ),
            config=config,
        )

        react_executor.execute(
            objective="Create customer summary",
            state=state,
        )

        self.assertEqual(state.retrieved_count, 3)

    def test_preserves_selected_customer_across_executions(self):
        get_customer_tool_call = ChatCompletionMessageToolCall(
            id="call_1",
            type="function",
            function={
                "name": "get_customer",
                "arguments": '{"customer_id": 1}',
            },
        )

        mock_react_client = Mock()
        mock_react_client.chat.completions.create.side_effect = [
            make_react_client_response(
                content=None,
                tool_calls=[get_customer_tool_call],
            ),
            make_react_client_response(
                content="Customer found",
                tool_calls=[],
            ),
            make_react_client_response(
                content="Summary created",
                tool_calls=[],
            ),
        ]

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

        self.assertIsNotNone(state.retrieved_customer)
        self.assertEqual(state.retrieved_customer.id, 1)

        react_executor.execute(
            objective="Create customer summary",
            state=state,
        )

        self.assertIsNotNone(state.retrieved_customer)
        self.assertEqual(state.retrieved_customer.id, 1)
