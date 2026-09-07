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
from state.customer import Customer
from tests.utils.client_responses import make_react_client_response
from tool_registry import build_llm_tools, tool_registry

tools = build_llm_tools(
    tool_registry=tool_registry,
    config=config,
)


class TestReActExecutor(TestCase):
    def setUp(self):
        super().setUp()
        self.mock_react_llm = Mock(spec=ReActLLM)
        self.mock_react_llm.return_value = make_react_client_response(
            content="Done",
            tool_calls=[],
        )
        self.react_executor = ReActExecutor(
            llm_call=self.mock_react_llm,
            config=config,
        )

    def test_final_assistant_response(self):
        self.mock_react_llm.return_value = make_react_client_response(
            content="Customer found",
            tool_calls=[],
        )

        state = AgentState()

        self.react_executor.execute(objective="Find customer", state=state)

        self.assertEqual(state.messages[-1]["role"], "assistant")

        self.assertEqual(state.messages[-1]["content"], "Customer found")

    def test_starts_new_conversation_for_each_execution(self):
        self.mock_react_llm.side_effect = [
            make_react_client_response(content="Customer found", tool_calls=[]),
            make_react_client_response(content="Orders found", tool_calls=[]),
        ]

        state = AgentState()

        self.react_executor.execute(objective="Find customer", state=state)
        self.react_executor.execute(objective="Find orders", state=state)

        self.assertEqual(
            state.messages[0]["content"],
            build_agent_system_prompt(config=config),
        )
        self.assertEqual(state.messages[1]["role"], "system")
        self.assertEqual(state.messages[2]["content"], "Find orders")
        self.assertEqual(state.messages[3]["content"], "Orders found")

    def test_resets_iteration_for_each_execution(self):
        self.mock_react_llm.return_value = make_react_client_response(
            content="Customer found",
            tool_calls=[],
        )

        state = AgentState()

        state.iteration = 2

        self.react_executor.execute(objective="Find customer", state=state)

        self.assertEqual(state.iteration, 0)

    def test_preserves_semantic_state_across_executions(self):
        self.mock_react_llm.return_value = make_react_client_response(
            content="Customer found",
            tool_calls=[],
        )

        state = AgentState(retrieved_count=3)

        self.react_executor.execute(
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

        self.mock_react_llm.side_effect = [
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

        self.react_executor.execute(
            objective="Find customer",
            state=state,
        )

        self.assertIsNotNone(state.retrieved_customer)
        self.assertEqual(state.retrieved_customer.id, 1)

        self.react_executor.execute(
            objective="Create customer summary",
            state=state,
        )

        self.assertIsNotNone(state.retrieved_customer)
        self.assertEqual(state.retrieved_customer.id, 1)

    def test_includes_retrieved_customer_in_prompt(self):
        state = AgentState(
            retrieved_customer=Customer(
                id=1,
                name="Alice",
                email="alice@example.com",
                plan="pro",
            )
        )

        self.react_executor.execute(
            objective="Create customer summary",
            state=state,
        )

        messages = self.mock_react_llm.call_args.kwargs["messages"]

        self.assertIn("Alice", messages[1]["content"])

    def test_recovers_after_tool_error(self):
        first_response = make_react_client_response(
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function={
                        "name": "calculator",
                        "arguments": '{"a": 10, "b": 0, "operation": "divide"}',
                    },
                )
            ],
        )

        second_response = make_react_client_response(
            content="The answer is 5.",
            tool_calls=[],
        )

        self.mock_react_llm.side_effect = [
            first_response,
            second_response,
        ]

        result = self.react_executor.execute(
            objective="Calculate 10 divided by 0, then give me the answer.",
            state=AgentState(),
        )

        self.assertEqual(result.response, "The answer is 5.")
        self.assertEqual(self.mock_react_llm.call_count, 2)

        second_messages = self.mock_react_llm.call_args_list[1].kwargs["messages"]

        tool_messages = [
            message for message in second_messages if message["role"] == "tool"
        ]

        self.assertEqual(len(tool_messages), 1)

        self.assertIn("tool_execution_error", tool_messages[0]["content"])
