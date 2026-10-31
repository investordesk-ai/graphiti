"""Response recovery validates corrections and preserves transport and cancellation boundaries."""

import asyncio
import json

from unittest.mock import AsyncMock, Mock

import httpx
import openai
import pytest

from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.errors import LLMResponseError, RateLimitError, RefusalError
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.prompts.models import Message

# a complete result must match its contract before reaching the operation that requested it
class Result(BaseModel):
    records: list[int]

# exercise the real public recovery method with isolated provider responses
@pytest.mark.parametrize("invalid", [{}, {"records": None}, {"records": "wrong"}, None, [], 7])
def test_invalid_output_gets_a_distinct_correction_prompt(invalid):

    async def run():
        client = OpenAIClient(config = LLMConfig(model = "9B"), client = object())
        original = [Message(role = "system", content = "Extract source records"),
                    Message(role = "user", content = "source record 42")]
        snapshots = []

        async def generate(messages, *args):
            snapshots.append([message.model_copy(deep = True) for message in messages])
            return (invalid if len(snapshots) == 1 else {"records": [42]}, 11, 7)

        client._generate_response = generate
        result = await client.generate_response(original, response_model = Result, prompt_name = "extract")
        assert result == {"records": [42]}
        assert len(snapshots) == 2
        assert snapshots[0][1].content == snapshots[1][1].content == "source record 42"
        assert "CORRECTION REQUEST" in snapshots[1][-1].content
        diagnostic = json.loads(snapshots[1][-1].content.split("\n", 1)[1])
        assert diagnostic["diagnostics"]["response_excerpt"]
        if isinstance(invalid, dict):
            assert diagnostic["diagnostics"]["validation_errors"]
        assert diagnostic["response_schema"]["required"] == ["records"]
        assert len(original) == 2 and original[0].content == "Extract source records"
        usage = client.token_tracker.get_total_usage()
        assert usage.input_tokens == 22 and usage.output_tokens == 14

    asyncio.run(run())

# a bounded correction cycle must expose its evidence rather than returning an invalid success
def test_exhausted_correction_preserves_evidence_and_attempts():

    async def run():
        client = OpenAIClient(config = LLMConfig(model = "9B"), client = object())
        client._generate_response = AsyncMock(return_value = ({"records": "wrong"}, 13, 5))
        with pytest.raises(LLMResponseError) as caught:
            await client.generate_response([Message(role = "system", content = "source")],
                                           response_model = Result, prompt_name = "extract")
        error = caught.value
        assert error.recovery_action == "reduce"
        assert error.diagnostics["attempts"] == 3
        assert error.diagnostics["input_tokens"] == 39
        assert error.diagnostics["output_tokens"] == 15
        assert "wrong" in error.diagnostics["response_excerpt"]
        assert client._generate_response.await_count == 3

    asyncio.run(run())

# structured response adapters can require decomposition or operator action without an unchanged retry
@pytest.mark.parametrize("action", ["reduce", "retry", "review"])
def test_explicit_recovery_action_is_not_retried_locally(action):

    async def run():
        client = OpenAIClient(config = LLMConfig(model = "9B"), client = object())
        failure = LLMResponseError("Response rejected", recovery_action = action,
                                   diagnostics = {"finish_reason": "length"})
        client._generate_response = AsyncMock(side_effect = failure)
        with pytest.raises(LLMResponseError) as caught:
            await client.generate_response([Message(role = "system", content = "source")])
        assert caught.value is failure
        assert client._generate_response.await_count == 1

    asyncio.run(run())

# endpoint outages retain their retry classification and never get a semantic correction prompt
@pytest.mark.parametrize("error_type", [openai.APIConnectionError, openai.APITimeoutError])
def test_transport_error_returns_to_owning_stage(error_type):

    async def run():
        client = OpenAIClient(config = LLMConfig(model = "9B"), client = object())
        failure = error_type(request = httpx.Request("POST", "http://model.invalid"))
        client._generate_response = AsyncMock(side_effect = failure)
        with pytest.raises(LLMResponseError) as caught:
            await client.generate_response([Message(role = "system", content = "source")])
        assert caught.value.recovery_action == "retry"
        assert caught.value.__cause__ is failure
        assert client._generate_response.await_count == 1

    asyncio.run(run())

# rejected API requests cannot be corrected by repeatedly resubmitting the same configuration
@pytest.mark.parametrize("error_type,status", [
    (openai.AuthenticationError, 401), (openai.PermissionDeniedError, 403), (openai.BadRequestError, 400),
])
def test_invalid_api_request_requires_review(error_type, status):

    async def run():
        client = OpenAIClient(config = LLMConfig(model = "9B"), client = object())
        response = httpx.Response(status, request = httpx.Request("POST", "http://model.invalid"))
        failure = error_type("Rejected", response = response, body = None)
        client._generate_response = AsyncMock(side_effect = failure)
        with pytest.raises(LLMResponseError) as caught:
            await client.generate_response([Message(role = "system", content = "source")])
        assert caught.value.recovery_action == "review"
        assert caught.value.__cause__ is failure
        assert client._generate_response.await_count == 1

    asyncio.run(run())

# the full correction cycle shares one deadline and cancels a provider that has not returned
def test_recovery_deadline_cancels_inflight_request():

    async def run():
        client = OpenAIClient(config = LLMConfig(model = "9B", recovery_timeout_seconds = 0.02),
                              client = object())
        cancelled = asyncio.Event()

        async def generate(*args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        client._generate_response = AsyncMock(side_effect = generate)
        with pytest.raises(LLMResponseError) as caught:
            await client.generate_response([Message(role = "system", content = "source")])
        assert caught.value.recovery_action == "reduce"
        assert client._generate_response.await_count == 1
        assert cancelled.is_set()

    asyncio.run(run())

# caller cancellation is never transformed into a correction or a successful response
def test_cancellation_is_preserved():

    async def run():
        client = OpenAIClient(config = LLMConfig(model = "9B"), client = object())
        client._generate_response = AsyncMock(side_effect = asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await client.generate_response([Message(role = "system", content = "source")])
        assert client._generate_response.await_count == 1

    asyncio.run(run())

# provider refusal and rate limits retain their existing public contracts
@pytest.mark.parametrize("error", [RefusalError("Refused"), RateLimitError()])
def test_refusal_and_rate_limit_do_not_become_corrections(error):

    async def run():
        client = OpenAIClient(config = LLMConfig(model = "9B"), client = object())
        client._generate_response = AsyncMock(side_effect = error)
        with pytest.raises(type(error)):
            await client.generate_response([Message(role = "system", content = "source")])
        assert client._generate_response.await_count == 1

    asyncio.run(run())

# a malformed deadline must be rejected before a request can become unbounded
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, None, "600"])
def test_recovery_deadline_is_validated(value):
    with pytest.raises(ValueError):
        LLMConfig(recovery_timeout_seconds = value)

# valid JSON syntax is insufficient when a schema-free response has the wrong envelope or non-finite values
@pytest.mark.parametrize("invalid", [[], None, 17, {"value": float("nan")}, {"value": float("inf")}])
def test_schema_free_output_requires_a_finite_object(invalid):

    async def run():
        client = OpenAIClient(config = LLMConfig(model = "9B"), client = object())
        client._generate_response = AsyncMock(side_effect = [(invalid, 1, 1), ({"value": 42}, 1, 1)])
        result = await client.generate_response([Message(role = "system", content = "source")])
        assert result == {"value": 42}
        assert client._generate_response.await_count == 2
        assert "CORRECTION REQUEST" in client._generate_response.await_args.args[0][-1].content

    asyncio.run(run())

# SDK length rejection must request decomposition instead of spending the same output budget three times
def test_sdk_length_rejection_reduces_without_local_retry():

    async def run():
        client = OpenAIClient(config = LLMConfig(model = "9B"), client = object())
        completion = openai.types.chat.ChatCompletion(
            id = "rejected", object = "chat.completion", created = 0, model = "9B",
            choices = [{"index": 0, "finish_reason": "length",
                        "message": {"role": "assistant", "content": "{truncated"}}],
        ) # ChatCompletion
        client._create_structured_completion = AsyncMock(
            side_effect = openai.LengthFinishReasonError(completion = completion),
        ) # AsyncMock
        with pytest.raises(LLMResponseError) as caught:
            await client.generate_response([Message(role = "system", content = "source")],
                                           response_model = Result, max_tokens = 4096)
        assert caught.value.recovery_action == "reduce"
        assert caught.value.diagnostics["finish_reason"] == "length"
        assert caught.value.diagnostics["max_tokens"] == 4096
        assert client._create_structured_completion.await_count == 1

    asyncio.run(run())

# accounting defects cannot transform a valid answer into another inference request
@pytest.mark.parametrize("input_usage,output_usage", [
    (None, 4), ("12", 4), (-1, 4), (True, 4), (12, float("nan")), (12, -1),
])
def test_invalid_usage_does_not_retry_valid_output(input_usage, output_usage):

    async def run():
        client = OpenAIClient(config = LLMConfig(model = "9B"), client = object())
        client._generate_response = AsyncMock(return_value = ({"records": [42]}, input_usage, output_usage))
        result = await client.generate_response([Message(role = "system", content = "source")],
                                               response_model = Result)
        assert result == {"records": [42]}
        assert client._generate_response.await_count == 1
        usage = client.token_tracker.get_total_usage()
        assert usage.input_tokens == (input_usage if type(input_usage) is int and input_usage >= 0 else 0)
        assert usage.output_tokens == (output_usage if type(output_usage) is int and output_usage >= 0 else 0)

    asyncio.run(run())

def test_token_tracker_failure_does_not_retry_valid_output():

    async def run():
        client = OpenAIClient(config = LLMConfig(model = "9B"), client = object())
        client._generate_response = AsyncMock(return_value = ({"records": [42]}, 12, 4))
        client.token_tracker.record = Mock(side_effect = RuntimeError("Accounting unavailable"))
        result = await client.generate_response([Message(role = "system", content = "source")],
                                               response_model = Result)
        assert result == {"records": [42]}
        assert client._generate_response.await_count == 1

    asyncio.run(run())
