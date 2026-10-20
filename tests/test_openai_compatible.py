"""Regression tests for arbitrary OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from chessbench import constants
from chessbench.benchmark.runner import BenchmarkConfig, BenchmarkRunner
from chessbench.cli import _load_api_keys
from chessbench.common.common_types import ChatMessage, ModelInfo
from chessbench.providers.chess_ai import ProviderChessAI
from chessbench.providers.openai_compatible import (
    OpenAICompatibleProvider,
    normalize_base_url,
    validate_endpoint_url,
)
from chessbench.providers.registry import get_provider, list_providers
from chessbench.ui.helpers import player_banner_html, render_move_ticker_html


@pytest.fixture
def compatible_server():
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def _json_response(self, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            requests.append({"method": "GET", "path": self.path})
            self._json_response(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "local/chess-model",
                            "object": "model",
                            "created": 0,
                            "owned_by": "test",
                        }
                    ],
                }
            )

        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(content_length))
            requests.append(
                {
                    "method": "POST",
                    "path": self.path,
                    "body": body,
                    "authorization": self.headers.get("Authorization"),
                }
            )
            self._json_response(
                {
                    "id": "completion-1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "e2e4"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 3,
                        "total_tokens": 15,
                    },
                }
            )

        def log_message(self, format: str, *args: object) -> None:
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("sandbox does not permit binding a loopback test socket")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/v1", requests
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_provider_is_registered_and_accepts_constructor_config():
    assert "openai_compatible" in list_providers()

    provider = get_provider(
        "openai_compatible",
        base_url="https://llm.example.test/v1/",
        models=["vendor/model-a"],
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == "https://llm.example.test/v1"
    assert provider.configured_models == ("vendor/model-a",)


def test_named_provider_instance_uses_compatible_adapter():
    provider = get_provider(
        "my_local_server",
        adapter="openai_compatible",
        base_url="http://127.0.0.1:8000/v1",
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.name == "my_local_server"


def test_api_key_is_optional_for_local_compatible_servers():
    provider = OpenAICompatibleProvider(base_url="http://127.0.0.1:8000/v1")

    assert provider.requires_api_key is False
    assert provider.validate_key("") is True
    assert provider.validate_key("arbitrary-provider-key") is True


@pytest.mark.parametrize(
    "url",
    [
        "",
        "javascript://attacker.example/v1",
        "https://user:password@example.com/v1",
        "https://example.com/v1?target=http://127.0.0.1",
        "https://example.com/v1#@127.0.0.1",
        "https://example.com/v1/../admin",
        "https://example.com/v1/%2e%2e/admin",
        "https://example.com/v1\\models",
        "https://example.com/v1\r\nHost: 127.0.0.1",
        "https://example.com/v1/models",
        "https://example.com/v1/chat/completions",
        "https://example.com:99999/v1",
    ],
)
def test_malicious_or_ambiguous_endpoint_shapes_are_rejected(url):
    with pytest.raises(
        ValueError,
        match=r"base_url|URL|port|query|path|host|user info|required",
    ):
        validate_endpoint_url(url)


def test_endpoint_url_is_canonicalized_without_changing_api_path():
    assert (
        normalize_base_url("HTTPS://Example.COM.:443/custom/v1/")
        == "https://example.com:443/custom/v1"
    )


def test_public_endpoint_validation_rejects_private_dns_resolution():
    private_resolution = [
        (2, 1, 6, "", ("127.0.0.1", 443)),
    ]
    with (
        patch(
            "chessbench.providers.openai_compatible.socket.getaddrinfo",
            return_value=private_resolution,
        ),
        pytest.raises(ValueError, match="Private/local endpoints"),
    ):
        validate_endpoint_url(
            "https://attacker.example/v1",
            enforce_public_networks=True,
        )


def test_provider_revalidates_dns_before_each_request():
    public_resolution = [(2, 1, 6, "", ("93.184.216.34", 443))]
    private_resolution = [(2, 1, 6, "", ("169.254.169.254", 443))]
    with (
        patch(
            "chessbench.providers.openai_compatible.socket.getaddrinfo",
            side_effect=[public_resolution, private_resolution],
        ),
        patch("chessbench.providers.openai_compatible.AsyncOpenAI") as client,
    ):
        provider = OpenAICompatibleProvider(
            base_url="https://attacker.example/v1",
            enforce_public_networks=True,
        )
        with pytest.raises(ValueError, match="Private/local endpoints"):
            provider._client("secret")

    client.assert_not_called()


def test_endpoint_host_allowlist_is_exact():
    with pytest.raises(ValueError, match="ALLOWED_HOSTS"):
        validate_endpoint_url(
            "https://evil.example/v1",
            allowed_hosts=["api.example"],
        )


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "internal.example"},
        {"Content-Length": "0"},
        {"X-Test\r\nInjected": "value"},
        {"X-Test": "value\r\nInjected: yes"},
    ],
)
def test_unsafe_custom_headers_are_rejected(headers):
    with pytest.raises(ValueError, match="Unsafe custom HTTP header"):
        OpenAICompatibleProvider(
            base_url="https://example.com/v1",
            default_headers=headers,
        )


def test_api_key_header_injection_is_rejected():
    provider = OpenAICompatibleProvider(base_url="https://example.com/v1")

    with pytest.raises(ValueError, match="API key contains invalid"):
        provider._client("secret\r\nX-Injected: yes")


def test_provider_and_move_html_escapes_untrusted_values():
    banner = player_banner_html(
        name='<img src=x onerror="alert(1)">',
        spec="</div><script>alert(1)</script>",
        color="white",
    )
    move = MagicMock(
        is_illegal=True,
        move='<svg onload="alert(1)">',
        move_san=None,
        captured_piece=None,
    )
    ticker = render_move_ticker_html([move])

    assert "<script>" not in banner
    assert "<img" not in banner
    assert "<svg" not in ticker
    assert "&lt;script&gt;" in banner
    assert "&lt;svg" in ticker


@pytest.mark.asyncio
async def test_list_models_keeps_arbitrary_model_ids():
    provider = OpenAICompatibleProvider(
        base_url="https://llm.example.test/v1/",
        context_window=65_536,
        default_headers={"X-Custom": "value"},
        default_query={"api-version": "2026-01-01"},
    )

    with (
        patch("chessbench.providers.openai_compatible.AsyncOpenAI") as client_class,
        patch("chessbench.providers.openai_compatible.httpx.AsyncClient") as http_client_class,
    ):
        client = AsyncMock()
        http_client = MagicMock()
        http_client_class.return_value = http_client
        client.models.list = AsyncMock(
            return_value=MagicMock(
                data=[
                    MagicMock(id="acme/chess-reasoner"),
                    MagicMock(id="vendor/model-with-no-known-prefix"),
                ]
            )
        )
        client_class.return_value = client

        models = await provider.list_models("")

    assert [model.id for model in models] == [
        "acme/chess-reasoner",
        "vendor/model-with-no-known-prefix",
    ]
    assert all(model.context_window == 65_536 for model in models)
    client_class.assert_called_once_with(
        api_key="not-needed",
        base_url="https://llm.example.test/v1",
        timeout=10.0,
        default_headers={"X-Custom": "value"},
        default_query={"api-version": "2026-01-01"},
        max_retries=0,
        http_client=http_client,
    )
    http_client_class.assert_called_once_with(
        follow_redirects=False,
        trust_env=False,
    )


@pytest.mark.asyncio
async def test_configured_models_fallback_when_models_endpoint_is_missing():
    provider = OpenAICompatibleProvider(
        base_url="https://llm.example.test/v1",
        models="model-a, model-b",
    )

    with patch("chessbench.providers.openai_compatible.AsyncOpenAI") as client_class:
        client = AsyncMock()
        client.models.list = AsyncMock(side_effect=RuntimeError("404 /models"))
        client_class.return_value = client

        models = await provider.list_models("endpoint-key")

    assert [model.id for model in models] == ["model-a", "model-b"]


@pytest.mark.asyncio
async def test_completion_uses_compatible_chat_contract_and_optional_parameters():
    provider = OpenAICompatibleProvider(
        base_url="https://llm.example.test/v1",
        supports_tools=False,
        send_temperature=False,
        max_tokens_field="max_completion_tokens",
        extra_body={"provider_option": True},
    )

    with patch("chessbench.providers.openai_compatible.AsyncOpenAI") as client_class:
        client = AsyncMock()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="e2e4", tool_calls=None))]
        response.usage = MagicMock(prompt_tokens=12, completion_tokens=3)
        response.model_dump.return_value = {"id": "completion-1"}
        client.chat.completions.create = AsyncMock(return_value=response)
        client_class.return_value = client

        result = await provider.complete(
            "endpoint-key",
            "vendor/model-a",
            [ChatMessage(role="user", content="Choose a move")],
            temperature=0.2,
            max_tokens=128,
            tools=[{"type": "function"}],
            tool_choice="required",
        )

    assert result.text == "e2e4"
    assert result.tokens_in == 12
    assert result.tokens_out == 3
    request = client.chat.completions.create.await_args.kwargs
    assert request == {
        "model": "vendor/model-a",
        "messages": [{"role": "user", "content": "Choose a move"}],
        "max_completion_tokens": 128,
        "extra_body": {"provider_option": True},
    }


@pytest.mark.asyncio
async def test_tool_parameters_are_forwarded_only_when_enabled():
    provider = OpenAICompatibleProvider(
        base_url="https://llm.example.test/v1",
        supports_tools=True,
    )

    with patch("chessbench.providers.openai_compatible.AsyncOpenAI") as client_class:
        client = AsyncMock()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="", tool_calls=None))]
        response.usage = None
        response.model_dump.return_value = {}
        client.chat.completions.create = AsyncMock(return_value=response)
        client_class.return_value = client

        await provider.complete(
            "",
            "model-a",
            [ChatMessage(role="user", content="move")],
            tools=[{"type": "function"}],
            tool_choice="required",
        )

    request = client.chat.completions.create.await_args.kwargs
    assert request["tools"] == [{"type": "function"}]
    assert request["tool_choice"] == "required"


def test_provider_config_is_not_forwarded_as_completion_parameter():
    provider = MagicMock()
    with patch(
        "chessbench.providers.chess_ai.get_provider", return_value=provider
    ) as get_provider_mock:
        ai = ProviderChessAI(
            provider_name="openai_compatible",
            model_id="model-a",
            api_key="",
            provider_config={"base_url": "http://127.0.0.1:8000/v1"},
            temperature=0.0,
        )

    get_provider_mock.assert_called_once_with(
        "openai_compatible", base_url="http://127.0.0.1:8000/v1"
    )
    assert ai.params == {"temperature": 0.0}


def test_benchmark_config_round_trips_endpoint_configuration(tmp_path):
    path = tmp_path / "benchmark.yaml"
    config = BenchmarkConfig(
        players=["openai_compatible:model-a", "openai_compatible:model-b"],
        provider_configs={
            "openai_compatible": {
                "base_url": "https://llm.example.test/v1",
                "models": ["model-a", "model-b"],
            }
        },
    )

    config.save_yaml(str(path))
    loaded = BenchmarkConfig.from_yaml(str(path))

    assert loaded.provider_configs == config.provider_configs


def test_benchmark_log_config_redacts_endpoint_credentials():
    config = BenchmarkConfig(
        api_keys={"my_endpoint": "secret-key"},
        provider_configs={
            "my_endpoint": {
                "adapter": "openai_compatible",
                "base_url": "https://llm.example.test/v1",
                "default_headers": {"X-API-Key": "header-secret"},
            }
        },
    )

    logged = config.to_dict(redact_secrets=True)

    assert logged["api_keys"] == {"my_endpoint": "<redacted>"}
    assert logged["provider_configs"]["my_endpoint"]["default_headers"] == {
        "X-API-Key": "<redacted>"
    }
    assert config.api_keys["my_endpoint"] == "secret-key"


def test_named_compatible_provider_uses_generic_environment_key(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "endpoint-secret")

    keys = _load_api_keys({"my_endpoint": {"adapter": "openai_compatible"}})

    assert keys["my_endpoint"] == "endpoint-secret"


def test_runner_forwards_provider_config_to_every_player(tmp_path):
    config = BenchmarkConfig(
        players=["my_endpoint:model-a", "my_endpoint:model-b"],
        output_dir=str(tmp_path),
        provider_configs={
            "my_endpoint": {
                "adapter": "openai_compatible",
                "base_url": "https://llm.example.test/v1",
                "models": ["model-a", "model-b"],
            }
        },
    )

    runner = BenchmarkRunner(config)

    assert set(runner.players) == {"my_endpoint:model-a", "my_endpoint:model-b"}
    assert all(player.provider.name == "my_endpoint" for player in runner.players.values())
    assert all(
        player.provider.base_url == "https://llm.example.test/v1"
        for player in runner.players.values()
    )


@pytest.mark.asyncio
async def test_real_client_round_trip_against_compatible_http_server(
    compatible_server,
):
    base_url, requests = compatible_server
    provider = OpenAICompatibleProvider(base_url=base_url)

    models = await provider.list_models("endpoint-key")
    result = await provider.complete(
        "endpoint-key",
        models[0].id,
        [ChatMessage(role="user", content="Choose a legal move")],
        temperature=0.0,
        max_tokens=32,
    )

    assert [model.id for model in models] == ["local/chess-model"]
    assert result.text == "e2e4"
    assert requests == [
        {"method": "GET", "path": "/v1/models"},
        {
            "method": "POST",
            "path": "/v1/chat/completions",
            "body": {
                "messages": [{"role": "user", "content": "Choose a legal move"}],
                "model": "local/chess-model",
                "max_tokens": 32,
                "temperature": 0.0,
            },
            "authorization": "Bearer endpoint-key",
        },
    ]


def test_streamlit_add_provider_then_probe_and_select_models(monkeypatch):
    from chessbench.ui import streamlit_app

    monkeypatch.setattr(constants, "HOSTED_PROVIDERS", ("openai_compatible",))
    monkeypatch.setattr(streamlit_app, "HOSTED_PROVIDERS", ("openai_compatible",))
    monkeypatch.setenv("CHESSBENCH_ALLOW_PRIVATE_ENDPOINTS", "true")
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
    models = [
        ModelInfo(
            id="model-a",
            name="Model A",
            provider="custom_local_test",
            context_window=8192,
        ),
        ModelInfo(
            id="model-b",
            name="Model B",
            provider="custom_local_test",
            context_window=8192,
        ),
    ]
    probe = AsyncMock(return_value=models)

    with patch(
        "chessbench.providers.openai_compatible.OpenAICompatibleProvider.list_models",
        new=probe,
    ):
        app = AppTest.from_file("streamlit_app.py", default_timeout=20).run()
        inputs = {item.label: item for item in app.text_input}
        inputs["Provider name"].set_value("Local Test")
        inputs["Base URL"].set_value("http://127.0.0.1:8000/v1")
        next(button for button in app.button if button.label == "Add provider").click()
        app.run()

    assert not app.exception
    assert list(app.session_state["custom_openai_providers"]) == ["custom_local_test"]
    selectboxes = {item.label: item for item in app.selectbox}
    assert selectboxes["Player 1 provider"].value == "custom_local_test"
    assert selectboxes["Player 2 provider"].value == "custom_local_test"
    assert list(selectboxes["Player 1 model"].options) == ["Model A", "Model B"]
    assert list(selectboxes["Player 2 model"].options) == ["Model A", "Model B"]
    assert probe.await_count == 2

    with patch(
        "chessbench.providers.openai_compatible.OpenAICompatibleProvider.list_models",
        new=probe,
    ):
        inputs = {item.label: item for item in app.text_input}
        inputs["Provider name"].set_value("Second Endpoint")
        inputs["Base URL"].set_value("http://127.0.0.1:9000/v1")
        next(button for button in app.button if button.label == "Add provider").click()
        app.run()
        probes_before_selection = probe.await_count
        selectboxes = {item.label: item for item in app.selectbox}
        selectboxes["Player 1 provider"].set_value("custom_second_endpoint")
        app.run()

    assert probe.await_count == probes_before_selection + 1
    assert app.session_state["player_provider_1"] == "custom_second_endpoint"


def test_hosted_ui_blocks_private_endpoint_unless_explicitly_enabled(monkeypatch):
    from chessbench.ui import streamlit_app

    monkeypatch.setattr(streamlit_app, "HOSTED_PROVIDERS", ("openai_compatible",))
    monkeypatch.delenv("CHESSBENCH_ALLOW_PRIVATE_ENDPOINTS", raising=False)

    with pytest.raises(ValueError, match="Private/local endpoints"):
        streamlit_app._validate_user_endpoint("https://127.0.0.1:8000/v1")
    with pytest.raises(ValueError, match="requires HTTPS"):
        streamlit_app._validate_user_endpoint("http://8.8.8.8/v1")

    monkeypatch.setenv("CHESSBENCH_ALLOW_PRIVATE_ENDPOINTS", "true")
    assert (
        streamlit_app._validate_user_endpoint("http://127.0.0.1:8000/v1/")
        == "http://127.0.0.1:8000/v1"
    )


def test_ui_endpoint_policy_is_public_by_default(monkeypatch):
    from chessbench.ui import streamlit_app

    monkeypatch.setattr(streamlit_app, "HOSTED_PROVIDERS", None)
    monkeypatch.delenv("CHESSBENCH_PUBLIC_DEPLOYMENT", raising=False)
    monkeypatch.delenv("CHESSBENCH_ALLOW_PRIVATE_ENDPOINTS", raising=False)

    with pytest.raises(ValueError, match="Private/local endpoints"):
        streamlit_app._validate_user_endpoint("https://127.0.0.1:8000/v1")
