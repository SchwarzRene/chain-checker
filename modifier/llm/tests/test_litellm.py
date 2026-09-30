from types import SimpleNamespace
from unittest.mock import patch

import pytest
from django.apps import AppConfig

from core.appconfig import ChainAppConfig
from chain_checker.modifier.llm.litellm import (
    LiteLLM,
    _fetch_available_tiers,
    _UsageCapture,
)
from chain_checker.utils.errors import CheckerError


def _fake_generation(*, usage_metadata=None, token_usage=None, model_name=""):
    has_message = usage_metadata is not None
    message = SimpleNamespace(usage_metadata=usage_metadata) if has_message else None
    llm_output = (
        {"token_usage": token_usage, "model_name": model_name} if token_usage or model_name else {}
    )
    return SimpleNamespace(generations=[[SimpleNamespace(message=message)]], llm_output=llm_output)


# --------------------------------------------------------------------------
# _UsageCapture: reading usage/model name off a langchain callback response
# --------------------------------------------------------------------------


def test_usage_capture_prefers_the_modern_usage_metadata_keys():
    cap = _UsageCapture()

    cap.on_llm_end(_fake_generation(usage_metadata={"input_tokens": 10, "output_tokens": 5}))

    assert cap.prompt_tokens == 10
    assert cap.completion_tokens == 5


def test_usage_capture_falls_back_to_legacy_token_usage_when_no_usage_metadata():
    cap = _UsageCapture()

    cap.on_llm_end(_fake_generation(token_usage={"prompt_tokens": 7, "completion_tokens": 3}))

    assert cap.prompt_tokens == 7
    assert cap.completion_tokens == 3


def test_usage_capture_leaves_total_tokens_none_when_the_provider_omits_it():
    # LiteLLM._record_usage() owns deriving a missing total - _UsageCapture
    # must not guess one of its own.
    cap = _UsageCapture()

    cap.on_llm_end(_fake_generation(usage_metadata={"input_tokens": 1, "output_tokens": 1}))

    assert cap.total_tokens is None


def test_usage_capture_reads_a_reported_total_when_present():
    cap = _UsageCapture()

    cap.on_llm_end(_fake_generation(token_usage={"total_tokens": 42}))

    assert cap.total_tokens == 42


def test_usage_capture_keeps_a_genuine_zero_total_instead_of_falling_back():
    # A fully-cached call can legitimately report 0 total tokens - that is a
    # real value, not a missing one, so the legacy field must not override it.
    cap = _UsageCapture()

    cap.on_llm_end(
        _fake_generation(
            usage_metadata={"input_tokens": 5, "output_tokens": 0, "total_tokens": 0},
            token_usage={"total_tokens": 99},
        )
    )

    assert cap.total_tokens == 0


def test_usage_capture_falls_back_when_usage_metadata_reports_a_field_as_none():
    # usage_metadata carrying the key with a None value (not simply missing
    # it) must still defer to the legacy token_usage field, not read as 0.
    cap = _UsageCapture()

    cap.on_llm_end(
        _fake_generation(
            usage_metadata={"input_tokens": None, "output_tokens": None},
            token_usage={"prompt_tokens": 42, "completion_tokens": 8},
        )
    )

    assert cap.prompt_tokens == 42
    assert cap.completion_tokens == 8


def test_usage_capture_reads_the_model_name():
    cap = _UsageCapture()

    cap.on_llm_end(_fake_generation(model_name="gpt-real-fast"))

    assert cap.model_name == "gpt-real-fast"


# --------------------------------------------------------------------------
# LiteLLM._resolve_app_config / _resolve_llm: the two failure paths a bad
# --modifier-app can hit, isolated from the real Django app registry.
# --------------------------------------------------------------------------


def test_an_unknown_app_label_fails_clearly():
    with patch(
        "chain_checker.modifier.llm.litellm.django_apps.get_app_config",
        side_effect=LookupError,
    ):
        with pytest.raises(CheckerError, match="No Django app config found"):
            LiteLLM._resolve_app_config("not-a-real-app")


def test_an_unknown_app_label_lists_the_labels_that_are_actually_registered():
    # A typo like 'tonalit' (missing the final -y) should surface what --
    # modifier-app could have meant, not just that the given one is wrong.
    chain_app = object.__new__(ChainAppConfig)
    chain_app.label = "tonality"
    plain_app = object.__new__(AppConfig)
    plain_app.label = "not_a_chain_app"

    with (
        patch(
            "chain_checker.modifier.llm.litellm.django_apps.get_app_config",
            side_effect=LookupError,
        ),
        patch(
            "chain_checker.modifier.llm.litellm.django_apps.get_app_configs",
            return_value=[chain_app, plain_app],
        ),
    ):
        with pytest.raises(CheckerError, match="Registered app labels: tonality") as excinfo:
            LiteLLM._resolve_app_config("tonalit")

    # Only the apps that could ever work as --modifier-app - not_a_chain_app has
    # no llm() either, so listing it would just repeat the same mistake.
    assert "not_a_chain_app" not in str(excinfo.value)


def test_a_plain_django_app_is_rejected_for_having_no_llm_of_its_own():
    # A bare instance, never through AppConfig.create(): the check under
    # test is a plain isinstance(), which doesn't need __init__ to have run,
    # and this keeps the test independent of which apps are really installed.
    plain_app = object.__new__(AppConfig)

    with (
        patch(
            "chain_checker.modifier.llm.litellm.django_apps.get_app_config",
            return_value=plain_app,
        ),
        patch(
            "chain_checker.modifier.llm.litellm.django_apps.get_app_configs",
            return_value=[],
        ),
    ):
        with pytest.raises(CheckerError, match="not a ChainAppConfig"):
            LiteLLM._resolve_app_config("some-app")


def test_a_chain_app_config_is_accepted():
    chain_app = object.__new__(ChainAppConfig)

    with patch(
        "chain_checker.modifier.llm.litellm.django_apps.get_app_config",
        return_value=chain_app,
    ):
        assert LiteLLM._resolve_app_config("some-app") is chain_app


def test_a_backend_error_building_the_llm_fails_with_the_api_key_hint():
    chain_app = SimpleNamespace(llm=lambda tier: (_ for _ in ()).throw(RuntimeError("no key")))

    with pytest.raises(CheckerError, match="LITELLM_API_KEY_MYAPP"):
        LiteLLM._resolve_llm(chain_app, "myapp", "fast")


# --------------------------------------------------------------------------
# _fetch_available_tiers: reads the proxy's own /models list, never raises -
# a broken listing reports as None, not an exception.
# --------------------------------------------------------------------------


def _fake_models_response(*, ids=None, ok=True):
    response = SimpleNamespace()
    if ok:
        response.raise_for_status = lambda: None
    else:

        def _raise():
            raise Exception("bad status")

        response.raise_for_status = _raise
    response.json = lambda: {"data": [{"id": model_id} for model_id in (ids or [])]}
    return response


def test_fetch_available_tiers_lists_the_ids_the_proxy_reports():
    with patch(
        "chain_checker.modifier.llm.litellm.httpx.get",
        return_value=_fake_models_response(ids=["fast", "balanced", "thinking"]),
    ):
        assert _fetch_available_tiers("sk-test") == ["fast", "balanced", "thinking"]


def test_fetch_available_tiers_is_none_when_the_proxy_is_unreachable():
    with patch(
        "chain_checker.modifier.llm.litellm.httpx.get",
        side_effect=ConnectionError("refused"),
    ):
        assert _fetch_available_tiers("sk-test") is None


def test_fetch_available_tiers_is_none_on_a_bad_status():
    with patch(
        "chain_checker.modifier.llm.litellm.httpx.get",
        return_value=_fake_models_response(ok=False),
    ):
        assert _fetch_available_tiers("sk-test") is None


def test_fetch_available_tiers_is_none_on_an_unexpected_response_shape():
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"unexpected": True})

    with patch("chain_checker.modifier.llm.litellm.httpx.get", return_value=response):
        assert _fetch_available_tiers("sk-test") is None


# --------------------------------------------------------------------------
# LiteLLM._check_tier_available: tells the user upfront whether --modifier-
# tier will work for --modifier-app's key, instead of only failing later.
# --------------------------------------------------------------------------


def test_an_unavailable_tier_fails_clearly_and_lists_what_is_available():
    chain_app = SimpleNamespace(litellm_api_key="sk-test")

    with patch(
        "chain_checker.modifier.llm.litellm._fetch_available_tiers",
        return_value=["fast", "balanced"],
    ):
        with pytest.raises(CheckerError, match="not available") as excinfo:
            LiteLLM._check_tier_available(chain_app, "myapp", "thinking")

    assert "balanced, fast" in str(excinfo.value)


def test_an_available_tier_passes_the_check_silently():
    chain_app = SimpleNamespace(litellm_api_key="sk-test")

    with patch(
        "chain_checker.modifier.llm.litellm._fetch_available_tiers",
        return_value=["fast", "balanced"],
    ):
        LiteLLM._check_tier_available(chain_app, "myapp", "fast")  # does not raise


def test_a_failed_listing_never_blocks_the_call_it_could_not_verify():
    # available=None means "couldn't tell" - the real .llm(tier) call below is
    # left to succeed or fail on its own, with its own clearer message.
    chain_app = SimpleNamespace(litellm_api_key="sk-test")

    with patch(
        "chain_checker.modifier.llm.litellm._fetch_available_tiers",
        return_value=None,
    ):
        LiteLLM._check_tier_available(chain_app, "myapp", "some-unlisted-tier")  # no raise


def test_an_unavailable_tier_is_caught_before_the_llm_is_even_built():
    chain_app = SimpleNamespace(
        litellm_api_key="sk-test",
        llm=lambda tier: (_ for _ in ()).throw(AssertionError("should never be called")),
    )

    with patch(
        "chain_checker.modifier.llm.litellm._fetch_available_tiers",
        return_value=["fast"],
    ):
        with pytest.raises(CheckerError, match="not available"):
            LiteLLM._resolve_llm(chain_app, "myapp", "thinking")


# --------------------------------------------------------------------------
# LiteLLM.__call__: the proxy call itself failing (e.g. unreachable) must
# fail clearly, the same way every other LiteLLM setup step here does -
# never a raw exception straight out of the run.
# --------------------------------------------------------------------------


def test_a_connection_error_calling_the_modifier_fails_clearly():
    llm = object.__new__(LiteLLM)
    llm._app_label = "myapp"
    llm._tier = "fast"
    llm._runnable = SimpleNamespace(
        invoke=lambda *a, **k: (_ for _ in ()).throw(ConnectionError("refused"))
    )

    with pytest.raises(CheckerError, match="Modifier LLM call failed") as excinfo:
        llm("some prompt")

    assert "LITELLM_API_KEY_MYAPP" in str(excinfo.value)
