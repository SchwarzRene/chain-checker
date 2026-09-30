import os

import pytest

from chain_checker.utils import bootstrap


@pytest.fixture(autouse=True)
def _reset_done_flag(monkeypatch):
    # setup_django() is a one-shot guarded by a private module-level flag;
    # reset it for every test so each one observes a fresh "first call"
    # without a real Django settings module ever having to be configured.
    monkeypatch.setattr(bootstrap, "_done", False)


def test_the_default_settings_module_is_set_when_none_is_configured(monkeypatch):
    monkeypatch.delenv("DJANGO_SETTINGS_MODULE", raising=False)
    monkeypatch.setattr("django.setup", lambda: None)

    bootstrap.setup_django()

    assert os.environ["DJANGO_SETTINGS_MODULE"] == bootstrap.DJANGO_SETTINGS_MODULE


def test_an_already_set_settings_module_is_left_alone(monkeypatch):
    monkeypatch.setenv("DJANGO_SETTINGS_MODULE", "custom.settings")
    monkeypatch.setattr("django.setup", lambda: None)

    bootstrap.setup_django()

    assert os.environ["DJANGO_SETTINGS_MODULE"] == "custom.settings"


def test_django_setup_only_runs_on_the_first_call(monkeypatch):
    calls = []
    monkeypatch.setattr("django.setup", lambda: calls.append(1))

    bootstrap.setup_django()
    bootstrap.setup_django()
    bootstrap.setup_django()

    assert calls == [1]


def test_a_fresh_flag_allows_booting_again(monkeypatch):
    calls = []
    monkeypatch.setattr("django.setup", lambda: calls.append(1))

    bootstrap.setup_django()
    monkeypatch.setattr(bootstrap, "_done", False)
    bootstrap.setup_django()

    assert calls == [1, 1]


def test_use_generous_llm_timeout_sets_the_env_var(monkeypatch):
    monkeypatch.delenv("LITELLM_REQUEST_TIMEOUT", raising=False)

    bootstrap.use_generous_llm_timeout()

    assert os.environ["LITELLM_REQUEST_TIMEOUT"] == bootstrap.CHAIN_TOOL_LLM_TIMEOUT_SECONDS


def test_use_generous_llm_timeout_overrides_an_existing_value(monkeypatch):
    # checker.py/trainingLoop.py always want the generous timeout - unlike
    # setup_django()'s DJANGO_SETTINGS_MODULE, this is not a "leave it alone
    # if already set" default.
    monkeypatch.setenv("LITELLM_REQUEST_TIMEOUT", "60")

    bootstrap.use_generous_llm_timeout()

    assert os.environ["LITELLM_REQUEST_TIMEOUT"] == bootstrap.CHAIN_TOOL_LLM_TIMEOUT_SECONDS
