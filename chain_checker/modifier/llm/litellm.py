from typing import TYPE_CHECKING, Any

import httpx
from django.apps import apps as django_apps
from django.conf import settings
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

from core.appconfig import ChainAppConfig
from chain_checker.modifier.llm.llm_baseclass import LLM
from chain_checker.utils.bootstrap import setup_django
from chain_checker.utils.console import link_print
from chain_checker.utils.errors import fail

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

DEFAULT_LITELLM_APP_LABEL = "tonality"
DEFAULT_LITELLM_TIER = "fast"


def _fetch_available_tiers(api_key: str) -> list[str] | None:
    """Lists the tier aliases available to this app's key, via the LiteLLM
    proxy's /models endpoint."""
    try:
        response = httpx.get(
            f"{settings.LITELLM_BASE_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        return [item["id"] for item in response.json()["data"]]
    except Exception:
        return None


def _prefer_reported(primary: int | None, fallback: int | None) -> int | None:
    # `primary if primary is not None else fallback`, not `primary or
    # fallback`: a genuinely reported 0 (e.g. a fully cached call) is a real
    # value, not a missing one, and must not be discarded in favour of the
    # other source.
    return primary if primary is not None else fallback


class _UsageCapture(BaseCallbackHandler):
    """Reads token usage and model name off the one `invoke()` call each
    `LiteLLM.__call__` makes - langchain only reports either through a
    callback, never through the return value."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens: int | None = None
        self.model_name = ""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        generation = response.generations[0][0]
        message = getattr(generation, "message", None)
        usage_metadata = getattr(message, "usage_metadata", None) or {}
        token_usage = (response.llm_output or {}).get("token_usage") or {}

        self.prompt_tokens = (
            _prefer_reported(usage_metadata.get("input_tokens"), token_usage.get("prompt_tokens"))
            or 0
        )
        self.completion_tokens = (
            _prefer_reported(
                usage_metadata.get("output_tokens"), token_usage.get("completion_tokens")
            )
            or 0
        )
        # None here, not a derived sum: LiteLLM._record_usage() owns that
        # fallback, the same way it does for every other backend.
        self.total_tokens = _prefer_reported(
            usage_metadata.get("total_tokens"), token_usage.get("total_tokens")
        )

        self.model_name = (response.llm_output or {}).get("model_name", "")


class LiteLLM(LLM):
    """Modifier backend that calls this project's own LiteLLM proxy through a
    Django app's configured credential, instead of a separate API key."""

    def __init__(
        self,
        app_label: str = DEFAULT_LITELLM_APP_LABEL,
        tier: str = DEFAULT_LITELLM_TIER,
    ) -> None:
        super().__init__()
        self._app_label = app_label
        self._tier = tier
        self._last_model_name = ""

        setup_django()
        app_config = LiteLLM._resolve_app_config(app_label)
        llm = LiteLLM._resolve_llm(app_config, app_label, tier)

        prompt = ChatPromptTemplate.from_messages([("human", "{input}")])
        self._runnable = prompt | llm

    @staticmethod
    def _resolve_app_config(app_label: str) -> ChainAppConfig:
        try:
            app_config = django_apps.get_app_config(app_label)
        except LookupError:
            fail(
                f"No Django app config found for app label '{app_label}'.\n"
                f"  Check it's a real apps/<label>/ directory (see "
                f"config/settings.py's PROJECT_APPS auto-discovery).\n"
                f"  Registered app labels: "
                f"{', '.join(LiteLLM._chain_app_labels()) or '(none found)'}."
            )

        if not isinstance(app_config, ChainAppConfig):
            fail(
                f"App '{app_label}' is not a ChainAppConfig.\n"
                f"  Found: a plain Django app ({type(app_config).__name__}), "
                f"so it has no llm() and no LiteLLM key of its own.\n"
                f"  Point --modifier-app at an apps/<label>/ app (see "
                f"core/appconfig.py).\n"
                f"  Registered app labels: "
                f"{', '.join(LiteLLM._chain_app_labels()) or '(none found)'}."
            )
        return app_config

    @staticmethod
    def _chain_app_labels() -> list[str]:
        # Apps without a ChainAppConfig have no llm() either, so they
        # can never work as --modifier-app - exclude them from the list.
        return sorted(
            config.label
            for config in django_apps.get_app_configs()
            if isinstance(config, ChainAppConfig)
        )

    @staticmethod
    def _resolve_llm(app_config: ChainAppConfig, app_label: str, tier: str) -> "BaseChatModel":
        LiteLLM._check_tier_available(app_config, app_label, tier)
        try:
            return app_config.llm(tier)
        except Exception as e:
            fail(
                f"Could not get an LLM for tier '{tier}' from app '{app_label}'.\n"
                f"  Reason: {e}\n"
                f"  Check the LiteLLM api key for this app "
                f"(LITELLM_API_KEY_{app_label.upper()}) is set, e.g. in .env.local."
            )

    @staticmethod
    def _check_tier_available(app_config: ChainAppConfig, app_label: str, tier: str) -> None:
        try:
            available = _fetch_available_tiers(app_config.litellm_api_key)
        except Exception:
            available = None

        if available is not None and tier not in available:
            fail(
                f"Tier '{tier}' is not available for app '{app_label}'.\n"
                f"  Tiers this app's key can call: {', '.join(sorted(available)) or '(none)'}.\n"
                f"  Pick one of those with --modifier-tier, or add '{tier}' to the "
                f"LiteLLM proxy config first."
            )

    def get_run_info(self) -> dict[str, str]:
        return {
            "backend": "litellm",
            "app": self._app_label,
            "tier": self._tier,
            "model": self._last_model_name,
        }

    def __call__(self, prompt: str) -> str:
        cap = _UsageCapture()
        try:
            message = self._runnable.invoke({"input": prompt}, config={"callbacks": [cap]})
        except Exception as e:
            fail(
                f"Modifier LLM call failed (app '{self._app_label}', tier "
                f"'{self._tier}').\n"
                f"  Reason: {e}\n"
                f"  Check the LiteLLM proxy is reachable and the api key for this "
                f"app (LITELLM_API_KEY_{self._app_label.upper()}) is set, e.g. in "
                f".env.local."
            )

        self._record_usage(cap.prompt_tokens, cap.completion_tokens, cap.total_tokens)
        self._last_model_name = cap.model_name

        text = getattr(message, "content", None) or ""
        return self._require_nonempty_text(
            text,
            f"App '{self._app_label}' tier '{self._tier}' generated no text content at all.\n"
            f"  The model may have run out of tokens before producing an answer.",
        )


if __name__ == "__main__":
    llm = LiteLLM()
    output = llm("Say hello in one short sentence.")
    link_print(output)
