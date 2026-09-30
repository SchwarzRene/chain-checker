from types import ModuleType
from typing import Any, Protocol, runtime_checkable

from chain_checker.baseclasses.chain.think_safe_llm import make_think_safe
from chain_checker.utils.bootstrap import setup_django
from chain_checker.utils.errors import fail


@runtime_checkable
class _ChainAppConfig(Protocol):
    """Structural contract for the two capabilities a chain app must expose.
    Kept as a runtime-checkable Protocol, not a shared base class, because
    each app's concrete AppConfig lives in its own package with no reason to
    import this module."""

    def llm(self, tier: str) -> Any: ...

    def services(self) -> ModuleType: ...


def _chain_registry() -> Any:
    # Imported lazily: core.registry pulls in Django app machinery that isn't
    # configured until setup_django() (called by load_chain, below) has run.
    from core.registry import registry

    return registry


def _chains_by_app(registry: Any) -> dict[str, list[str]]:
    by_app: dict[str, list[str]] = {}
    for label, name in registry.chains:
        by_app.setdefault(label, []).append(name)
    return {label: sorted(names) for label, names in sorted(by_app.items())}


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"    - {item}" for item in items)


def load_chain(app_label: str, chain_type: str) -> Any:
    setup_django()
    registry = _chain_registry()

    try:
        chain = registry.require(app_label, chain_type)
    except Exception as e:
        chains_by_app = _chains_by_app(registry)
        if app_label not in chains_by_app:
            # No chains registered under app_label at all is at least as
            # likely to mean --type itself is misspelled as it is a real,
            # empty app - say so explicitly instead of a hard-to-parse
            # "chains available: []" for a type that was never valid.
            fail(
                f"Could not load chain '{chain_type}' for type '{app_label}'.\n"
                f"  Reason: '{app_label}' is not a known --type - no chains "
                f"are registered under that app label.\n"
                f"  Known --type values:\n{_bullet_list(sorted(chains_by_app))}"
            )
        fail(
            f"Could not load chain '{chain_type}' for type '{app_label}'.\n"
            f"  Reason: {e}\n"
            f"  Check it is registered with @register under core.registry "
            f"(see documenation/CHAIN-REQUIREMENTS.md).\n"
            f"  Chains available for '{app_label}':\n"
            f"{_bullet_list(chains_by_app[app_label])}"
        )

    return chain


def build_runnable(chain: Any, app_label: str, chain_type: str, tier: str | None = None) -> Any:
    config = _app_config(app_label)
    llm = _build_llm(config, tier or chain.tier, app_label)

    try:
        return chain.build(llm=llm, services=config.services())
    except Exception as e:
        fail(
            f"Chain '{chain_type}' failed to build.\n"
            f"  Reason: {e}\n"
            f"  build(llm, services) must return a Runnable without "
            f"making an LLM call itself (see documenation/CHAIN-REQUIREMENTS.md)."
        )


def _app_config(app_label: str) -> _ChainAppConfig:
    # Imported lazily for the same reason as _chain_registry: Django isn't
    # configured until load_chain has called setup_django().
    from django.apps import apps

    try:
        config = apps.get_app_config(app_label)
    except LookupError as e:
        fail(
            f"No Django app config found for type '{app_label}'.\n"
            f"  Reason: {e}\n"
            f"  Check --type matches an installed app label."
        )

    if not isinstance(config, _ChainAppConfig):
        fail(
            f"Django app '{app_label}' is not a chain app.\n"
            f"  Its app config ({type(config).__name__}) has no "
            f"llm()/services().\n"
            f"  Only an app whose config subclasses ChainAppConfig (see "
            f"core/appconfig.py) can be run with --type."
        )
    return config


def _build_llm(config: _ChainAppConfig, tier: str, app_label: str) -> Any:
    try:
        llm = config.llm(tier)
    except Exception as e:
        fail(
            f"Could not get an LLM for tier '{tier}' from app '{app_label}'.\n"
            f"  Reason: {e}\n"
            f"  Check the LiteLLM api key for this app is set in the "
            f"environment (see documenation/CHAIN-REQUIREMENTS.md)."
        )
    return make_think_safe(llm)
