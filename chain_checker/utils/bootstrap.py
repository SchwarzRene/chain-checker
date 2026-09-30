import os

DJANGO_SETTINGS_MODULE = "config.settings"

# Running local LLMs needs more time => timeout 1 hour
CHAIN_TOOL_LLM_TIMEOUT_SECONDS = "3600"

_done = False


def use_generous_llm_timeout() -> None:
    os.environ["LITELLM_REQUEST_TIMEOUT"] = CHAIN_TOOL_LLM_TIMEOUT_SECONDS


def setup_django() -> None:
    global _done
    if _done:
        return

    print(">>> Setting up django...")
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", DJANGO_SETTINGS_MODULE)

    import django

    django.setup()

    _done = True
