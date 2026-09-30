import types

import pytest

from chain_checker.baseclasses.chain import builder


class _Chain:
    tier = "fast"

    class InputSchema:
        pass

    def __init__(self, runnable="a runnable", build_raises=None) -> None:
        self.built: list = []
        self._runnable = runnable
        self._build_raises = build_raises

    def build(self, llm, services):
        self.built.append((llm, services))
        if self._build_raises is not None:
            raise self._build_raises
        return self._runnable


class _Registry:
    def __init__(self) -> None:
        self.asked: list = []
        self.chain = _Chain()
        self.error: Exception | None = None
        self.chains: dict[tuple[str, str], object] = {}

    def require(self, app_label, chain_type):
        self.asked.append((app_label, chain_type))
        if self.error is not None:
            raise self.error
        return self.chain


class _AppConfig:
    def __init__(self, llm_raises=None) -> None:
        self.tiers: list = []
        self._llm_raises = llm_raises

    def llm(self, tier):
        self.tiers.append(tier)
        if self._llm_raises is not None:
            raise self._llm_raises
        return "an llm"

    def services(self):
        return "the services module"


@pytest.fixture
def registry(monkeypatch):

    monkeypatch.setattr(builder, "setup_django", lambda: None)
    fake = _Registry()
    monkeypatch.setattr(builder, "_chain_registry", lambda: fake)
    return fake


@pytest.fixture
def app_config(monkeypatch):
    config = _AppConfig()
    monkeypatch.setattr("django.apps.apps.get_app_config", lambda label: config)
    return config


def test_the_registered_chain_is_what_comes_back(registry):
    assert builder.load_chain("tonality", "template_checklist") is registry.chain


def test_the_chain_is_looked_up_by_app_label_and_chain_type(registry):
    builder.load_chain("tonality", "template_checklist")

    assert registry.asked == [("tonality", "template_checklist")]


def test_django_is_booted_before_the_registry_is_read(monkeypatch):

    order = []
    monkeypatch.setattr(builder, "setup_django", lambda: order.append("boot"))

    def _read_registry():
        order.append("registry")
        return _Registry()

    monkeypatch.setattr(builder, "_chain_registry", _read_registry)

    builder.load_chain("tonality", "template_checklist")

    assert order == ["boot", "registry"]


def test_a_chain_the_registry_does_not_have_says_how_to_register_it(registry):
    registry.error = KeyError("no such chain")
    registry.chains[("tonality", "other_chain")] = object()

    with pytest.raises(SystemExit, match="registered with @register"):
        builder.load_chain("tonality", "template_checklist")


def test_a_chain_that_could_not_be_loaded_names_the_type_and_the_chain(registry):
    registry.error = KeyError("no such chain")
    registry.chains[("tonality", "other_chain")] = object()

    with pytest.raises(SystemExit, match="'template_checklist' for type 'tonality'"):
        builder.load_chain("tonality", "template_checklist")


def test_a_chain_that_could_not_be_loaded_lists_the_apps_registered_chains(registry):
    registry.error = KeyError("no such chain")
    registry.chains[("tonality", "real_template_checklist")] = object()
    registry.chains[("tonality", "local_template_checklist")] = object()

    with pytest.raises(
        SystemExit,
        match=r"Chains available for 'tonality':\n"
        r"    - local_template_checklist\n"
        r"    - real_template_checklist",
    ):
        builder.load_chain("tonality", "template_checklist")


def test_a_chain_that_could_not_be_loaded_does_not_repeat_the_app_and_chain_type(registry):
    registry.error = KeyError("no such chain")
    registry.chains[("tonality", "other_chain")] = object()

    with pytest.raises(SystemExit) as excinfo:
        builder.load_chain("tonality", "template_checklist")

    assert "app label" not in str(excinfo.value)
    assert "chain-type name" not in str(excinfo.value)


def test_a_type_with_no_chains_at_all_is_named_as_an_unknown_type(registry):
    # A typo'd --type (e.g. 'tonalit' for 'tonality') has no
    # chains registered under it at all - that must be reported as an unknown
    # --type, not as "chains available: []" for a type that never existed.
    registry.error = KeyError("no such chain")
    registry.chains[("tonality", "template_checklist")] = object()
    registry.chains[("echo", "template_checklist")] = object()

    with pytest.raises(
        SystemExit,
        match=r"'nope' is not a known --type[\s\S]*"
        r"Known --type values:\n    - echo\n    - tonality",
    ):
        builder.load_chain("nope", "template_checklist")


def test_an_unknown_type_error_still_names_the_requested_chain(registry):
    registry.error = KeyError("no such chain")

    with pytest.raises(SystemExit, match="'template_checklist' for type 'nope'"):
        builder.load_chain("nope", "template_checklist")


def test_the_chain_is_built_with_the_apps_llm_and_services(app_config):
    chain = _Chain()

    runnable = builder.build_runnable(chain, "tonality", "template_checklist")

    assert chain.built == [("an llm", "the services module")]
    assert runnable == "a runnable"


def test_the_llm_handed_to_build_is_made_think_safe(app_config):
    # A plain object (unlike the "an llm" string other tests use) actually
    # supports the __class__ swap make_think_safe does - confirms
    # build_runnable really does route every chain's llm through it, not
    # just that a string happens to pass through unScathed.
    class _FakeLLM:
        def with_structured_output(self, schema, *, method="function_calling", **kwargs):
            raise AssertionError("not exercised by this test")

    app_config.llm = lambda tier: _FakeLLM()
    chain = _Chain()

    builder.build_runnable(chain, "tonality", "template_checklist")

    llm, _services = chain.built[0]
    assert type(llm).__name__ == "ThinkSafe_FakeLLM"


def test_the_llm_is_asked_for_the_chains_own_tier(app_config):
    class _Thinking(_Chain):
        tier = "thinking"

    builder.build_runnable(_Thinking(), "tonality", "template_checklist")

    assert app_config.tiers == ["thinking"]


def test_an_app_that_is_not_installed_says_to_check_the_type_flag(monkeypatch):
    def _missing(label):
        raise LookupError(f"No installed app with label '{label}'")

    monkeypatch.setattr("django.apps.apps.get_app_config", _missing)

    with pytest.raises(SystemExit, match="--type matches an installed app label"):
        builder.build_runnable(_Chain(), "nope", "template_checklist")


def test_an_installed_app_that_is_not_a_chain_app_is_named_as_such(monkeypatch):

    monkeypatch.setattr("django.apps.apps.get_app_config", lambda label: types.SimpleNamespace())

    with pytest.raises(SystemExit, match="is not a chain app"):
        builder.build_runnable(_Chain(), "admin", "template_checklist")


def test_an_llm_that_cannot_be_built_points_at_the_api_key(monkeypatch):
    monkeypatch.setattr(
        "django.apps.apps.get_app_config",
        lambda label: _AppConfig(llm_raises=RuntimeError("no key configured")),
    )

    with pytest.raises(SystemExit, match="LiteLLM api key"):
        builder.build_runnable(_Chain(), "tonality", "template_checklist")


def test_a_build_that_raises_says_what_build_owes_the_caller(app_config):
    chain = _Chain(build_raises=RuntimeError("the graph is not wired up"))

    with pytest.raises(SystemExit, match="must return a Runnable"):
        builder.build_runnable(chain, "tonality", "template_checklist")


def test_the_reason_a_build_failed_is_kept_in_the_message(app_config):
    chain = _Chain(build_raises=RuntimeError("the graph is not wired up"))

    with pytest.raises(SystemExit, match="the graph is not wired up"):
        builder.build_runnable(chain, "tonality", "template_checklist")
