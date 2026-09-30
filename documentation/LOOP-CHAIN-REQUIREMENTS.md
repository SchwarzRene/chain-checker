# Loop Chain Requirements

`trainingLoop.py` runs a chain exactly like `checker.py` does, case by case, computing the same metrics. Everything in [Checker Chain Requirements](CHECKER-CHAIN-REQUIREMENTS.md) applies here too — this page only covers what a chain needs *in addition* to become trainable.

---

## Required, in addition

| Name | What it does | In the example below |
|---|---|---|
| `SYSTEM_PROMPT` | the rewritable prompt template<br>- a plain module-level string a rewrite can `setattr` directly<br>- no need to run the chain or call `build()` to reach it | `SYSTEM_PROMPT = """..."""` |
| `generateSystemPrompt` | its return value is what a modifier rewrite reads and replaces | `def generateSystemPrompt(payload): ...` |

1. **`generateSystemPrompt` is optional** → only needed when `SYSTEM_PROMPT` has a `{placeholder}` that must be filled with real data before a modifier can read it. 

---

## Example of a trainable chain

Example trimmed to what matters. Turns a chain shaped like the one in [Checker Chain Requirements](CHECKER-CHAIN-REQUIREMENTS.md#what-a-chain-needs-to-look-like) into a trainable one:

```python
SYSTEM_PROMPT = """\
You review marketing text for brand compliance.

Rules:
{tonality_rules}
...
"""

USER_PROMPT = """\
Text to evaluate:

{text}
"""


def generateSystemPrompt(payload: dict) -> str:
    # Resolves SYSTEM_PROMPT into the prompt actually sent - this string is
    # exactly what a prompt-rewriting modifier reads and may reword.
    return SYSTEM_PROMPT.format(tonality_rules=render_rules_block(get_rules()))


def insertInput(prompt: str, payload: dict) -> list:
    # Appends whatever must stay hidden from the modifier, plus the
    # caller's actual content - never seen or touched by the modifier.
    return [
        SystemMessage(content=prompt),
        HumanMessage(content=USER_PROMPT.format(text=payload["text"])),
    ]
```

---

## Why the prompt is split into three pieces

This split only matters once a prompt has a `{placeholder}` that needs real, live data. A static prompt with none of those has no reason to add `generateSystemPrompt` in the first place (see above). <br>The string `generateSystemPrompt` returns is the exact string a modifier rewrite reads and edits. Splitting it into three pieces keeps the modifier from ever touching the caller's actual data.

1. **`SYSTEM_PROMPT`** → the rewritable template. Persists across calls; a modifier rewrite replaces this string directly.
2. **`generateSystemPrompt(payload)`** → resolves the template fresh on every call, so a live `{tonality_rules}` placeholder always renders the current rules, not a stale copy.
3. **`insertInput(prompt, payload)`** → appends the entry's own data (`payload["text"]`), invisible to the modifier by construction — it never sees anything built here.

---

## Calling them from `build()`

`chain_checker` only cares that `generateSystemPrompt` runs before `insertInput`, in that order. How `build()` gets there is not enforced.

Multiple examples trimmed to what matters:

<table>
<tr valign="top">
<td>

```python
@staticmethod
def build(llm: BaseChatModel, services: Any) -> Runnable:
    structured = llm.with_structured_output(TonalityChecklist, method="json_mode")

    async def _run(payload: dict) -> dict:
        prompt = generateSystemPrompt(payload)
        messages = insertInput(prompt, payload)
        checklist = await structured.ainvoke(messages)
        return checklist.model_dump()

    return RunnableLambda(_run)
```

</td>
<td>

```python
@staticmethod
def build(llm: BaseChatModel, services: Any) -> Runnable:
    structured = llm.with_structured_output(TonalityChecklist, method="json_mode")

    def _verdicts_from_items(items: list[ChecklistItem]) -> dict[str, bool]:
        # derived from items, never set independently - can't disagree
        return {item.rule_id: item.passed for item in items}

    async def _run(payload: dict) -> dict:
        prompt = generateSystemPrompt(payload)
        messages = insertInput(prompt, payload)
        checklist = await structured.ainvoke(messages)
        dumped = checklist.model_dump()
        dumped["verdicts"] = _verdicts_from_items(checklist.items)
        return dumped

    return RunnableLambda(_run)
```

</td>
</tr>
</table>

---

## Additional Information

1. **Modifier visibility** → The modifier only ever sees `Model.get_system_prompt()`'s return value: `generateSystemPrompt`'s resolved output where the chain has one, the unresolved `SYSTEM_PROMPT` otherwise. <br>Read via `get_system_prompt()`, replaced via `set_new_system_prompt()`. Anything built inside `insertInput()` stays invisible to it.

2. **Rendered placeholders** → A placeholder is shown filled in, not as a literal token: the modifier sees `{placeholder}` replaced with the actual text. <br>Its rewrite can leave `{placeholder}` written back out (still auto-filled every call) or write the rendered rules directly (frozen from then on) — `str.format` accepts either.

3. **Empty-payload resolve** → Whenever the current prompt needs reading (writing an epoch's report, handing it to the modifier), `get_system_prompt()` calls `generateSystemPrompt({})` fresh, outside any corpus entry. <br>A `generateSystemPrompt` that reads `payload["text"]` anyway raises on that call (e.g. `KeyError`) — caught, a warning logged, falling back to the unresolved `SYSTEM_PROMPT` instead of ending the run.

4. **Orchestrators aren't trainable** → A chain that only composes other chains' already-built runnables has none of this trio. `checker.py` still runs it fine (see [Checker Chain Requirements](CHECKER-CHAIN-REQUIREMENTS.md)); `trainingLoop.py` catches the resulting `AttributeError` at startup and fails with a clear reason instead of wasting an epoch on it.

5. **Brace escaping** → A rewrite that pastes in a JSON example like `{"rule_id": "...", "passed": true}` is safe: `set_new_system_prompt()` doubles every `{`/`}` that isn't a declared placeholder (auto-detected from `SYSTEM_PROMPT`) before storing it, so the next `.format(...)` neither raises nor substitutes the wrong thing.

6. **Near-miss placeholders** → A rewrite that mistypes `{place_holder}` as `{ place_holder }` or `{place-holder}` is flagged by `find_near_miss_placeholders()`. <br>`set_new_system_prompt()` turns that into a `ValueError` naming it, instead of the keys silently disappearing from every future call.
