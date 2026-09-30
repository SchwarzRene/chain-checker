# Checker Chain Requirements

`checker.py` runs a chain against a fixed set of labeled test cases. Everything below is checked before any case is run, and applies to any registered chain — leaf chain or pure orchestrator.

---

## Required Variable-Names

Found **by name** — when a name is not found, it silently stops the chain being loadable:

| Chain-Class-Vars | What it does | In the example below |
|------------------|---|---|
| `name`           | the chain-type name — what `--chain-type` selects | `name = "template_checklist"` |
| `InputSchema`    | validates a corpus case's `input:` before the chain ever runs | `class InputSchema(BaseModel): text: str; threshold: float` |
| `OutputSchema`   | declares every top-level key a corpus case's `output:` can check | `OutputSchema = ExampleChecklist` |
| `tier`           | which LLM tier `build()` gets called with | `tier: str = "fast"` |
| `build`          | builds the `Runnable` that actually answers a call | `@staticmethod def build(...) -> Runnable: ...` |


## What a chain needs to look like

Example trimmed to what matters:

```python
from core.registry import register
from workflows.<type>.schemas import ExampleChecklist, _MAX_TEXT_LENGTH


@register
class TemplateChecklistChain:
    name: str = "template_checklist"
    tier: str = "fast"

    class InputSchema(BaseModel):
        text: str = Field(..., max_length=_MAX_TEXT_LENGTH)
        threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    OutputSchema = ExampleChecklist

    @staticmethod
    def build(llm, services) -> Runnable:
        structured = llm.with_structured_output(ExampleChecklist, method="json_mode")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You review marketing text for brand compliance."),
            ("human", "Text to evaluate:\n\n{text}"),
        ])

        async def _run(payload: dict) -> dict:
            checklist = await (prompt | structured).ainvoke(payload)
            dumped = checklist.model_dump()
            dumped["verdicts"] = {i.rule_id: i.passed for i in checklist.items}
            return dumped

        return RunnableLambda(_run)
```
---

## Chain's output vs. Corpus Entry's output:

Every rule below mentioned is checked bevor any model call. 
The chain's build() returns a Runnable whose reply becomes a plain dict (a pydantic model gets .model_dump()-ed). OutputSchema and a case's output: block each constrain that dict from a different side, and comparison happens key by key between them:

1. **Allowed Keys** → In the .yaml file are only keys allowed which also appear in the OutputSchema of the model. Using an unknown key fails immediately, before any case runs.
2. **Compared Keys** → Only the keys from the .yaml file are checked. Every other field the chain returns is simply ignored for that case.
3. **Missing Key** → If the chain does not return a key a case checks, that key is scored wrong, not an error. The rest of the case, and the run, continue normally.
4. **Different value Types** → 
   - Booleans must match exactly
   - Numbers allow a tiny rounding difference
   - A dict of only true/false values is scored one sub-rule per key
   - Everything else is compared as one whole value
5. **Value is a dict** → Is the value of a key itself a dict so must each key appear in the models output else it is counted as a wrong entry.
6. **Continuous Key Type** → once one case checks `score` as a number, every other case checking `score` must also use a number. Mixing types fails immediately.

### Side by side

Chain Return Object (left) vs. Corpus Case (right). <br>
verdicts/score/passed are checked, the rest not because it does not appear in the .yaml file 

<table>
<tr>
<th>Chain's <code>OutputSchema</code>, dumped by <code>build()</code></th>
<th>Matching corpus case</th>
</tr>
<tr valign="top">
<td>

```python
class ChecklistItem(BaseModel):
    rule_id: str
    passed: bool


class ExampleChecklist(BaseModel):
    items: list[ChecklistItem]
    score: float
    passed: bool


# what the chain's _run() returns,
# after checklist.model_dump() and
# adding "verdicts" (see the example
# chain above):
{
    "items": [
        {"rule_id": "brand-uppercase", "passed": True},
        {"rule_id": "no-exclamation-spam", "passed": True},
        {"rule_id": "no-competitor-mentions", "passed": True},
    ],
    "verdicts": {
        "brand-uppercase": True,
        "no-exclamation-spam": True,
        "no-competitor-mentions": True,
    },
    "score": 1.0,
    "passed": True,
}
```

</td>
<td>

```yaml
- id: 2
  input:
    text: "With ACME, your home just works."
    threshold: 0.8
  output:
    verdicts:
      brand-uppercase: true
      no-exclamation-spam: true
      no-competitor-mentions: true
    score: 1.0
    passed: true
```

</td>
</tr>
</table>

### Additional Information
The builder of a chain does not need to worry about calling the model invoke or ainvoke.
Bevor any model call is make it is checked by the chain-checker what type the chain uses and calls then the appropriate one.
1. **No build** → A build() that returns something with neither ainvoke nor invoke — fails immediately, with a message naming what it actually got.
2. **(a)invoke choice** → Model calls .invoke(). This is the only condition that switches it to sync — every LangChain Runnable provides ainvoke, so it's rare. If the runnable does have ainvoke, Model always calls it; no other condition ever routes it to .invoke() instead.
3. **Unparsable OutputSchema:** → A reply that fails to parse into OutputSchema will be counted in Parsing-Metrics. The model does not try to call the other invoke version and continues with the rest of the dataset.

