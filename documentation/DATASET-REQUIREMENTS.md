# Dataset Requirements

`checker.py` and `trainingLoop.py` both run a chain against a fixed set of
labeled test cases. This page describes the .yaml shape a
dataset file needs, and what gets checked before any case is run.

---

## What the corpus `.yaml` needs to look like

Required keys for the corpus dataset:

```yaml
cases:
  - id: xy
    input:
      key: value
    output:
      key: value
    info:       # optional
      key: value
```
- `cases` — Stores the entries. Is a required key
- `id` — must be a string or integer (not a float/bool/list/mapping), unique within the file, and safe to use as a file name (the run cache writes each prediction to `<id>.txt`). Mixing string and integer ids is fine as long as none of them collide once stringified (`id: 2` and `id: "2"` do collide; `id: 2` and `id: "3"` don't).
- `input` — matches `InputSchema` exactly: same keys, same shape. Caught
  before any real model call if it doesn't.
- `output` — output keys of the model which should be compared. Does not need to include all of them but at least one of them
- `info` — optional, never compared. `language`/`modification`/`labeller`. anything else is just for a human reading the file.
---
**Example:**
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
---
### Where to place the file:

There are multiple options:
- When exactly one `.yaml` sits directly under `workflows/<type>/`, both
tools find it automatically if `--file` isn't given.
- The file path can also be passed through `--file`.
- A held-out validation corpus can be passed through `--val-file` —
it's the same shape, just passed explicitly and never auto-discovered.
