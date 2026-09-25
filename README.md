*This project has been created as part of the 42 curriculum by lospina-.*

# Call Me Maybe

## Description

**Call Me Maybe** is a function-calling tool that translates natural-language prompts into structured, schema-compliant function calls, using **constrained decoding** with a small local LLM (`Qwen/Qwen3-0.6B`).

Given a prompt such as *"What is the sum of 2 and 3?"*, the goal is not to make the model answer the question in plain text. Instead, the program must return a structured call:

```json
{
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

Small language models (0.5B–0.6B parameters) are notoriously unreliable at producing valid, schema-compliant JSON on their own — success rates around 30% are common when simply prompted for structured output. This project does not rely on the model "guessing" a correct JSON shape. Instead, it guides the model's generation **token by token**, masking out any token that would break the expected JSON schema, so that the final output is always valid and always matches the function definitions it was given.

For every prompt in `data/input/function_calling_tests.json`, the program:
1. Selects the correct function name from `data/input/function_definitions.json`.
2. Generates a value for each of that function's parameters, respecting its declared type.
3. Writes the result to `data/output/function_calling_results.json` as a JSON array of `{prompt, name, parameters}` objects.

## Instructions

### Requirements
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk` package (provided separately via the 42 intranet), placed at the root of this repository, next to `src/`

### Setup

```bash
# 1. Clone this repository
git clone <this-repo-url>
cd call-me-maybe

# 2. Download llm_sdk from the 42 intranet and place it at the project root:
#    call-me-maybe/
#    ├── llm_sdk/     <-- goes here
#    ├── src/
#    └── ...

# 3. Install dependencies
make install
# equivalent to: uv sync
```

`uv sync` installs `pydantic`, `numpy`, and `llm_sdk` (declared as a local path dependency in `pyproject.toml`). No manual `pip install` steps are required.

### Running

```bash
make run
# equivalent to: uv run python -m src
```

By default the program reads `function_calling_tests.json` and `function_definitions.json` from `data/input/`, and writes `data/output/function_calling_results.json`.

Custom paths can be supplied:

```bash
uv run python -m src --input path/to/input_dir --output path/to/output_dir
```

### Other Makefile targets

| Target | Description |
|---|---|
| `make install` | Install dependencies (`uv sync`) |
| `make run` | Run the program |
| `make debug` | Run the program under `pdb` |
| `make lint` | Run `flake8` and `mypy` (required flags) |
| `make lint-strict` | Run `mypy --strict` (optional, stricter check) |
| `make clean` | Remove caches (`__pycache__`, `.mypy_cache`, `.venv`) |

The first run will download the `Qwen/Qwen3-0.6B` model weights from Hugging Face Hub and cache them locally; subsequent runs reuse the cache.

## Algorithm Explanation

The core problem with small LLMs and structured output is that, left unconstrained, the model frequently produces text that isn't valid JSON, or that drifts away from the requested schema after a few tokens. Constrained decoding solves this by intervening directly in the generation loop, **before** a token is sampled, rather than trying to fix the output afterwards.

**The generation loop**, at a high level:

1. Encode the current context (prompt + instructions + tokens generated so far) into `input_ids`.
2. Call `get_logits_from_input_ids` to get a raw logit score for every token in the vocabulary — a score of how likely the model thinks each token is to come next.
3. For every token in the vocabulary, check whether appending it to what has been generated so far would still be **valid** given the current constraint. If not, its logit is set to `-inf`, removing it from consideration entirely.
4. Pick the token with the highest remaining logit (never `-inf`).
5. Append it to the generated text and to `input_ids`, and repeat until a stopping condition specific to what is being decoded is reached.

This project applies that loop with **different validity rules depending on what is being generated**:

- **Function name** (`fn_name`): the set of valid function names is known in advance (from `function_definitions.json`). A candidate is valid if it is a *prefix* of at least one known function name (`function_name.startswith(candidate)`), and generation stops the moment the candidate exactly matches one of them.
- **Number parameters**: the candidate must remain a syntactically valid number (digits, an optional decimal point, an optional leading minus sign) at every step. Generation stops once the model proposes a token that would break that pattern (a space, a comma, a letter), at which point the value collected so far is kept.
- **String parameters**: almost any token is allowed, since string content is largely unconstrained text. Generation stops when a quote character (or another explicit stop character, such as a control character or comma) appears, similar to how a JSON string is delimited.
- **Boolean parameters**: the candidate must be a prefix of either `"true"` or `"false"`, using the same prefix-matching approach as function names.

The purely structural parts of the JSON (braces, colons, commas, key names) are **not** generated by the model at all — they are written directly by the program in Python, since there is no ambiguity in how they should look. The model is only consulted for the parts that genuinely require understanding the natural-language request: which function to call, and what value each parameter should take.

## Design Decisions

- **Type-specific decoding functions.** Rather than a single generic decoder, each parameter type (`number`, `string`, `boolean`) has its own decoding function with its own validity check and stop condition, since each type has fundamentally different rules for what a "valid partial value" looks like.
- **Context construction per step.** A separate context string is built for the function-name decision and for each parameter, including the original prompt, the selected function's name and description, and (for parameters after the first) the values already found for previous parameters. This was one of the most impactful design choices: early, minimal contexts caused the model to guess incorrectly far more often than contexts that explicitly named the function's purpose and the exact parameter being requested.
- **Vocabulary lookup precomputed once.** `llm_sdk` exposes `get_path_to_vocab_file()`, which returns a `vocab.json` mapping token strings to token IDs. This is inverted once (`id -> token string`) at startup and reused for the entire run, instead of calling `model.decode()` for every candidate token at every generation step — the latter was measured to be prohibitively slow (potentially minutes per token).
- **Pydantic models throughout.** `FunctionDefinition`, `PromptItem`, and `FunctionCallResult` are all Pydantic `BaseModel`s. This gives type validation "for free" (including a custom `PType` enum for parameter types) and produces clear, structured errors when an input file doesn't match the expected shape.
- **Fail gracefully at the file level, fail loudly at the record level.** A missing input file or malformed JSON at the top level raises a custom exception and stops the program with a clear message. A single malformed *entry* inside an otherwise valid JSON array (one bad function definition, one bad prompt) is logged and skipped, so that one bad record doesn't discard an entire otherwise-usable input file.

## Performance Analysis

- **Accuracy:** on the provided public test set, all sample prompts pass. Against the private evaluation test set, the program correctly resolves approximately 90% of prompts.
- **Reliability of JSON validity:** 100% — because the output is assembled from validated Python dictionaries via Pydantic and `json.dump`, malformed JSON is not possible regardless of what the model generates internally.
- **Speed:** individual prompts typically resolve in well under a second once the model is loaded (model loading itself takes a few seconds); the full test suite completes well within the 5-minute budget.
- **Where accuracy drops:** almost all remaining failures involve parameters that require the model to construct something *not* literally present in the prompt — most notably regular-expression patterns (e.g. turning "replace all vowels" into `[aeiouAEIOU]`). Extracting a literal value from the prompt (a name, a number, a quoted phrase) is highly reliable; generating an abstract symbolic representation of a concept is a genuinely harder task for a 0.6B model and is the main source of remaining errors.

## Challenges Faced

- **Confusing token strings with token IDs.** Early versions repeatedly called `model.decode()` inside the per-token validity loop to find out what text a given token ID corresponds to. This made the vocabulary scan enormously slow. The fix was to load `vocab.json` once and invert it into an `id -> token` dictionary, turning a model call into a dictionary lookup.
- **Empty-context crash.** Passing an empty string as the initial generation context caused a `RuntimeError` inside the model's attention layers (a zero-element tensor). The context always needs to contain a real instruction/prompt before the first token is generated.
- **Tokenizer special characters.** The BPE tokenizer used by Qwen3 represents a leading space as `Ġ` and a newline as `Ċ` in its vocabulary strings, rather than literal whitespace characters. Comparing generated candidates against the raw prompt failed until these were translated back to real whitespace before comparison.
- **Backslash duplication.** For string values containing backslashes (e.g. Windows file paths), the tokenizer sometimes emits a token whose literal text is two backslash characters instead of one, inconsistently across positions in the same string. This was resolved with a normalization step that collapses any run of consecutive backslashes down to one before storing the final value.
- **Unbounded/looping generation.** Without an explicit stop condition tied to the type being generated, the model would occasionally continue producing tokens indefinitely (e.g. an endless sequence of numbers), especially once type-specific character restrictions were relaxed to allow more expressive values (like regex patterns). This required adding type-specific stop conditions rather than relying on the model to spontaneously decide it was done.
- **`llm_sdk` import surfacing at module load time.** Wrapping `Small_LLM_Model()` instantiation in a `try/except ImportError` inside `main()` did not catch missing-module errors, because other modules imported `Small_LLM_Model` directly at the top of the file for type hints, which fails at import time, before any function runs. This was resolved by importing it only under `typing.TYPE_CHECKING` and using the class name as a string literal in type hints elsewhere in the codebase.

## Testing Strategy

- **Manual testing against the provided sample prompts and function definitions** during development, verifying the printed intermediate reasoning (chosen function, parameter values found) at each step of the pipeline.
- **Deliberately malformed/missing input testing:** running the program against a missing `function_definitions.json`, a syntactically invalid JSON file, a JSON file whose root is not an array, and individual malformed entries within an otherwise valid array, confirming that each produces a clear, non-crashing error message on `stderr`.
- **Edge-case prompts:** prompts with quoted substrings, embedded numbers inside strings, Windows-style file paths with backslashes, negated boolean phrasing ("without strict mode" vs. "using strict mode"), and multi-parameter functions where later parameters must be distinguished from earlier ones in the same context.
- **Interrupt handling:** verifying `Ctrl+C` during a run is caught and reported cleanly rather than producing a raw traceback.
- **Static checks:** `flake8` and `mypy` (both the required flag set and `--strict`) are run via `make lint` / `make lint-strict` before every commit.

## Example Usage

```bash
make install
make run
```

Given `data/input/function_calling_tests.json`:
```json
[
  "What is the sum of 2 and 3?",
  "Greet shrek"
]
```

And a matching `data/input/function_definitions.json` entry for `fn_add_numbers` and `fn_greet`, running the program produces `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  }
]
```

Custom input/output locations:
```bash
uv run python -m src --input data/input --output data/output
```

## Resources

- [Constrained Decoding: How LLMs Generate Valid JSON](https://gradientupdate.substack.com/p/constrained-decoding-how-llms-generate) — overview of logit masking for guaranteed-valid JSON generation.
- [Structured Output from LLMs: How to Get Guaranteed JSON Every Time](https://www.morphllm.com/structured-output-llm) — comparison of best-effort prompting, JSON mode, and true structured output via constrained decoding.
- [Constrained Decoding: How to Force an LLM to Speak Valid JSON Without Slowing It Down](https://mohakchugh.is-a.dev/blog/constrained-decoding-fsm-grammar-guided-llm-structured-output) — background on grammar/FSM-based token masking and its performance characteristics.
- [Pydantic documentation](https://docs.pydantic.dev/) — used for all input/output data validation in this project.
- [Hugging Face `transformers` documentation](https://huggingface.co/docs/transformers/) — underlying library used by `llm_sdk` to load and run `Qwen/Qwen3-0.6B`.
- Personal learning log for this project (process notes, dead ends, and decisions as they were made): [Notion — Call me maybe](https://app.notion.com/p/Call-me-maybe-3c9ef47991ff80a1a907c30d670397c5)

### AI Usage

An AI assistant (Claude) was used throughout this project as a **tutor**, not as a code generator. Concretely, it was used to:
- Explain concepts needed to understand the assignment (tokenization, logits, constrained decoding, Pydantic validation, `uv`/packaging mechanics) before writing any related code.
- Review code written independently, pointing out bugs, logic errors, and inconsistencies (e.g. incorrect loop conditions, variable scoping issues, mismatched Pydantic field types) without providing corrected code unless explicitly requested.
- Help diagnose runtime issues by suggesting targeted debugging steps (e.g. inspecting `repr()` output to catch a token-duplication bug, checking `__file__` to diagnose a broken package installation).
- Discuss design trade-offs (e.g. how to handle a partially invalid input file, how to build effective prompts for parameter extraction) and reasoning about the limits of a 0.6B-parameter model on more abstract sub-tasks (such as regex generation).

All code in `src/` was written by the author; the AI assistant did not generate implementation code for the core decoding logic, Pydantic models, or file I/O.