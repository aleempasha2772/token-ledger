```markdown
# token-ledger

> Track every token. Know every cost. Zero changes to your existing code.

A lightweight Python library that adds **one decorator** to your LLM functions
and instantly gives you token usage, cost per call, session totals, budget
enforcement, and a full cost report — across OpenAI, Anthropic, Google Gemini,
and 1600+ models via LiteLLM.

---

## Why token-ledger?

Every LLM call costs money. But most developers only find out how much they
spent at the end of the month when the invoice arrives.

**token-ledger makes cost visible in real time** — per call, per session, per
model — so you can catch expensive bugs early, enforce budgets in production,
and optimize before costs spiral.

| Problem | What token-ledger does |
|---|---|
| "I don't know how many tokens my prompts use" | Logs input + output tokens on every call |
| "My RAG pipeline is expensive but I don't know why" | Tracks embedding cost separately from completion cost |
| "My agent ran in a loop and cost $50" | Budget cap raises an exception before it goes further |
| "I need to report LLM costs by feature" | Per-call naming lets you group and analyze costs |
| "I want to see cost without changing my code" | One decorator — zero changes to existing logic |

---

## When to Use token-ledger

Use it when you are building any of the following:

- **Chatbots and assistants** — track cost per conversation
- **RAG pipelines** — see embedding cost vs completion cost separately
- **AI agents** — trace cost across every step of a multi-step workflow
- **Multi-model applications** — compare costs across OpenAI, Claude, and Gemini in one report
- **Production LLM systems** — enforce spending limits and get alerted before budgets are crossed
- **Development and debugging** — understand exactly what each function call costs before you ship

---

## What token-ledger Tracks

For every LLM call:

```
{
  "model":          "gpt-4o",
  "input_tokens":   1200,
  "output_tokens":  350,
  "total_tokens":   1550,
  "cost":           0.004225,
  "call_type":      "completion",
  "status":         "success",
  "latency_ms":     843.2,
  "timestamp":      "2026-03-28T14:22:01Z"
}
```

Across your full session:

- Total tokens consumed
- Total cost in USD
- Cost broken down by model
- Embedding cost vs completion cost (critical for RAG)
- Retry costs (failed attempts that still cost money)
- Agent call trees (which sub-call triggered which)

---

## Installation

```bash
pip install token-ledger
```

Install with your provider's SDK:

```bash
# OpenAI
pip install token-ledger[openai]

# Anthropic
pip install token-ledger[anthropic]

# Google Gemini
pip install token-ledger[gemini]

# All providers + tiktoken for accurate token counting
pip install token-ledger[all]
```

**Zero hard dependencies.** The core library installs nothing extra.
Provider SDKs and tokenizers are optional — only install what you use.

---

## Quick Start

```python
from token_ledger import configure, track_llm_cost
from openai import OpenAI

# 1. Configure once at startup
configure(project_name="my_app", budget=5.00)

client = OpenAI()

# 2. Add the decorator — nothing else changes
@track_llm_cost(model="gpt-4o")
def generate(messages):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

# 3. Call your function normally
response = generate([{"role": "user", "content": "Hello!"}])
```

Console output after each call:

```
[C] generate | in=12 out=18 | $0.000225
```

On program exit, a summary prints automatically and a report file is saved:

```
======================================================
  TOKEN LEDGER — COST SUMMARY
======================================================
  Project      : my_app
  Total Calls  : 1
  Input Tokens : 12
  Output Tokens: 18
  Total Tokens : 30
------------------------------------------------------
  Total Cost   : $0.000225
  Budget       : $5.00 [OK]
  Remaining    : $4.999775
------------------------------------------------------
  Cost by Model:
    gpt-4o                               $0.000225
======================================================
```

Report saved to: `my_app_token_cost.txt`

---

## Features

### One Decorator — Any Function

```python
@track_llm_cost(model="gpt-4o")
def my_llm_function():
    ...
```

Works with sync functions, async functions, and streaming — with no changes to
the function body.

---

### Async Support

```python
import asyncio
from openai import AsyncOpenAI
from token_ledger import track_llm_cost, configure

configure(project_name="async_app")
client = AsyncOpenAI()

@track_llm_cost(model="gpt-4o")
async def generate_async(messages):
    return await client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

asyncio.run(generate_async([{"role": "user", "content": "Hello!"}]))
```

---

### Budget Enforcement

Stop runaway agents and unexpected cost spikes before they hit your invoice.

```python
from token_ledger import configure, track_llm_cost, BudgetExceededError

configure(project_name="my_app", budget=1.00)

@track_llm_cost(model="gpt-4o")
def generate(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages)

try:
    while True:
        generate([{"role": "user", "content": "Keep going..."}])
except BudgetExceededError as e:
    print(f"Stopped at ${e.total_cost:.4f} — budget was ${e.budget:.2f}")
```

---

### Embedding Cost Tracking (Essential for RAG)

Track indexing and retrieval costs separately from generation costs.

```python
from token_ledger import configure, track_llm_cost, track_embedding_cost
from openai import OpenAI

configure(project_name="rag_pipeline", budget=5.00)
client = OpenAI()

@track_embedding_cost(model="text-embedding-3-small")
def embed_documents(texts):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

@track_llm_cost(model="gpt-4o")
def answer_question(messages):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

# Index your documents
embed_documents(["Document 1 content", "Document 2 content"])

# Answer a query
answer_question([
    {"role": "system", "content": "Use the retrieved context to answer."},
    {"role": "user",   "content": "What is in document 1?"}
])
```

Session summary will show:

```
  Total Cost      : $0.000642
  Embedding Cost  : $0.000004
  Completion Cost : $0.000638
```

---

### Multi-Provider in One Session

Track spend across different providers in a single unified report.

```python
from token_ledger import configure, track_llm_cost
from openai    import OpenAI
from anthropic import Anthropic

configure(project_name="multi_provider_app")

openai_client    = OpenAI()
anthropic_client = Anthropic()

@track_llm_cost(model="gpt-4o", name="openai_call")
def call_openai(prompt):
    return openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

@track_llm_cost(model="claude-3-5-sonnet-20241022", name="claude_call")
def call_claude(prompt):
    return anthropic_client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

call_openai("Summarize this document.")
call_claude("Verify the summary.")
```

Cost by model in the summary:

```
  Cost by Model:
    gpt-4o                               $0.003250
    claude-3-5-sonnet-20241022           $0.008100
```

---

### Agent Call Tree Tracking

When one decorated function calls another, token-ledger automatically
links them as parent and child — showing you which agent step caused
which cost.

```python
from token_ledger import configure, track_llm_cost

configure(project_name="agent_system")

@track_llm_cost(model="gpt-4o", name="retriever")
def retrieve(query):
    return client.chat.completions.create(...)

@track_llm_cost(model="gpt-4o", name="summarizer")
def summarize(context):
    return client.chat.completions.create(...)

@track_llm_cost(model="gpt-4o", name="orchestrator")
def run_agent(query):
    retrieved  = retrieve(query)       # depth=1, parent=orchestrator
    summarized = summarize(retrieved)  # depth=1, parent=orchestrator
    return summarized
```

Console output shows the hierarchy:

```
[C] orchestrator | in=850 out=120  | $0.003250
  [C] retriever  | in=200 out=800  | $0.009500
  [C] summarizer | in=950 out=250  | $0.007250
```

---

### On-Complete Callback

Run your own code after every tracked call — push to a database, send to
a monitoring system, or log to a file.

```python
from token_ledger import track_llm_cost, Trace

def on_each_call(trace: Trace):
    print(f"[MONITOR] {trace.model} | ${trace.cost.total_cost:.6f}")
    # push to your database, Datadog, Grafana, etc.

@track_llm_cost(model="gpt-4o", on_complete=on_each_call)
def generate(messages):
    return client.chat.completions.create(model="gpt-4o", messages=messages)
```

---

### Query Session Totals Anytime

Access live aggregates during your program — not just at the end.

```python
from token_ledger import get_session

session = get_session()

print(session.total_cost)        # float — total USD spent
print(session.total_tokens)      # int — all tokens across all calls
print(session.cost_by_model)     # dict — {model_name: cost}
print(len(session.get_traces())) # int — number of calls recorded
```

---

### Automatic Exit Report

When your program ends — normally, via Ctrl-C, or from an exception —
token-ledger automatically:

1. Prints the full cost summary to the console
2. Saves `{project_name}_token_cost.txt` to your configured output directory

The file contains the full call log, aggregated totals, per-model breakdown,
and raw JSON for programmatic processing.

Disable it if you don't want it:

```python
configure(project_name="my_app", save_on_exit=False, print_each=False)
```

---

### Supported Models

token-ledger uses **LiteLLM's pricing database** as its source of truth,
covering 1600+ models across every major provider.

| Provider | Examples |
|---|---|
| OpenAI | gpt-4o, gpt-4o-mini, o1, o3-mini, gpt-3.5-turbo |
| Anthropic | claude-3-5-sonnet, claude-3-5-haiku, claude-opus |
| Google | gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash |
| Mistral | mistral-large, mistral-small |
| DeepSeek | deepseek-chat, deepseek-reasoner |
| xAI | grok-2-latest |
| AWS Bedrock | All Claude and Llama deployments |
| Azure OpenAI | All Azure GPT deployments |
| Ollama | Local models via dict response format |
| Any OpenAI-compatible server | vLLM, LM Studio, Groq, Together AI |

If a model is not found in the pricing database, token-ledger logs a warning
and records a cost of $0.00 — it never crashes your application.

---

## Full API Reference

### `configure()`

Call once at the start of your application before any decorated functions run.

```python
from token_ledger import configure

configure(
    project_name = "my_app",      # used for the output filename
    budget       = 5.00,          # optional — raises BudgetExceededError when crossed
    output_dir   = ".",           # where to save the .txt report
    print_each   = True,          # print a log line after every call
    save_on_exit = True,          # write the report file on program exit
)
```

---

### `@track_llm_cost()`

Decorator for completion and chat calls.

```python
@track_llm_cost(
    model       = "gpt-4o",        # required — model name string
    name        = "my_step",       # optional — human label in output
    on_complete = my_callback,     # optional — function(trace: Trace)
    session     = my_session,      # optional — use a custom session
)
def my_function():
    ...
```

---

### `@track_embedding_cost()`

Decorator for embedding calls. Tracks tokens and cost under the embedding
budget separately from completions.

```python
@track_embedding_cost(
    model       = "text-embedding-3-small",
    name        = "doc_indexer",
    on_complete = my_callback,
    session     = my_session,
)
def embed(input):
    ...
```

---

### `get_session()`

Access the live session aggregate from anywhere in your code.

```python
from token_ledger import get_session

session = get_session()
session.total_cost        # float
session.total_tokens      # int
session.cost_by_model     # dict[str, float]
session.get_traces()      # list[Trace]
session.get_summary()     # dict with all aggregated fields
```

---

### `BudgetExceededError`

Raised when total cost crosses the configured budget.

```python
from token_ledger import BudgetExceededError

try:
    generate(messages)
except BudgetExceededError as e:
    print(e.budget)      # float — the configured limit
    print(e.total_cost)  # float — actual spend when limit was crossed
```

---

### `Trace` Object

Every tracked call produces a `Trace`. Access it via `on_complete` callback
or `session.get_traces()`.

```python
trace.id                    # str  — unique call ID
trace.parent_id             # str  — parent call ID (agent trees)
trace.depth                 # int  — nesting depth (0 = root)
trace.name                  # str  — human label
trace.model                 # str  — model name
trace.call_type             # str  — "completion" or "embedding"
trace.timestamp             # str  — ISO 8601
trace.latency_ms            # float

trace.usage.input_tokens    # int
trace.usage.output_tokens   # int
trace.usage.cached_tokens   # int
trace.usage.reasoning_tokens # int  (o1, o3 models)
trace.usage.embedding_tokens # int
trace.usage.is_estimated    # bool — True if tokenizer fallback was used
trace.usage.total_tokens    # int

trace.cost.input_cost       # float
trace.cost.output_cost      # float
trace.cost.cached_cost      # float
trace.cost.reasoning_cost   # float
trace.cost.embedding_cost   # float
trace.cost.retry_cost       # float
trace.cost.total_cost       # float

trace.status                # str  — "success" | "failed"
trace.retry_count           # int
trace.error_type            # str
```

---

## Cost Formula

token-ledger uses the full production billing formula:

```
Total Cost =
    (Input Tokens  × Input Price)
  + (Output Tokens × Output Price)
  + (Cached Tokens × Cache Read Price)   ← prompt cache hits
  + (Reasoning Tokens × Output Price)    ← o1 / o3 thinking tokens
  + (Embedding Tokens × Input Price)     ← embedding calls
  + (Retry Cost)                         ← sum of failed attempt costs
```

This matches how every major provider actually bills — including OpenAI prompt
caching, Anthropic cache reads, and reasoning token surcharges on o1 and o3
models.

---

## Output File Format

Every session produces `{project_name}_token_cost.txt` containing:

```
============================================================
LLM TOKEN COST REPORT — my_app
Generated: 2026-03-28T14:30:00Z
============================================================

SUMMARY
----------------------------------------
Total Calls    : 5
Total Tokens   : 7,820
  Input        : 6,100
  Output       : 1,720
Total Cost     : $0.023410
Embedding Cost : $0.000040
Retry Cost     : $0.000000

Cost by Model:
  gpt-4o                               $0.019800
  text-embedding-3-small               $0.000040
  claude-3-5-sonnet-20241022           $0.003570

CALL LOG
----------------------------------------
[C] orchestrator | in=850  out=120 | $0.003250 @ 2026-03-28T14:22:01Z
  [C] retriever  | in=200  out=800 | $0.009500 @ 2026-03-28T14:22:02Z
  [C] summarizer | in=950  out=250 | $0.007050 @ 2026-03-28T14:22:04Z
[E] doc_indexer  | in=5000 out=0   | $0.000040 [estimated]
[C] final_answer | in=100  out=550 | $0.003570 @ 2026-03-28T14:22:06Z

RAW DATA (JSON)
----------------------------------------
{ ... full JSON ... }
```

`[C]` = Completion call. `[E]` = Embedding call.
`[estimated]` means a tokenizer fallback was used instead of provider-returned counts.

---

## Design Principles

**Zero intrusion.** Your existing LLM call code does not change.
The decorator sits outside your function — it does not touch what goes in
or what comes out.

**Zero hard dependencies.** The core library has no required packages.
tiktoken, LiteLLM, and provider SDKs are all optional extras.

**Never crashes your application.** Every failure inside token-ledger is
caught and logged as a warning. Unknown models return $0.00 cost.
Missing usage blocks fall back to tokenizer estimation.

**Thread-safe and async-safe.** Uses Python `ContextVar` for call tree
tracking, which works correctly across asyncio tasks and threads without
any shared mutable state between concurrent calls.

**Deterministic.** The same input always produces the same token count
and the same cost. token-ledger pins tokenizer selection per model family.

---

## Development

```bash
git clone https://github.com/aleempasha2772/token-ledger.git
cd token-ledger
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Links

- **PyPI:** https://pypi.org/project/token-ledger/
- **GitHub:** https://github.com/aleempasha2772/token-ledger
- **Issues:** https://github.com/aleempasha2772/token-ledger/issues
```