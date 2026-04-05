# Llama API Reference
### Classes & Methods in `llama-cpp-python` — A Guide for Biology Students

> This document is a complete reference for every class and method covered in the course.
> Each entry includes: what it does, all parameters, return value, and a biology example.

---

## Table of Contents

1. [Import Statement](#1-import-statement)
2. [Class: `Llama`](#2-class-llama)
   - [`__init__` — Create the model](#llama__init__)
   - [`__call__` — Ask a question (simple)](#llama__call__)
   - [`create_chat_completion` — Ask with roles](#llamacreate_chat_completion)
3. [Class: `Conversation` (custom)](#3-class-conversation)
   - [`__init__` — Start a session](#conversation__init__)
   - [`create_completion` — Send a message](#conversationcreate_completion)
4. [Response Objects (What gets returned)](#4-response-objects)
   - [Text completion response](#text-completion-response)
   - [Chat completion response](#chat-completion-response)
5. [Parameters Cheat Sheet](#5-parameters-cheat-sheet)

---

## 1. Import Statement

```python
from llama_cpp import Llama
```

| Part | Meaning |
|------|---------|
| `llama_cpp` | The installed library (package name) |
| `Llama` | The class you load from it |

> **Analogy:** Like pulling a specific instrument out of a shared equipment cabinet. `from llama_cpp` = open the cabinet; `import Llama` = take out the sequencer.

---

## 2. Class: `Llama`

The `Llama` class is the main object you interact with. It loads the model file and exposes methods to generate text.

---

### `Llama.__init__`

**Creates (loads) the Llama model.**

```python
llm = Llama(model_path="path/to/model.gguf")
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_path` | `str` | **Yes** | Path to the downloaded `.gguf` model file |
| `n_ctx` | `int` | No | Context window size in tokens (default: 512). Increase for longer conversations. |
| `n_gpu_layers` | `int` | No | Number of model layers to offload to GPU (default: 0 = CPU only) |
| `verbose` | `bool` | No | Print loading logs to console (default: `True`). Set `False` to silence. |

#### Example

```python
from llama_cpp import Llama

# Minimal — just load the model
llm = Llama(model_path="/models/llama3.gguf")

# With options — larger context, silent loading
llm = Llama(
    model_path="/models/llama3.gguf",
    n_ctx=2048,       # allow longer conversations
    verbose=False     # suppress loading messages
)
```

> **What is `n_ctx`?** It controls how many tokens (≈ words) the model can "see" at once — both your input and its output combined. For multi-turn conversations, increase this.

---

### `Llama.__call__`

**Sends a text prompt and gets a response. The simplest way to use Llama.**

Called by writing `llm("your prompt")` — Python calls `__call__` automatically when you use parentheses on an object.

```python
output = llm("Your prompt here")
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | **required** | The text question or instruction |
| `max_tokens` | `int` | `128` | Maximum number of tokens to generate (≈ max words in response) |
| `temperature` | `float` | `0.8` | Creativity / randomness. `0` = factual, `1` = creative |
| `top_k` | `int` | `40` | Number of top word choices the model considers at each step |
| `top_p` | `float` | `0.95` | Cumulative probability cutoff for word selection |
| `stop` | `list[str]` | `[]` | List of strings that cause generation to stop immediately |
| `echo` | `bool` | `False` | If `True`, includes your prompt in the output text |

#### Returns

A **dictionary** (see [Text completion response](#text-completion-response)).

#### Examples

```python
# Basic call
output = llm("What is the function of the Golgi apparatus?")

# With parameters — factual, concise
output = llm(
    "Describe the steps of mitosis.",
    temperature=0.1,    # factual
    top_k=10,           # predictable word choice
    max_tokens=200      # moderate length
)

# With stop word — stop before generating a second question
output = llm(
    "Q: What is DNA?\nA:",
    stop=["Q:"],
    max_tokens=100
)

# Extract the answer text
text = output["choices"][0]["text"]
print(text)
```

#### Parameter guide for biology use cases

| Task | `temperature` | `top_k` | `max_tokens` |
|------|--------------|---------|-------------|
| Lab report / factual summary | `0.1` | `5–10` | `100–200` |
| Exam study guide | `0.3` | `20` | `200–400` |
| Science communication / outreach | `0.8` | `40` | `150–300` |
| Creative analogy / metaphor | `0.9` | `50` | `100–200` |

---

### `Llama.create_chat_completion`

**Sends a structured conversation (with roles) and gets a response.**
Use this when you want to assign the model a specific persona or maintain a multi-turn exchange.

```python
response = llm.create_chat_completion(messages=message_list)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[dict]` | **required** | List of message dictionaries, each with `"role"` and `"content"` |
| `response_format` | `str` or `dict` | `None` | Set to `"json_object"` for JSON output, or a dict with a schema |
| `max_tokens` | `int` | `None` | Maximum tokens to generate |
| `temperature` | `float` | `0.8` | Creativity / randomness |
| `top_k` | `int` | `40` | Top word choices considered |
| `top_p` | `float` | `0.95` | Cumulative probability cutoff |
| `stop` | `list[str]` | `[]` | Stop generation at these strings |

#### The `messages` list

Each message is a dictionary with exactly two keys:

```python
{"role": "...", "content": "..."}
```

| `role` value | Who it represents | When to use |
|-------------|------------------|-------------|
| `"system"` | The AI's persona / job description | First message — sets context |
| `"user"` | You (the researcher/student) | Your questions |
| `"assistant"` | Llama's previous replies | Only when loading prior history |

#### Returns

A **dictionary** (see [Chat completion response](#chat-completion-response)).

#### Examples

```python
from llama_cpp import Llama

llm = Llama(model_path="/models/llama3.gguf")

# ── Basic chat with roles ──────────────────────────────────────────────────
message_list = [
    {"role": "system",  "content": "You are a cell biology expert for undergraduates."},
    {"role": "user",    "content": "What is the difference between mitosis and meiosis?"}
]

response = llm.create_chat_completion(messages=message_list)

# Extract the text
answer = response["choices"][0]["message"]["content"]
print(answer)


# ── With JSON output ───────────────────────────────────────────────────────
message_list = [
    {"role": "system", "content": "You are a taxonomy assistant. Respond with JSON only."},
    {"role": "user",   "content": "Classify Homo sapiens."}
]

response = llm.create_chat_completion(
    messages=message_list,
    response_format="json_object"   # forces JSON output
)

import json
data = json.loads(response["choices"][0]["message"]["content"])
print(data["kingdom"])   # → Animalia


# ── With a JSON schema ─────────────────────────────────────────────────────
schema = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "species":   {"type": "string"},
            "kingdom":   {"type": "string"},
            "cell_type": {"type": "string"}
        },
        "required": ["species", "kingdom", "cell_type"]
    }
}

response = llm.create_chat_completion(
    messages=message_list,
    response_format=schema
)
```

---

## 3. Class: `Conversation` (custom)

This class is **not** built into `llama_cpp` — it is a helper class you define yourself (taught in Chapter 2). It wraps `create_chat_completion` and automatically maintains conversation history across multiple turns.

```python
class Conversation:
    def __init__(self, llm, system_prompt='', history=[]):
        ...
    def create_completion(self, user_prompt=''):
        ...
```

---

### `Conversation.__init__`

**Creates a new conversation session.**

```python
conv = Conversation(llm, system_prompt="You are a genetics tutor.")
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `Llama` | **required** | A loaded `Llama` model instance |
| `system_prompt` | `str` | `''` | The persona / role for the AI (optional but recommended) |
| `history` | `list[dict]` | `[]` | Pre-existing messages to load (optional, for resuming a session) |

#### What it does internally

```python
def __init__(self, llm, system_prompt='', history=[]):
    self.llm           = llm
    self.system_prompt = system_prompt
    self.history       = [{"role": "system", "content": system_prompt}] + history
    #                     ↑ always starts with the system message
```

#### Example

```python
from llama_cpp import Llama

llm = Llama(model_path="/models/llama3.gguf")

# Start a fresh session
tutor = Conversation(llm, system_prompt="You are a molecular biology tutor.")

# Resume a previous session with pre-loaded history
prior = [
    {"role": "user",      "content": "What is PCR?"},
    {"role": "assistant", "content": "PCR stands for Polymerase Chain Reaction..."}
]
tutor = Conversation(llm, system_prompt="You are a molecular biology tutor.", history=prior)
```

---

### `Conversation.create_completion`

**Sends a message and returns the reply. Updates history automatically.**

```python
answer = conv.create_completion("Your question here")
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_prompt` | `str` | `''` | Your message / question for this turn |

#### Returns

`str` — the model's reply text (plain string, not a dictionary).

#### What it does internally

```python
def create_completion(self, user_prompt=''):
    # 1. Add your message to history
    self.history.append({"role": "user", "content": user_prompt})

    # 2. Send the full history to Llama
    output = self.llm.create_chat_completion(messages=self.history)

    # 3. Extract the reply
    reply = output["choices"][0]["message"]   # {"role": "assistant", "content": "..."}

    # 4. Save the reply to history (so it's remembered next turn)
    self.history.append(reply)

    # 5. Return just the text
    return reply["content"]
```

#### Example

```python
tutor = Conversation(llm, system_prompt="You are a virology tutor.")

r1 = tutor.create_completion("What is a retrovirus?")
print(r1)

r2 = tutor.create_completion("How does HIV integrate into the host genome?")
print(r2)   # Llama knows "HIV" refers to the retrovirus from r1

r3 = tutor.create_completion("What enzyme does it use for that?")
print(r3)   # Llama knows "that" = integration from r2
```

#### Viewing the full history

```python
# After a conversation, you can inspect all messages
for message in tutor.history:
    print(f"[{message['role'].upper()}]: {message['content'][:80]}...")
```

---

## 4. Response Objects

### Text completion response

Returned by `llm("prompt")` (`__call__`).

```python
{
    "id":      "cmpl-af88304f-97b0-49f5-ba20-db87f86c4068",  # unique ID
    "object":  "text_completion",                             # type of response
    "created": 1715222298,                                    # Unix timestamp
    "model":   "./llama3.gguf",                               # model used
    "choices": [
        {
            "text":          "The mitochondria is the powerhouse...",  # ← the answer
            "index":         0,
            "finish_reason": "stop"    # "stop"=natural end, "length"=hit max_tokens
        }
    ]
}
```

**How to extract the answer:**

```python
text = output["choices"][0]["text"]
```

---

### Chat completion response

Returned by `llm.create_chat_completion(messages=...)`.

```python
{
    "id":      "chatcmpl-abc123",
    "object":  "chat.completion",
    "created": 1715222298,
    "model":   "./llama3.gguf",
    "choices": [
        {
            "index": 0,
            "message": {
                "role":    "assistant",                        # always "assistant"
                "content": "Transcription is the process..."  # ← the answer
            },
            "finish_reason": "stop"
        }
    ]
}
```

**How to extract the answer:**

```python
text = response["choices"][0]["message"]["content"]
```

---

### Side-by-side comparison

| | `__call__` | `create_chat_completion` |
|--|------------|--------------------------|
| **Input** | Plain string prompt | List of role/content dicts |
| **Extract answer** | `output["choices"][0]["text"]` | `response["choices"][0]["message"]["content"]` |
| **Supports roles** | No | Yes |
| **Supports JSON output** | No | Yes (`response_format`) |
| **Use when** | Simple one-off questions | Role-based chat, JSON, pipelines |

---

## 5. Parameters Cheat Sheet

### `temperature` — creativity dial

```
0.0 ──────────────────────────── 1.0
Factual / Deterministic      Creative / Random
```

| Value | Best for |
|-------|---------|
| `0.0–0.2` | Lab reports, factual Q&A, literature summaries |
| `0.3–0.5` | Study guides, balanced explanations |
| `0.6–0.8` | Teaching materials, patient-facing communication |
| `0.9–1.0` | Science outreach, creative analogies, grant introductions |

---

### `top_k` — word-choice breadth

```
top_k=1       → Only the single most likely next word (very predictable)
top_k=10      → Top 10 choices (more natural)
top_k=40–50   → Many choices (diverse phrasing)
```

---

### `top_p` — confidence cutoff

```
top_p=0.1  → Only the most confident words
top_p=0.9  → Broad vocabulary, varied phrasing
top_p=1.0  → No cutoff applied
```

> Use `top_p` and `top_k` together: `top_k` sets the max pool size; `top_p` narrows it by probability.

---

### `max_tokens` — response length

| Value | Approximate output |
|-------|--------------------|
| `20–50` | One sentence / definition |
| `100–150` | Short paragraph |
| `200–400` | Full explanation |
| `500+` | Multi-paragraph essay |

> One token ≈ one word (roughly). Some long words or technical terms count as multiple tokens.

---

### `stop` — stop words

```python
stop=["Q:"]          # Stop before generating a new question block
stop=["###"]         # Stop before a new section header
stop=["Q:", "---"]   # Multiple stop words — stops at whichever comes first
```

> The stop string is **not included** in the output.

---

### `response_format` — output type

```python
# Plain text (default — no need to specify)
response_format = None

# Force JSON output
response_format = "json_object"

# Force JSON with a specific structure (schema)
response_format = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "field_name": {"type": "string"},   # text field
            "count":      {"type": "integer"},  # whole number field
            "score":      {"type": "number"},   # decimal number field
            "tags":       {"type": "array",     # list field
                           "items": {"type": "string"}}
        },
        "required": ["field_name"]   # fields that must always appear
    }
}
```

---

## Complete usage map

```
llama_cpp
└── Llama (class)
    ├── __init__(model_path, n_ctx, n_gpu_layers, verbose)
    │       → loads the model, returns Llama instance
    │
    ├── __call__(prompt, max_tokens, temperature, top_k, top_p, stop, echo)
    │       → returns text_completion dict
    │       → extract: output["choices"][0]["text"]
    │
    └── create_chat_completion(messages, response_format, max_tokens,
    │                          temperature, top_k, top_p, stop)
    │       → returns chat_completion dict
    │       → extract: response["choices"][0]["message"]["content"]
    │
    └── [custom] Conversation (class built on top of Llama)
        ├── __init__(llm, system_prompt, history)
        │       → initializes history with system message
        │
        └── create_completion(user_prompt)
                → appends to history, calls create_chat_completion
                → returns str (the reply text directly)
```

---

*Based on: "Working with Llama 3" — DataCamp, by Imtihan Ahmed (Machine Learning Engineer)*
*Library: `llama-cpp-python` — Python bindings for `llama.cpp`*
