# Working with Llama 3 — Chapter 1
### A Practical Guide for Biology Students

> **No prior Python experience needed!** Each code block is explained line by line.
> All examples use biology contexts you already know.

---

## Table of Contents
1. [What is Llama 3?](#1-what-is-llama-3)
2. [Why Run Llama Locally?](#2-why-run-llama-locally)
3. [Setting Up: Install the Library](#3-setting-up-install-the-library)
4. [Asking Llama a Question](#4-asking-llama-a-question)
5. [Reading the Output](#5-reading-the-output)
6. [Tuning Parameters](#6-tuning-parameters)
7. [Assigning Chat Roles](#7-assigning-chat-roles)
8. [Exercises](#8-exercises)

---

## 1. What is Llama 3?

**Llama 3** is an AI model — a computer program that can understand and generate text.
Think of it like a very well-read lab assistant that has read the equivalent of **2,000 Wikipedias**.

It was developed by **Meta** (the company behind Facebook) and is **open-source** — meaning it is free to use and modify.

### What can it do?

| Task | Biology Example |
|------|----------------|
| **Summarization** | Summarize a 20-page paper on CRISPR into 5 bullet points |
| **Data analysis** | Interpret results from a PCR experiment |
| **Coding assistant** | Help write a Python script to process DNA sequences |

> **Analogy:** Llama is like a universal lab notebook that you can *talk to*. You ask it a question (called a **prompt**), and it writes back a response.

---

## 2. Why Run Llama Locally?

"Locally" means running the AI **on your own computer**, not sending your data to an external server.

### Why this matters for biologists:

| Reason | What it means for you |
|--------|----------------------|
| **Privacy & Security** | Patient data, unpublished sequences, or proprietary experimental results never leave your machine |
| **Cost efficiency** | No API fees — once set up, unlimited use for free |
| **Customization** | You can fine-tune the model on your specific domain (e.g., genomics literature) |

> **Real example:** A research lab using Llama to predict protein maintenance needs for industrial biotechnology — all running locally to protect intellectual property.

---

## 3. Setting Up: Install the Library

We use a Python library called **`llama-cpp-python`**. Think of a library as a ready-made toolbox you download once.

### Step 1 — Install the library

Open your terminal (or Anaconda Prompt) and type:

```bash
pip install llama-cpp-python
```

> **What is `pip`?** It is Python's package manager — like an app store for Python tools. `install` tells it to download and install the package.

### Step 2 — Download the model file

Llama 3 models are stored in a special format called **GGUF** (a compressed file format for AI models).

You need to download a `.gguf` file and save it somewhere on your computer, for example:
```
/home/yourname/models/llama3.gguf
```

> **Analogy:** The `.gguf` file is like a brain in a box. The Python library is the body that runs it.

---

## 4. Asking Llama a Question

### The 3 steps — explained

```python
# STEP 1: Import the Llama "tool" from the library
from llama_cpp import Llama
```
> `from ... import ...` means: "go into the toolbox called `llama_cpp` and take out the tool called `Llama`."

```python
# STEP 2: Load the model (like opening a program)
llm = Llama(model_path="path/to/model.gguf")
```
> `llm` is just a variable name (short for "large language model") — you could call it anything.
> `model_path=` tells Python where your downloaded model file is located.

```python
# STEP 3: Ask it a question
output = llm("What are the mechanisms of antibiotic resistance?")
```
> You pass your question as a text string inside `llm(...)`.
> The answer is stored in the variable `output`.

### Full example (biology context)

```python
from llama_cpp import Llama

# Load the model
llm = Llama(model_path="path/to/model.gguf")

# Ask a biology question
output = llm("Explain the central dogma of molecular biology in simple terms.")
```

### How it works under the hood

```
Your question (prompt)  -->  [ Llama 3 model ]  -->  Response
```

The model receives your text, processes it (using patterns learned during training), and returns a text response.

---

## 5. Reading the Output

The response is **not** just plain text — it is a **dictionary** (a structured data format in Python, like a labeled set of drawers).

### What the raw output looks like

```python
print(output)
```

```
{
  'id': 'cmpl-af88304f-97b0-49f5-ba20-db87f86c4068',
  'object': 'text_completion',
  'created': 1715222298,
  'model': './Llama3-gguf-unsloth.Q4_K_M.gguf',
  'choices': [
    {'text': 'The central dogma describes the flow of genetic information...'}
  ]
}
```

> **What is a dictionary?** In Python, a dictionary stores data as **key: value** pairs.
> Think of it like a lab report form — each field (key) has a value.
> Example: `'object': 'text_completion'` → the field "object" contains the value "text_completion".

### Extracting just the text answer

You only want the actual answer, not all the metadata. Here is how to navigate to it:

```python
# Access: output → "choices" → first item [0] → "text"
answer = output["choices"][0]["text"]
print(answer)
```

**Step by step:**

| Code | Meaning |
|------|---------|
| `output["choices"]` | Go to the "choices" drawer |
| `output["choices"][0]` | Take the first item (index 0 = first in Python) |
| `output["choices"][0]["text"]` | Get the "text" field from that item |

> **Why `[0]`?** Python starts counting at 0, not 1. So the first item is always `[0]`.

### Complete working example

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

output = llm("What is the role of mitochondria in cellular respiration?")

# Extract clean text
answer = output["choices"][0]["text"]
print(answer)
```

---

## 6. Tuning Parameters

Parameters are **dials** that control how Llama generates its response.

> **Analogy:** Like adjusting a PCR machine — you can change temperature cycles to get different results. Here, you adjust parameters to control the *style* of the text.

### The 4 main parameters

| Parameter | Controls | Range |
|-----------|----------|-------|
| `temperature` | Randomness / creativity | 0 to 1 |
| `top_k` | How many word choices to consider | Integer (e.g., 1–50) |
| `top_p` | Word selection based on confidence | 0 to 1 |
| `max_tokens` | Maximum length of the response | Integer (e.g., 50, 500) |

---

### Parameter 1: `temperature`

Controls how **creative vs. precise** the response is.

```
Low (≈ 0)  →  Factual, predictable, concise
High (≈ 1) →  Creative, varied, expressive
```

**Biology example:**

```python
# Factual report style (low temperature)
output_factual = llm(
    "Describe CRISPR-Cas9.",
    temperature=0.1
)
# Output: "CRISPR-Cas9 is a genome editing tool that uses a guide RNA
#           to direct the Cas9 endonuclease to a specific DNA sequence."

# Engaging presentation style (high temperature)
output_creative = llm(
    "Describe CRISPR-Cas9.",
    temperature=0.9
)
# Output: "Imagine molecular scissors guided by a GPS — that's CRISPR-Cas9,
#           revolutionizing how we rewrite the book of life."
```

> **When to use low temperature:** Lab reports, literature summaries, factual Q&A.
> **When to use high temperature:** Grant proposal introductions, science communication, outreach material.

---

### Parameter 2: `top_k`

Limits **how many of the most likely words** Llama can choose from at each step.

```
Low k (e.g., 1)   →  Only the single most likely word → very predictable
High k (e.g., 50) →  Many options → more diverse phrasing
```

**Biology example:**

```python
# Very predictable (top_k=1)
output_plain = llm(
    "Describe DNA replication.",
    top_k=1
)
# Output: "DNA replication is the process by which DNA is copied in a cell."

# More varied phrasing (top_k=50)
output_varied = llm(
    "Describe DNA replication.",
    top_k=50
)
# Output: "During cell division, the double helix unwinds, and each strand
#           serves as a template for a new complementary strand..."
```

---

### Parameter 3: `top_p`

Controls word selection based on **cumulative probability** (confidence).

```
High top_p (≈ 1)  →  More varied and diverse vocabulary
Low top_p  (≈ 0)  →  Sticks to the most confident, common words
```

> **Analogy:** `top_p=0.9` means: "keep adding word choices until the total probability reaches 90%, then stop." This balances variety with confidence.

---

### Parameter 4: `max_tokens`

Limits the **length** of the response. One token ≈ one word (roughly).

```
max_tokens=20   →  Short summary
max_tokens=500  →  Detailed explanation
```

**Biology example:**

```python
# Short abstract-style answer
output_short = llm(
    "What is apoptosis?",
    max_tokens=30
)
# Output: "Apoptosis is programmed cell death, essential for development
#           and tissue homeostasis."

# Full explanation
output_long = llm(
    "What is apoptosis?",
    max_tokens=300
)
# Output: "Apoptosis, or programmed cell death, is a regulated process
#           crucial for multicellular organisms. It involves two main
#           pathways: the intrinsic (mitochondrial) and extrinsic (death
#           receptor) pathways. Caspases are the key executioner proteins..."
```

---

### Combining all parameters

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

# Precise, concise scientific answer
output_concise = llm(
    "Describe the structure of a eukaryotic cell.",
    temperature=0.2,   # factual
    top_k=1,           # predictable word choice
    top_p=0.4,         # stick to confident words
    max_tokens=50      # short answer
)

# Rich, detailed explanation for teaching
output_detailed = llm(
    "Describe the structure of a eukaryotic cell.",
    temperature=0.8,   # creative
    top_k=10,          # more word variety
    top_p=0.9,         # broader vocabulary
    max_tokens=300     # longer answer
)
```

---

## 7. Assigning Chat Roles

Instead of a single question, you can set up a **conversation with roles** — like instructing a lab assistant before asking them a question.

### The two roles

| Role | Who it represents | What it does |
|------|------------------|--------------|
| **`system`** | The AI's "job description" | Sets personality, tone, expertise |
| **`user`** | You (the researcher) | Your actual question |

> **Analogy:** The `system` role is like the "Methods" section of a paper — it sets the context. The `user` role is the actual experiment you're running.

---

### Step-by-step: Building a biology assistant

#### Step 1 — Define the system message

```python
# Tell Llama what role to play
system_message = "You are a molecular biology expert who explains concepts
                  clearly to undergraduate students."
```

#### Step 2 — Define your question (user message)

```python
# Your actual question
user_message = "What is the difference between transcription and translation?"
```

#### Step 3 — Build the message list

```python
# A list containing both messages, each labeled with their role
message_list = [
    {"role": "system",  "content": system_message},
    {"role": "user",    "content": user_message}
]
```

> **What is a list `[...]`?** An ordered collection of items. Here each item is a dictionary `{...}` with a `"role"` and `"content"`.

#### Step 4 — Send and receive

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

system_message = "You are a molecular biology expert who explains concepts clearly to undergraduate students."
user_message   = "What is the difference between transcription and translation?"

message_list = [
    {"role": "system", "content": system_message},
    {"role": "user",   "content": user_message}
]

# Use create_chat_completion() instead of llm() directly
response = llm.create_chat_completion(messages=message_list)

# Extract the text answer
answer = response["choices"][0]["message"]["content"]
print(answer)
```

> **Note the difference when extracting the answer:**
> - Simple call: `output["choices"][0]["text"]`
> - Chat completion: `response["choices"][0]["message"]["content"]`

---

### The `assistant` role (reading the response)

When you look at `response["choices"][0]`, you will see the model responds with role `"assistant"`:

```
{
  'index': 0,
  'message': {
    'role': 'assistant',
    'content': 'Transcription is the process of copying DNA into mRNA in the nucleus...'
  },
  'finish_reason': 'length'
}
```

The `"assistant"` role is simply Llama's label for its own responses.

---

## 8. Exercises

### Exercise 1 — Your first question

**Task:** Ask Llama to explain what a ribosome is.

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

# Fill in the blank:
output = llm("___________________________")

# Print only the text answer:
print(output["choices"][0]["text"])
```

**Expected output:** A plain text explanation of ribosomes.

---

### Exercise 2 — Compare temperatures

**Task:** Run the same biology prompt with two different temperatures. Compare the outputs.

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

prompt = "Explain how vaccines work."

# Version 1: factual (for a lab report)
output_factual = llm(prompt, temperature=0.1, max_tokens=100)

# Version 2: engaging (for a public talk)
output_engaging = llm(prompt, temperature=0.9, max_tokens=100)

print("--- FACTUAL ---")
print(output_factual["choices"][0]["text"])

print("\n--- ENGAGING ---")
print(output_engaging["choices"][0]["text"])
```

**Questions to reflect on:**
- Which version would you use in a scientific paper? Why?
- Which version would work better for communicating with the general public?

---

### Exercise 3 — Build a biology chatbot

**Task:** Create a Llama assistant that acts as a **genetics tutor** and answer a question about Mendel's laws.

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

# STEP 1: Define the role
system_message = "You are a genetics tutor helping first-year biology students understand inheritance patterns."

# STEP 2: Write your question
user_message = "Can you explain Mendel's Law of Segregation with a simple example?"

# STEP 3: Build the message list
message_list = [
    {"role": "system", "content": system_message},
    {"role": "user",   "content": user_message}
]

# STEP 4: Get the response
response = llm.create_chat_completion(messages=message_list)

# STEP 5: Print the answer
print(response["choices"][0]["message"]["content"])
```

---

### Exercise 4 — Challenge: Parameter exploration

**Task:** You are writing a brief description of SARS-CoV-2 spike protein for two different audiences. Use parameters to control the output.

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

prompt = "Describe the SARS-CoV-2 spike protein."

# For a scientific journal (complete the parameters)
output_scientific = llm(
    prompt,
    temperature=___,   # should be low (factual)
    top_k=___,         # should be low (predictable)
    max_tokens=___     # should be moderate (100–150)
)

# For a health magazine (complete the parameters)
output_magazine = llm(
    prompt,
    temperature=___,   # should be high (engaging)
    top_k=___,         # should be higher (diverse)
    max_tokens=___     # should be moderate
)

print(output_scientific["choices"][0]["text"])
print(output_magazine["choices"][0]["text"])
```

**Fill in the blanks** and explain your choices.

---

## Quick Reference Card

```python
# ── BASIC USAGE ──────────────────────────────────────────
from llama_cpp import Llama
llm    = Llama(model_path="path/to/model.gguf")
output = llm("Your question here")
text   = output["choices"][0]["text"]          # extract answer

# ── WITH PARAMETERS ──────────────────────────────────────
output = llm(
    "Your question",
    temperature = 0.2,    # 0=factual, 1=creative
    top_k       = 10,     # low=predictable, high=diverse
    top_p       = 0.5,    # low=safe words, high=varied words
    max_tokens  = 100     # max length of response
)

# ── CHAT WITH ROLES ──────────────────────────────────────
message_list = [
    {"role": "system", "content": "You are a biology expert."},
    {"role": "user",   "content": "Your question here"}
]
response = llm.create_chat_completion(messages=message_list)
text     = response["choices"][0]["message"]["content"]   # extract answer
```

---

*Based on: "Working with Llama 3" — DataCamp, by Imtihan Ahmed (Machine Learning Engineer)*
