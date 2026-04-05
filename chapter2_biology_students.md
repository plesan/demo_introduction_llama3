# Working with Llama 3 — Chapter 2
### A Practical Guide for Biology Students

> **No prior Python experience needed!** Each code block is explained line by line.
> All examples use biology contexts you already know.

---

## Table of Contents
1. [Writing Better Prompts](#1-writing-better-prompts)
2. [Zero-Shot Prompting](#2-zero-shot-prompting)
3. [Few-Shot Prompting](#3-few-shot-prompting)
4. [Stop Words](#4-stop-words)
5. [Generating Structured JSON Output](#5-generating-structured-json-output)
6. [Defining a JSON Schema](#6-defining-a-json-schema)
7. [Building Multi-Turn Conversations](#7-building-multi-turn-conversations)
8. [Exercises](#8-exercises)

---

## 1. Writing Better Prompts

A **prompt** is the question or instruction you give to Llama. The quality of the answer depends heavily on how well you phrase the question.

> **Analogy:** Think of prompting like writing a lab protocol. A vague protocol ("mix the chemicals") gives unpredictable results. A precise protocol ("add 2 µL of enzyme to 50 µL of buffer at 37°C for 30 min") gives reliable, reproducible results.

---

### Vague vs. Specific Prompts

| Vague Prompt | Specific Prompt |
|-------------|-----------------|
| "Tell me about PCR." | "Summarize the key steps of PCR, focusing on the role of DNA polymerase and primer annealing." |
| "What are proteins?" | "Explain the four levels of protein structure, with one example of each." |

The vague prompt may give a broad, unfocused answer. The specific prompt guides the model toward exactly what you need.

---

### The 4 Components of an Effective Prompt

| Component | What it means | Biology example |
|-----------|--------------|-----------------|
| **Precision** | Be specific about what you want | "List 3 enzymes used in DNA cloning" not "tell me about enzymes" |
| **Avoid ambiguity** | Use clear terms, not jargon-free phrases | Write "eukaryotic cells" not "complex cells" |
| **Keywords** | Include domain-specific words | "restriction enzyme", "transcription factor", "ligand binding" |
| **Action words** | Use verbs: summarize, compare, list, explain | "Compare mitosis and meiosis in a table" |

---

### Example: Improving a prompt

```python
# Vague prompt — unpredictable result
vague_output = llm("Tell me about CRISPR.")

# Improved prompt — precise and actionable
better_output = llm(
    "Summarize how CRISPR-Cas9 edits genes, "
    "focusing on the roles of the guide RNA and Cas9 protein. "
    "Write 3–4 sentences suitable for an undergraduate student."
)
```

> **What changed?**
> - Added **action word**: "Summarize"
> - Added **keywords**: "guide RNA", "Cas9 protein"
> - Specified **audience and length**: "3–4 sentences for undergrad"

---

## 2. Zero-Shot Prompting

**Zero-shot** means you give the model an instruction **without any examples** — just a clear directive.

> **Analogy:** Like telling a new lab assistant: "Pipette 10 µL into each well." You don't demonstrate it first; you just give a precise instruction.

---

### Basic zero-shot prompt

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

# A clear instruction with no example needed
output = llm("List 3 key differences between prokaryotic and eukaryotic cells.")

print(output["choices"][0]["text"])
```

---

### Refined zero-shot: using labels

You can structure your prompt using **labels** (like `INSTRUCTION:`, `QUESTION:`, `ANSWER:`) to guide the model more precisely.

```python
# Build a labeled prompt as a multi-line string
text = """
INSTRUCTION: Write concisely in 2–3 sentences covering only key points.
QUESTION: What is the role of ATP synthase in cellular respiration?
ANSWER:
"""
```

> **What are the labels doing?**
> - `INSTRUCTION:` — tells Llama *how* to respond (concise, 2–3 sentences)
> - `QUESTION:` — contains your actual question
> - `ANSWER:` — leaves a blank for Llama to fill in, like a fill-in-the-blank form

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

text = """
INSTRUCTION: Write concisely in 2–3 sentences covering only key points.
QUESTION: What is the role of ATP synthase in cellular respiration?
ANSWER:
"""

output = llm(text, max_tokens=100)
print(output["choices"][0]["text"])
```

**Expected output:**
```
ATP synthase is an enzyme in the inner mitochondrial membrane that
synthesizes ATP from ADP and inorganic phosphate, driven by the
proton gradient established during the electron transport chain.
```

---

### Why use labels?

Without labels, the model might:
- Answer in an essay format when you wanted bullet points
- Include extra commentary you didn't ask for
- Answer a different interpretation of your question

Labels act like the **structure of a scientific abstract** — each section has a clear role.

---

## 3. Few-Shot Prompting

**Few-shot prompting** means you provide **2–3 examples** of the pattern you want, then let the model continue the pattern for a new case.

> **Analogy:** Like training a student to classify organisms. You show them: "Organism A = Eukaryote, Organism B = Prokaryote." Then you ask: "What about Organism C?" The student applies the pattern.

---

### Building a few-shot prompt

You write a block of text that contains completed examples, then leave the last entry blank for Llama to fill in.

```python
# A template with 2 completed examples and 1 blank entry
text = """
Species: Homo sapiens
Kingdom: Animalia
Cell Type: Eukaryote

Species: Escherichia coli
Kingdom: Bacteria
Cell Type: Prokaryote

Species: Saccharomyces cerevisiae
Kingdom: Fungi
Cell Type:
"""
```

> **What is happening here?**
> - The first two "blocks" are complete examples showing the format
> - The third block has `Cell Type:` left blank
> - Llama will recognize the pattern and predict: `Eukaryote`

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

text = """
Species: Homo sapiens
Kingdom: Animalia
Cell Type: Eukaryote

Species: Escherichia coli
Kingdom: Bacteria
Cell Type: Prokaryote

Species: Saccharomyces cerevisiae
Kingdom: Fungi
Cell Type:
"""

# f-string combines a fixed instruction with the template
output = llm(f"Continue the entries: {text}", max_tokens=20)
print(output["choices"][0]["text"])
# → Eukaryote
```

> **What is `f"...{text}"`?** An **f-string** (formatted string) lets you embed a variable inside a string. `f"Continue the entries: {text}"` inserts the value of `text` into the instruction sentence.

---

### More complex few-shot example

You can use few-shot prompting for more structured tasks, like classifying experimental observations:

```python
text = """
Observation: Cells show cytoplasmic blebbing and nuclear fragmentation.
Process: Apoptosis

Observation: Cells swell and membrane ruptures, releasing contents.
Process: Necrosis

Observation: Cells reduce in size, autophagosome formation observed.
Process:
"""

output = llm(f"Identify the biological process: {text}", max_tokens=10)
print(output["choices"][0]["text"])
# → Autophagy
```

---

### Zero-shot vs. Few-shot — when to use which?

| Situation | Use |
|-----------|-----|
| The task is straightforward | Zero-shot (simpler, faster) |
| The output format must be very specific | Few-shot (demonstrate the pattern) |
| The model keeps giving wrong format | Few-shot (show it what you want) |
| You have domain-specific classification tasks | Few-shot (teach it your categories) |

---

## 4. Stop Words

By default, Llama keeps generating text until it hits `max_tokens`. Sometimes you want it to **stop at a specific word or symbol** — for example, before it starts generating a second question.

> **Analogy:** A stop codon (UAA, UAG, UGA) tells a ribosome: "stop translating here." The `stop` parameter does the same for Llama — you define which token signals "end of response."

---

### The problem without stop words

When using labeled prompts, the model sometimes keeps going and generates a second `Q:` block after answering the first:

```
ANSWER: DNA replication begins with helicase unwinding the double helix...

Q: What is the role of primase?
ANSWER: Primase synthesizes...
```

You only wanted the first answer!

---

### Using the `stop` parameter

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

text = """
Q: What enzyme unwinds the DNA double helix during replication?
A: Helicase unwinds the double helix by breaking hydrogen bonds between base pairs.

Q: What is the role of DNA ligase?
A:
"""

# stop=["Q:"] tells Llama: stop as soon as you generate "Q:"
output = llm(text, stop=["Q:"], max_tokens=100)
print(output["choices"][0]["text"])
```

**Expected output (stops before generating a new question):**
```
DNA ligase seals the nicks between Okazaki fragments on the lagging strand,
joining them into a continuous DNA strand.
```

---

### How `stop` works

```python
stop=["Q:"]
```

> - `stop` takes a **list** of strings (you can have more than one stop word)
> - When Llama is about to generate `"Q:"`, it stops and returns everything it has written so far
> - The stop word itself is **not included** in the output

```python
# You can use multiple stop words
output = llm(text, stop=["Q:", "###", "---"])
```

---

## 5. Generating Structured JSON Output

So far, Llama has returned plain text. But for data analysis or programmatic use, you often need **structured data** — like a table or a database record.

**JSON** (JavaScript Object Notation) is a standard format for structured data. Think of it like a Python dictionary.

> **Analogy:** Plain text is like a lab notebook written in prose. JSON is like a structured data spreadsheet — every value is in a named, predictable column.

---

### Requesting JSON output

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

# Define a system prompt that requests JSON
system_message = (
    "You are a biology data assistant. "
    "Always respond with valid JSON only."
)

user_message = (
    "Provide a summary of the BRCA1 gene including: "
    "gene_name, chromosome, function, and associated_diseases."
)

message_list = [
    {"role": "system", "content": system_message},
    {"role": "user",   "content": user_message}
]

# response_format tells Llama to output JSON
output = llm.create_chat_completion(
    messages=message_list,
    response_format="json_object"   # ← this forces JSON output
)

print(output["choices"][0]["message"]["content"])
```

**Expected output:**
```json
{
  "gene_name": "BRCA1",
  "chromosome": "17",
  "function": "Tumor suppressor involved in DNA repair via homologous recombination",
  "associated_diseases": ["Breast cancer", "Ovarian cancer"]
}
```

---

### Why use JSON output?

| Use case | Why JSON helps |
|----------|---------------|
| Storing results in a database | JSON maps directly to table rows |
| Comparing results across samples | Each field is consistently named |
| Feeding results to another program | Easy to parse with Python |
| Building a data pipeline | Predictable structure, no text parsing needed |

---

### Extracting values from the JSON response

```python
import json

# Get the raw JSON string from the response
raw_json = output["choices"][0]["message"]["content"]

# Convert the JSON string into a Python dictionary
data = json.loads(raw_json)

# Access individual fields
print(data["gene_name"])          # → BRCA1
print(data["chromosome"])         # → 17
print(data["associated_diseases"]) # → ['Breast cancer', 'Ovarian cancer']
```

> **What is `json.loads()`?** It converts a JSON-formatted string into a Python dictionary. `loads` = "load string".

---

## 6. Defining a JSON Schema

Sometimes you need to guarantee the **exact structure** of the JSON output — for example, to ensure every response has the same fields for downstream processing.

A **schema** is a blueprint that defines what fields must be present and what type each field should be.

> **Analogy:** Like defining the required columns in a lab database before entering data. You specify: "every entry must have Species (text), Kingdom (text), Cell_Type (text)." No missing fields allowed.

---

### Defining and using a schema

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

# Define the schema — the blueprint for the output
response_format = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "species":   {"type": "string"},
            "kingdom":   {"type": "string"},
            "cell_type": {"type": "string"},
            "habitat":   {"type": "string"}
        },
        "required": ["species", "kingdom", "cell_type"]  # these fields must appear
    }
}
```

> **Breaking down the schema:**
> - `"type": "json_object"` — output must be JSON
> - `"properties"` — lists each field and its data type
> - `{"type": "string"}` — this field must be text
> - `"required"` — these fields **must** be present in every response

```python
message_list = [
    {"role": "system", "content": "You are a taxonomy database assistant. Respond with JSON only."},
    {"role": "user",   "content": "Classify Arabidopsis thaliana."}
]

output = llm.create_chat_completion(
    messages=message_list,
    response_format=response_format   # ← apply the schema
)

print(output["choices"][0]["message"]["content"])
```

**Expected output:**
```json
{
  "species":   "Arabidopsis thaliana",
  "kingdom":   "Plantae",
  "cell_type": "Eukaryote",
  "habitat":   "Temperate regions, commonly used as a model organism"
}
```

---

### Building a biology data pipeline

```python
import json
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

# Schema for classifying multiple organisms
response_format = {
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

organisms = [
    "Homo sapiens",
    "Escherichia coli",
    "Saccharomyces cerevisiae",
    "Arabidopsis thaliana"
]

results = []  # empty list to store results

# Loop over each organism
for organism in organisms:
    message_list = [
        {"role": "system", "content": "You are a taxonomy assistant. Respond with JSON only."},
        {"role": "user",   "content": f"Classify {organism}."}
    ]

    output = llm.create_chat_completion(
        messages=message_list,
        response_format=response_format
    )

    # Parse JSON and add to results list
    data = json.loads(output["choices"][0]["message"]["content"])
    results.append(data)

# Print all results
for r in results:
    print(r)
```

> **What is `for organism in organisms:`?** A **for loop** repeats the indented code once for each item in the list. Each time through, `organism` holds the current item (e.g., "Homo sapiens").

---

## 7. Building Multi-Turn Conversations

So far, each prompt has been independent — the model has no memory of previous questions. For research workflows, you often need a **back-and-forth conversation** where context carries over.

> **Analogy:** Like a supervised lab session. The first question is "What is PCR?" and the follow-up is "What are its main steps?" — the student (Llama) should remember what PCR is when answering the second question, not start from scratch.

---

### The problem with stateless calls

```python
# Call 1 — works fine
output1 = llm("What is CRISPR-Cas9?")

# Call 2 — Llama has NO memory of Call 1
output2 = llm("What are its main limitations?")
# ↑ Llama doesn't know what "its" refers to!
```

Each `llm()` call is independent. To maintain context, you must **pass the entire conversation history** each time.

---

### The `Conversation` class

A **class** is a reusable blueprint for an object. The `Conversation` class manages the conversation history for you automatically.

```python
class Conversation:
    def __init__(self, llm, system_prompt='', history=[]):
        self.llm           = llm            # store the Llama model
        self.system_prompt = system_prompt  # store the role/persona
        self.history = [                    # start with the system message
            {"role": "system", "content": self.system_prompt}
        ] + history                         # add any prior history if provided
```

> **Breaking down `__init__`:**
> - `__init__` is the **constructor** — it runs automatically when you create a new `Conversation` object
> - `self` refers to "this specific conversation instance"
> - `self.history` stores all messages so far as a list of dictionaries

---

### The `create_completion` method

```python
    def create_completion(self, user_prompt=''):
        # Step 1: Add the new user question to history
        self.history.append({"role": "user", "content": user_prompt})

        # Step 2: Send the FULL history to Llama
        output = self.llm.create_chat_completion(messages=self.history)

        # Step 3: Extract the model's reply
        reply = output["choices"][0]["message"]   # {"role": "assistant", "content": "..."}

        # Step 4: Add the reply to history (so it's remembered next time)
        self.history.append(reply)

        # Step 5: Return just the text content
        return reply["content"]
```

> **Why append to `self.history` each time?**
> Because `create_chat_completion()` needs the complete conversation history every time it is called. The class handles this automatically so you don't have to.

---

### Full example: Biology tutoring session

```python
from llama_cpp import Llama

# Full class definition
class Conversation:
    def __init__(self, llm, system_prompt='', history=[]):
        self.llm           = llm
        self.system_prompt = system_prompt
        self.history       = [{"role": "system", "content": self.system_prompt}] + history

    def create_completion(self, user_prompt=''):
        self.history.append({"role": "user", "content": user_prompt})
        output = self.llm.create_chat_completion(messages=self.history)
        reply  = output["choices"][0]["message"]
        self.history.append(reply)
        return reply["content"]


# ── Using the class ───────────────────────────────────────────────────────────

llm = Llama(model_path="path/to/model.gguf")

# Create a conversation with a biology tutor persona
tutor = Conversation(
    llm,
    system_prompt="You are a molecular biology tutor for undergraduate students. "
                  "Give clear, concise answers with examples."
)

# Turn 1: First question
response1 = tutor.create_completion("What is PCR?")
print("TUTOR:", response1)

# Turn 2: Follow-up — the model remembers "PCR" from Turn 1
response2 = tutor.create_completion("What are the three main steps in each cycle?")
print("TUTOR:", response2)

# Turn 3: Deeper follow-up
response3 = tutor.create_completion("Why does the denaturation step need high temperature?")
print("TUTOR:", response3)
```

**What Llama "sees" during Turn 3 (the full history):**
```
[system]:    "You are a molecular biology tutor..."
[user]:      "What is PCR?"
[assistant]: "PCR (Polymerase Chain Reaction) is a technique to amplify..."
[user]:      "What are the three main steps in each cycle?"
[assistant]: "The three steps are: Denaturation (94°C), Annealing, and Extension..."
[user]:      "Why does the denaturation step need high temperature?"
```

> The model always has full context, so "Why does it need high temperature?" is understood as referring to the denaturation step in PCR.

---

### Starting a conversation with prior history

You can also initialize a conversation with pre-existing messages:

```python
# Pre-load some context from a previous session
prior_history = [
    {"role": "user",      "content": "What is gel electrophoresis?"},
    {"role": "assistant", "content": "Gel electrophoresis separates DNA fragments by size..."}
]

# Resume the conversation with that context
tutor = Conversation(
    llm,
    system_prompt="You are a molecular biology tutor.",
    history=prior_history   # ← pre-loaded history
)

# Continue asking — model knows the prior exchange
response = tutor.create_completion("What voltage is typically used?")
print(response)
```

---

## 8. Exercises

### Exercise 1 — Improve a vague prompt

**Task:** Rewrite this vague prompt to make it more precise using the 4 components (precision, no ambiguity, keywords, action words).

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

# Vague prompt — replace this with a better version
vague = "Tell me about mutations."

# Your improved prompt:
better = "___________________________"

output = llm(better, max_tokens=150)
print(output["choices"][0]["text"])
```

**Hint:** What type of mutation? What should the answer cover? How long? For what audience?

---

### Exercise 2 — Few-shot classification

**Task:** Use few-shot prompting to classify a new experimental observation.

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

text = """
Symptom: Patient presents with elongated cells, hemolysis, and pain crises.
Condition: Sickle cell disease

Symptom: Patients lack functional chloride channels; thick mucus in lungs.
Condition: Cystic fibrosis

Symptom: Progressive muscle weakness due to missing dystrophin protein.
Condition:
"""

output = llm(f"Identify the genetic condition: {text}", max_tokens=15)
print(output["choices"][0]["text"])
```

**Expected output:** `Duchenne muscular dystrophy`

---

### Exercise 3 — Stop words in action

**Task:** Use a stop word to make Llama answer only one question and stop before generating a second.

```python
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

text = """
Q: What is the function of the cell membrane?
A: The cell membrane controls what enters and exits the cell, maintaining homeostasis.

Q: What are the two main types of cell transport?
A:
"""

# Fill in the stop parameter
output = llm(text, stop=[___], max_tokens=100)
print(output["choices"][0]["text"])
```

**Fill in the blank:** What stop word should you use?

---

### Exercise 4 — Structured JSON output

**Task:** Ask Llama to return information about a protein in JSON format.

```python
from llama_cpp import Llama
import json

llm = Llama(model_path="path/to/model.gguf")

message_list = [
    {"role": "system", "content": "You are a protein database assistant. Respond with JSON only."},
    {"role": "user",   "content": "Provide information about insulin: include protein_name, organism, function, and molecular_weight_kDa."}
]

output = llm.create_chat_completion(
    messages=message_list,
    response_format="json_object"
)

# Parse and print individual fields
data = json.loads(output["choices"][0]["message"]["content"])

# Fill in the blanks:
print("Protein:", data[___])
print("Function:", data[___])
```

---

### Exercise 5 — Multi-turn conversation

**Task:** Build a conversation with a virology tutor that covers SARS-CoV-2 across three turns.

```python
from llama_cpp import Llama

class Conversation:
    def __init__(self, llm, system_prompt='', history=[]):
        self.llm     = llm
        self.history = [{"role": "system", "content": system_prompt}] + history

    def create_completion(self, user_prompt=''):
        self.history.append({"role": "user", "content": user_prompt})
        output = self.llm.create_chat_completion(messages=self.history)
        reply  = output["choices"][0]["message"]
        self.history.append(reply)
        return reply["content"]


llm   = Llama(model_path="path/to/model.gguf")
tutor = Conversation(llm, system_prompt="You are a virology tutor for biology students.")

# Turn 1
r1 = tutor.create_completion("What type of virus is SARS-CoV-2?")
print("Turn 1:", r1)

# Turn 2 — follow-up (uses context from Turn 1)
r2 = tutor.create_completion("How does its spike protein help it enter human cells?")
print("Turn 2:", r2)

# Turn 3 — deeper follow-up
r3 = tutor.create_completion("Why was this spike protein the target for vaccine development?")
print("Turn 3:", r3)
```

**Questions to reflect on:**
- What would happen if you replaced `tutor.create_completion(...)` with direct `llm(...)` calls? Why would Turn 2 fail?
- Try changing the `system_prompt` to make the tutor explain things as if to a 10-year-old. How does the output change?

---

### Exercise 6 — Challenge: Full pipeline

**Task:** Build a pipeline that:
1. Takes a list of 3 pathogens
2. Uses a JSON schema to get structured data about each one
3. Prints a summary table

```python
import json
from llama_cpp import Llama

llm = Llama(model_path="path/to/model.gguf")

# Define a schema for pathogen data
response_format = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "pathogen":       {"type": "string"},
            "type":           {"type": "string"},   # virus, bacterium, fungus, etc.
            "disease_caused": {"type": "string"},
            "transmission":   {"type": "string"}
        },
        "required": ["pathogen", "type", "disease_caused", "transmission"]
    }
}

pathogens = ["Mycobacterium tuberculosis", "HIV", "Plasmodium falciparum"]

print(f"{'Pathogen':<30} {'Type':<12} {'Disease':<25} {'Transmission'}")
print("-" * 90)

for p in pathogens:
    message_list = [
        {"role": "system", "content": "You are a microbiology database. Respond with JSON only."},
        {"role": "user",   "content": f"Provide data for: {p}"}
    ]

    output = llm.create_chat_completion(
        messages=message_list,
        response_format=response_format
    )

    data = json.loads(output["choices"][0]["message"]["content"])
    print(f"{data['pathogen']:<30} {data['type']:<12} {data['disease_caused']:<25} {data['transmission']}")
```

---

## Quick Reference Card

```python
# ── ZERO-SHOT WITH LABELS ──────────────────────────────────────────────────
text = """
INSTRUCTION: Write 2–3 sentences covering only key points.
QUESTION: What is the function of ribosomes?
ANSWER:
"""
output = llm(text, max_tokens=100)
answer = output["choices"][0]["text"]

# ── FEW-SHOT PROMPTING ────────────────────────────────────────────────────
text = """
Species: Homo sapiens
Kingdom: Animalia
Cell Type: Eukaryote

Species: Escherichia coli
Kingdom: Bacteria
Cell Type: Prokaryote

Species: YOUR_ORGANISM
Kingdom: ?
Cell Type:
"""
output = llm(f"Continue the entries: {text}", max_tokens=20)

# ── STOP WORDS ────────────────────────────────────────────────────────────
output = llm(text, stop=["Q:"], max_tokens=100)   # stops at "Q:"

# ── JSON OUTPUT ───────────────────────────────────────────────────────────
output = llm.create_chat_completion(
    messages=message_list,
    response_format="json_object"
)
import json
data = json.loads(output["choices"][0]["message"]["content"])

# ── JSON WITH SCHEMA ──────────────────────────────────────────────────────
response_format = {
    "type": "json_object",
    "schema": {
        "type": "object",
        "properties": {
            "field1": {"type": "string"},
            "field2": {"type": "string"}
        },
        "required": ["field1", "field2"]
    }
}
output = llm.create_chat_completion(messages=message_list, response_format=response_format)

# ── MULTI-TURN CONVERSATION ───────────────────────────────────────────────
class Conversation:
    def __init__(self, llm, system_prompt='', history=[]):
        self.llm     = llm
        self.history = [{"role": "system", "content": system_prompt}] + history

    def create_completion(self, user_prompt=''):
        self.history.append({"role": "user", "content": user_prompt})
        output = self.llm.create_chat_completion(messages=self.history)
        reply  = output["choices"][0]["message"]
        self.history.append(reply)
        return reply["content"]

conv = Conversation(llm, system_prompt="You are a biology tutor.")
r1   = conv.create_completion("What is PCR?")
r2   = conv.create_completion("What are its main steps?")   # remembers context
```

---

*Based on: "Working with Llama 3" — DataCamp, by Imtihan Ahmed (Machine Learning Engineer)*
