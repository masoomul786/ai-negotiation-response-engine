# 🔍 Negotiation Next-Turn Predictor

A deterministic + LLM hybrid system for predicting the **next message** and **intent class** in negotiation conversations.

This system is optimized for **high reliability, low latency, and evaluation consistency**, combining rule-based logic, retrieval-augmented generation (RAG), and parallel multi-persona LLM inference.

---

## 📌 Problem Definition

Given a negotiation conversation:

* Predict the **next seller message**
* Classify the **intent** into one of:

  * `accept`
  * `counter_offer`
  * `reject`
  * `offer`
  * `inquiry`

Constraints:

* High semantic similarity with expected dataset responses
* Deterministic behavior for pass^k evaluation
* Low latency and token usage

---

## 🧠 System Overview

```
                ┌──────────────┐
                │  Input JSON  │
                └──────┬───────┘
                       │
              ┌────────▼────────┐
              │  hardRule()     │  ← deterministic classifier
              └────────┬────────┘
                       │ (fallback if null)
              ┌────────▼────────┐
              │   retrieve()     │  ← RAG (MiniLM embeddings)
              └────────┬────────┘
                       │
        ┌──────────────┴──────────────┐
        │     Parallel Personas        │
        │  (LLM - temp=0, seed=42)    │
        │                              │
        │  • Price Analyst             │
        │  • Emotion Mirror            │
        │  • Stage Strategist          │
        └──────────────┬──────────────┘
                       │
              ┌────────▼────────┐
              │ lockIntent()     │ ← majority voting
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ detectPattern()  │ ← dataset-aligned phrasing
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ scoreMessage()   │ ← final selection
              └─────────────────┘
```

---

## ⚙️ Core Components

### 1. Hard Rule Engine (`analysis.ts`)

* Regex-based deterministic classifier
* Covers ~90% of cases
* Eliminates unnecessary LLM calls
* Includes:

  * Buyer price filtering
  * Accept-gap guard (≤35%)
  * Firm rejection detection

---

### 2. Retrieval Layer (`rag.ts`)

* Model: `Xenova/all-MiniLM-L6-v2`
* Method: cosine similarity over embeddings
* Input: last 5 conversation turns
* Output: top-3 semantically relevant chunks

Used to enhance **Stage Strategist persona**.

---

### 3. Persona Inference (`personas.ts`)

Executed in parallel (Promise.all):

| Persona          | Responsibility                        |
| ---------------- | ------------------------------------- |
| Price Analyst    | Numerical reasoning, price boundaries |
| Emotion Mirror   | Tone alignment, human-like phrasing   |
| Stage Strategist | Conversation progression strategy     |

Settings:

* `temperature = 0`
* `seed = 42`
* `max_tokens = 120`

---

### 4. Intent Lock (`intentLock.ts`)

* Majority voting across personas
* Fallback to rule-based decision when needed
* Prevents inconsistent intent classification

---

### 5. Pattern Engine (`predict.ts`)

* Generates dataset-aligned responses
* Avoids LLM variability
* Improves embedding similarity scores

---

### 6. Scoring Layer

* Selects best candidate message
* No additional LLM calls
* Optimized for evaluation harness

---

## 🚀 Performance Optimizations

| Optimization         | Impact                          |
| -------------------- | ------------------------------- |
| Hard rules first     | Reduces LLM usage significantly |
| Parallel personas    | Eliminates serial latency       |
| No synthesis step    | Saves ~300ms + tokens           |
| Token cap (120)      | Efficient inference             |
| Deterministic config | Stable pass^k results           |

---

## 🧪 API

### POST `/v1/predict`

#### Request

```json
{
  "model": "gemma4:e2b",
  "conversation": [
    {"role":"seller","content":"Selling my guitar for $400."},
    {"role":"buyer","content":"Would you take $250?"},
    {"role":"seller","content":"Lowest I could do is $350."}
  ]
}
```

#### Response

```json
{
  "predicted_next_message": "I can meet you halfway. 300.",
  "predicted_intent_class": "counter_offer",
  "confidence": 0.95
}
```

---

### GET `/v1/memory`

Returns recent request traces:

* persona outputs
* tokens used
* latency

---

### GET `/v1/health`

Service liveness check.

---

## 🧱 Memory Layer

* Ring buffer (last 100 requests)
* Stores:

  * intents
  * predictions
  * reasoning traces
  * latency + token usage

---

## 🐞 Key Edge-Case Handling

| Case                       | Solution                                |
| -------------------------- | --------------------------------------- |
| Buyer quoting seller price | Filtered via negative framing detection |
| Large price gaps           | Accept blocked (>35%)                   |
| Firm buyer limit           | Forced reject classification            |
| Pickup discounts           | Parsed as delta, not absolute price     |
| Rental conversations       | Adds "per month" normalization          |

---

## 🧰 Setup

```bash
npm install
npm start
```

Requirements:

* Node.js ≥ 24
* Ollama running locally:

  * gemma4:e2b
  * qwen3.5:2b

---

## 🧪 Evaluation

```bash
npm run harness -- --dataset data/public/conversations.jsonl
npm run score -- results.json
```

---

## 📌 Design Principles

* Determinism over randomness
* Rule-first, LLM-second
* Parallelism over sequential pipelines
* Dataset-aligned outputs
* Minimal token usage

---

## 📄 License

MIT License
