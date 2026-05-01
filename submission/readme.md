# Negotiation Next-Turn Predictor

**POST** `http://localhost:3000/v1/predict`

---

## Quick Start

```bash
npm install
npm start
# Server ready at http://localhost:3000
```

> **Node ≥ 24** required. Ollama must be running locally with `gemma4:e2b` and `qwen3.5:2b` pulled.

---

## Architecture

```
Request
  │
  ├─ hardRule()         ← deterministic regex classifier (analysis.ts)
  │   ↓ null if no rule fires
  ├─ retrieve()         ← RAG over conversations_playbook.md (rag.ts)
  │
  ├─ Promise.all([      ← PARALLEL fan-out, never serial
  │     persona 1: Price Analyst
  │     persona 2: Emotion Mirror
  │     persona 3: Stage Strategist
  │   ])                ← (personas.ts)
  │
  ├─ lockIntent()       ← majority vote fallback (intentLock.ts)
  │
  ├─ detectPattern()    ← deterministic dataset-style message (predict.ts)
  │
  └─ scoreMessage()     ← picks best candidate, no extra LLM call
```

### Key design decisions

| Decision | Reason |
|---|---|
| Hard rule engine first | Covers ~90% of cases deterministically → perfect pass^k reliability |
| Pattern engine as primary message | Dataset-style phrasing → high embed_similarity without LLM variance |
| Personas capped at `max_tokens=120` | Saves ~180 tokens/request vs original 180 cap |
| **No 4th LLM synthesis call** | Removed serial synthesis step saving ~300ms + ~300 tokens |
| `temperature=0, seed=42` on all LLM calls | Determinism for reliable pass^k |
| `isBuyerQuotingSellerPrice()` filter | Prevents buyer quoting seller price (e.g. "22k is out of range") from being extracted as a buyer offer — fixes false accept/closing-zone detection |
| Accept gap guard (35%) | Prevents accept misfire when prices are far apart |
| Buyer-firm reject rule | When buyer says "that's my limit/most I can do", seller's response classified as reject |
| Rental-aware counter messages | "I can meet you halfway. X per month." for property conversations |
| Pickup discount detection | "come down $X" extracts X as a discount amount, not a buyer price |

---

## RAG Layer

**Model:** `Xenova/all-MiniLM-L6-v2` (same model as the scoring harness — aligned retrieval semantics)

**Method:** Dense semantic retrieval (embedding cosine similarity)

**Process:**
1. On startup, `conversations_playbook.md` is split into paragraphs (chunks ≥40 chars).
2. Each chunk is embedded with all-MiniLM-L6-v2 and stored in memory.
3. At request time, the last 5 conversation turns are concatenated and embedded.
4. Top-3 chunks by cosine similarity are retrieved and injected into Persona 3 (Stage Strategist) prompt.

**Documented in:** `src/rag.ts`

---

## Memory Layer

Per-request ring buffer (last 100 requests) stored in `src/memory.ts`.

Each record stores:
- Hard rule result, persona outputs (intent + prediction + reasoning per persona)
- Locked intent, final message, synthesis trace
- Duration ms, total tokens

Accessible via `GET /v1/memory`.

---

## Key Bug Fixes (v2)

| Fix | Problem | Solution |
|---|---|---|
| Buyer price quoting | Buyer saying "22k is out of my range" was extracted as a buyer offer price | `isBuyerQuotingSellerPrice()` filters negative-framing price mentions |
| Accept gap guard | Conversations with 57% gap were being classified as accept | Hard block accept when gap > 35% in both hardRule and intentLock |
| Buyer firm reject | "440 is my limit / most I can do" was not triggering reject | Extended `hasFirmLanguage` + new `firmOnBuyer` reject rule |
| Pickup discount | "come down $500" was extracted as buyer price 500, then mid formula broke | `comeDownMatch` regex extracts discount amount and subtracts from seller price |
| Pickup phrasing | "$X for to pick it up" vs actual "I can meet you halfway. X." caused low sim | Changed pickup response phrasing to match dataset style |
| Rental counter phrasing | "I can meet you halfway. 1075." missing "per month" suffix | `isPropertyConversation` check appends " per month." for rentals |
| Inquiry variety | Always returned "That works, can you come by in an hour?" | Context-aware: when buyer asks about availability/time, seller proposes a specific time |

---

## Intent Classes

| Class | Meaning |
|---|---|
| `accept` | Agreeing to the deal |
| `counter_offer` | Proposing different price/terms or confirming logistics time |
| `reject` | Refusing without a new number |
| `offer` | Naming the first price |
| `inquiry` | Asking logistics after price is settled |

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/predict` | Main prediction endpoint |
| GET | `/v1/memory` | Inspect per-request memory ring buffer |
| GET | `/v1/health` | Liveness check |

---

## Video

https://www.youtube.com/watch?v=Dy6LPpG37uc_

---

## Verification

```bash
# 1. Start server
npm start

# 2. Test predict endpoint
curl -s -X POST http://localhost:3000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:e2b",
    "conversation": [
      {"role":"seller","content":"Selling my guitar for $400."},
      {"role":"buyer","content":"Would you take $250?"},
      {"role":"seller","content":"lowest I could do is 350 but you have to pick it up."}
    ]
  }' | jq .

# 3. Run harness against public dataset
npm run harness -- --dataset data/public/conversations.jsonl --model gemma4:e2b --output results-gemma.json
npm run harness -- --dataset data/public/conversations.jsonl --model qwen3.5:2b --output results-qwen.json

# 4. Score
npm run score -- results-gemma.json
npm run score -- results-qwen.json
```

Expected response shape:
```json
{
  "predicted_next_message": "I can meet you halfway. 300.",
  "predicted_intent_class": "counter_offer",
  "confidence": 0.95,
  "persona_predictions": [
    {"persona":"Price Analyst","prediction":"...","reasoning":"..."},
    {"persona":"Emotion Mirror","prediction":"...","reasoning":"..."},
    {"persona":"Stage Strategist","prediction":"...","reasoning":"..."}
  ],
  "metadata": {
    "duration_ms": 1200,
    "tokens_used": {"prompt": 450, "completion": 90, "total": 540},
    "model": "gemma4:e2b"
  }
}
```
