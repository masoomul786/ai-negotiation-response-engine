#!/usr/bin/env node
/**
 * Cognitive Diplomat benchmark harness.
 *
 * Evaluates a predict API against a JSONL dataset and emits a canonical
 * dashboard-v1 JSON. This is the same harness the reviewer uses for grading,
 * so your score depends on how well your predict API does on THIS script.
 *
 * Usage:
 *   npx tsx harness.ts \
 *     --predict-url http://localhost:3000/v1/predict \
 *     --dataset ./data/public/conversations.jsonl \
 *     --dataset-id public-v1 \
 *     --trials 4 \
 *     --output ./results.json
 *
 * Every flag can also be provided via env var (CLI flags win):
 *   PREDICT_URL, DATASET, DATASET_ID, TRIALS, OUTPUT, LLM_MODEL
 *
 * On first run, ~50 MB of MiniLM weights are downloaded by @xenova/transformers
 * and cached under ~/.cache/huggingface. Subsequent runs are offline.
 */

import { Command } from 'commander';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline, env as xenovaEnv, type FeatureExtractionPipeline } from '@xenova/transformers';

// Prefer cached weights; allow local cache in ./.cache/xenova if preferred.
xenovaEnv.allowRemoteModels = true;

export const INTENT_CLASSES = [
  'accept',
  'counter_offer',
  'reject',
  'offer',
  'inquiry'
] as const;
export type IntentClass = (typeof INTENT_CLASSES)[number];

// Negotiation adjacency — partial credit (0.5) when the predicted class is
// a close relative of the ground truth.
export const INTENT_ADJACENCY: Record<IntentClass, IntentClass[]> = {
  accept: [],
  counter_offer: ['offer'],
  reject: ['counter_offer'],
  offer: ['counter_offer'],
  inquiry: []
};

export function isIntentClass(s: string): s is IntentClass {
  return (INTENT_CLASSES as readonly string[]).includes(s);
}

export function classMatch(predicted: IntentClass, truth: IntentClass): 0 | 0.5 | 1 {
  if (predicted === truth) return 1;
  if (INTENT_ADJACENCY[truth].includes(predicted)) return 0.5;
  return 0;
}

export interface Turn {
  role: 'buyer' | 'seller';
  content: string;
}

export interface DatasetRecord {
  id: string;
  conversation: Turn[];
  actual_next_message: string;
  actual_intent_class: IntentClass;
}

export interface PersonaPrediction {
  persona: string;
  prediction: string;
  reasoning: string;
}

export interface TokenUsage {
  prompt: number;
  completion: number;
  total: number;
}

export interface PredictResponse {
  predicted_next_message: string;
  predicted_intent_class: IntentClass;
  confidence: number;
  persona_predictions: PersonaPrediction[];
  metadata: {
    duration_ms: number;
    tokens_used: TokenUsage;
    model: string;
  };
}

export interface TrialResult {
  predicted_intent: IntentClass;
  class_match: number;
  embed_similarity: number;
  duration_ms: number;
  tokens: TokenUsage;
  behavioral_diversity: number;
  persona_predictions?: PersonaPrediction[];
}

export interface CaseResult {
  id: string;
  ground_truth_intent: IntentClass;
  trials: TrialResult[];
  pass_k: { k: number; passed: boolean };
}

export interface Dashboard {
  schema_version: 'dashboard-v1';
  model_id: string;
  dataset_id: string;
  k: number;
  overall: {
    intent_match_mean: number;
    class_accuracy: number;
    pass_k_rate: number;
    behavioral_diversity_mean: number;
    duration_ms_p50: number;
    duration_ms_p95: number;
    tokens_mean: number;
  };
  per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  confusion_matrix: { labels: string[]; matrix: number[][] };
  per_case: CaseResult[];
}


// Embedder — sentence-transformers/all-MiniLM-L6-v2 via @xenova/transformers

let extractorPromise: Promise<FeatureExtractionPipeline> | null = null;
async function getExtractor(): Promise<FeatureExtractionPipeline> {
  if (!extractorPromise) {
    extractorPromise = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2') as Promise<FeatureExtractionPipeline>;
  }
  return extractorPromise;
}

export async function embed(text: string): Promise<number[]> {
  const extractor = await getExtractor();
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data as Float32Array);
}

export function cosine(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

export async function embedSimilarity(a: string, b: string): Promise<number> {
  if (!a || !b) return 0;
  const [va, vb] = await Promise.all([embed(a), embed(b)]);
  const c = cosine(va, vb);
  // Min-max scale from [0.4, 1.0] → [0, 1], clipped.
  return Math.max(0, Math.min(1, (c - 0.4) / (1.0 - 0.4)));
}

export async function behavioralDiversity(texts: string[]): Promise<number> {
  if (texts.length < 2) return 0;
  const vecs = await Promise.all(texts.map((t) => (t ? embed(t) : Promise.resolve([] as number[]))));
  const pairs: number[] = [];
  for (let i = 0; i < vecs.length; i++) {
    for (let j = i + 1; j < vecs.length; j++) {
      const a = vecs[i] ?? [];
      const b = vecs[j] ?? [];
      // Empty-string persona predictions contribute 0 diversity (not 1).
      // Rewarding blank personas with max diversity was an exploit.
      if (a.length === 0 || b.length === 0) {
        pairs.push(0);
      } else {
        pairs.push(1 - cosine(a, b));
      }
    }
  }
  return pairs.reduce((s, x) => s + x, 0) / pairs.length;
}

// ──────────────────────────────────────────────────────────
// Statistics helpers
// ──────────────────────────────────────────────────────────

export function percentiles(xs: number[]): { p50: number; p95: number } {
  if (xs.length === 0) return { p50: 0, p95: 0 };
  const sorted = [...xs].sort((a, b) => a - b);
  const at = (p: number): number => {
    const i = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
    return sorted[i] ?? 0;
  };
  return { p50: at(0.5), p95: at(0.95) };
}

export function mean(xs: number[]): number {
  return xs.length === 0 ? 0 : xs.reduce((s, x) => s + x, 0) / xs.length;
}


export async function callPredict(
  url: string,
  conversation: Turn[],
  model: string,
  timeoutMs = 120_000
): Promise<PredictResponse> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ conversation, model }),
      signal: ctrl.signal
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} ${res.statusText}`);
    }
    const data = (await res.json()) as PredictResponse;
    if (!data || typeof data !== 'object') throw new Error('non-object response');
    if (!isIntentClass(data.predicted_intent_class ?? '')) {
      throw new Error(`invalid intent class: ${String(data.predicted_intent_class)}`);
    }
    if (!Array.isArray(data.persona_predictions)) {
      throw new Error('persona_predictions missing or not an array');
    }
    if (data.persona_predictions.length !== 3) {
      throw new Error(
        `persona_predictions must have exactly 3 entries, got ${data.persona_predictions.length}`
      );
    }
    for (let i = 0; i < data.persona_predictions.length; i++) {
      const p = data.persona_predictions[i] as unknown as {
        persona?: unknown;
        prediction?: unknown;
        reasoning?: unknown;
      };
      if (!p || typeof p.persona !== 'string' || p.persona.length === 0) {
        throw new Error(`persona_predictions[${i}].persona must be a non-empty string`);
      }
      if (typeof p.prediction !== 'string' || p.prediction.length === 0) {
        throw new Error(`persona_predictions[${i}].prediction must be a non-empty string`);
      }
      if (typeof p.reasoning !== 'string') {
        throw new Error(`persona_predictions[${i}].reasoning must be a string`);
      }
    }
    const personaNames = new Set(data.persona_predictions.map((p) => p.persona));
    if (personaNames.size !== 3) {
      throw new Error(
        `persona_predictions must have 3 distinct persona names, got ${personaNames.size} distinct`
      );
    }
    if (typeof data.predicted_next_message !== 'string' || data.predicted_next_message.length === 0) {
      throw new Error('predicted_next_message must be a non-empty string');
    }
    if (typeof data.confidence !== 'number' || data.confidence < 0 || data.confidence > 1) {
      throw new Error(`confidence must be a number in [0, 1], got ${String(data.confidence)}`);
    }
    if (!data.metadata || typeof data.metadata.model !== 'string' || !data.metadata.model) {
      throw new Error('metadata.model missing or not a non-empty string');
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

// ──────────────────────────────────────────────────────────
// Dataset loader
// ──────────────────────────────────────────────────────────

export async function loadDataset(filepath: string): Promise<DatasetRecord[]> {
  const raw = await readFile(filepath, 'utf8');
  const records: DatasetRecord[] = [];
  let lineNo = 0;
  for (const line of raw.split('\n')) {
    lineNo++;
    const trimmed = line.trim();
    if (!trimmed) continue;
    let rec: DatasetRecord;
    try {
      rec = JSON.parse(trimmed) as DatasetRecord;
    } catch (err) {
      throw new Error(`dataset parse error on line ${lineNo}: ${(err as Error).message}`);
    }
    if (
      typeof rec.id !== 'string' ||
      !Array.isArray(rec.conversation) ||
      rec.conversation.length === 0 ||
      typeof rec.actual_next_message !== 'string' ||
      rec.actual_next_message.length === 0 ||
      !isIntentClass(rec.actual_intent_class)
    ) {
      throw new Error(`dataset validation failed on line ${lineNo}`);
    }
    for (let i = 0; i < rec.conversation.length; i++) {
      const turn = rec.conversation[i] as unknown as { role?: unknown; content?: unknown };
      if (
        !turn ||
        typeof turn !== 'object' ||
        (turn.role !== 'buyer' && turn.role !== 'seller') ||
        typeof turn.content !== 'string' ||
        turn.content.length === 0
      ) {
        throw new Error(
          `dataset validation failed on line ${lineNo}: malformed turn at index ${i}`
        );
      }
    }
    records.push(rec);
  }
  return records;
}

export interface RunOptions {
  predictUrl: string;
  datasetPath: string;
  datasetId: string;
  trials: number;
  model: string;
  timeoutMs?: number;
}

export async function runHarness(opts: RunOptions): Promise<Dashboard> {
  const records = await loadDataset(opts.datasetPath);
  const caseResults: CaseResult[] = [];
  const modelsSeen = new Set<string>();

  for (const rec of records) {
    process.stdout.write(`[${rec.id}]`);
    const trials: TrialResult[] = [];

    for (let t = 0; t < opts.trials; t++) {
      const t0 = performance.now();
      try {
        const resp = await callPredict(opts.predictUrl, rec.conversation, opts.model, opts.timeoutMs);
        const durationMs = Math.round(performance.now() - t0);

        const cMatch = classMatch(resp.predicted_intent_class, rec.actual_intent_class);
        const eSim = await embedSimilarity(resp.predicted_next_message, rec.actual_next_message);
        const bDiv =
          resp.persona_predictions.length >= 2
            ? await behavioralDiversity(resp.persona_predictions.map((p) => p.prediction))
            : 0;

        modelsSeen.add(resp.metadata.model);
        trials.push({
          predicted_intent: resp.predicted_intent_class,
          class_match: cMatch,
          embed_similarity: eSim,
          duration_ms: durationMs,
          tokens: resp.metadata?.tokens_used ?? { prompt: 0, completion: 0, total: 0 },
          behavioral_diversity: bDiv,
          persona_predictions: resp.persona_predictions
        });
        process.stdout.write('.');
      } catch (err) {
        process.stdout.write('x');
        console.error(`  [${rec.id}] trial ${t + 1}/${opts.trials} failed: ${(err as Error).message}`);
        // skip — the trial is dropped; does not count toward pass^k
      }
    }
    process.stdout.write('\n');

    const allPass =
      trials.length === opts.trials &&
      trials.every((x) => x.predicted_intent === rec.actual_intent_class && x.embed_similarity >= 0.6);

    caseResults.push({
      id: rec.id,
      ground_truth_intent: rec.actual_intent_class,
      trials,
      pass_k: { k: opts.trials, passed: allPass }
    });
  }

  // Aggregates
  const allTrials = caseResults.flatMap((c) => c.trials);
  const durations = allTrials.map((t) => t.duration_ms);
  const tokensTotals = allTrials.map((t) => t.tokens.total);
  const { p50, p95 } = percentiles(durations);

  const intentMatches = allTrials.map((t) => 0.5 * t.class_match + 0.5 * t.embed_similarity);
  const intentMatchMean = mean(intentMatches);

  let correct = 0;
  for (const c of caseResults) {
    for (const t of c.trials) {
      if (t.predicted_intent === c.ground_truth_intent) correct++;
    }
  }
  const classAccuracy = allTrials.length === 0 ? 0 : correct / allTrials.length;

  const passKRate =
    caseResults.length === 0 ? 0 : caseResults.filter((c) => c.pass_k.passed).length / caseResults.length;

  const behavioralDiversityMean = mean(allTrials.map((t) => t.behavioral_diversity));
  const tokensMean = mean(tokensTotals);

  // Per-class precision / recall / f1
  const perClass: Dashboard['per_class'] = {};
  for (const cls of INTENT_CLASSES) {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let support = 0;
    for (const c of caseResults) {
      if (c.ground_truth_intent === cls) support++;
      for (const t of c.trials) {
        if (t.predicted_intent === cls && c.ground_truth_intent === cls) tp++;
        else if (t.predicted_intent === cls && c.ground_truth_intent !== cls) fp++;
        else if (t.predicted_intent !== cls && c.ground_truth_intent === cls) fn++;
      }
    }
    if (support > 0) {
      const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
      const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
      const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
      perClass[cls] = { precision, recall, f1, support };
    }
  }

  // Confusion matrix
  const labels = [...INTENT_CLASSES];
  const matrix: number[][] = labels.map(() => labels.map(() => 0));
  for (const c of caseResults) {
    const truthIdx = labels.indexOf(c.ground_truth_intent);
    for (const t of c.trials) {
      const predIdx = labels.indexOf(t.predicted_intent);
      if (truthIdx >= 0 && predIdx >= 0) {
        const row = matrix[truthIdx];
        if (row) row[predIdx] = (row[predIdx] ?? 0) + 1;
      }
    }
  }

  // Dashboard model_id is always the model we REQUESTED (opts.model). The
  // predict API must echo this in metadata.model; if the echoed value ever
  // differs from the requested one, warn — this is an integrity signal.
  const mismatches = [...modelsSeen].filter((m) => m !== opts.model);
  if (mismatches.length > 0) {
    console.warn(
      `[harness] WARN: predict API echoed a different model than requested. ` +
        `requested=${opts.model}, echoed=${[...modelsSeen].join(', ')}`
    );
  }

  return {
    schema_version: 'dashboard-v1',
    model_id: opts.model,
    dataset_id: opts.datasetId,
    k: opts.trials,
    overall: {
      intent_match_mean: intentMatchMean,
      class_accuracy: classAccuracy,
      pass_k_rate: passKRate,
      behavioral_diversity_mean: behavioralDiversityMean,
      duration_ms_p50: p50,
      duration_ms_p95: p95,
      tokens_mean: tokensMean
    },
    per_class: perClass,
    confusion_matrix: { labels, matrix },
    per_case: caseResults
  };
}

// ──────────────────────────────────────────────────────────
// CLI
// ──────────────────────────────────────────────────────────

async function main(): Promise<void> {
const program = new Command();
program
  .name('harness')
  .description('Cognitive Diplomat benchmark harness — evaluates a predict API against a dataset.')
  .option('--predict-url <url>', 'predict API URL', process.env.PREDICT_URL)
  .option('--dataset <path>', 'JSONL dataset path', process.env.DATASET)
  .option('--dataset-id <id>', 'dataset identifier (written into output)', process.env.DATASET_ID)
  .option('--trials <n>', 'trials per case (pass^k)', process.env.TRIALS ?? '4')
  .option('--output <path>', 'output JSON path', process.env.OUTPUT ?? './results.json')
  .option('--model <name>', 'model to pass to the predict API in every request body (becomes dashboard model_id)', process.env.LLM_MODEL)
  .option('--timeout <ms>', 'per-call safety timeout in ms (trial drops on hit; not a DQ)', process.env.TIMEOUT ?? '120000')
  .parse(process.argv);

const opts = program.opts<{
  predictUrl?: string;
  dataset?: string;
  datasetId?: string;
  trials: string;
  output: string;
  model?: string;
  timeout: string;
}>();

function die(msg: string): never {
  console.error(`harness: ${msg}`);
  process.exit(1);
}

if (!opts.predictUrl) die('missing --predict-url (or PREDICT_URL env)');
if (!opts.dataset) die('missing --dataset (or DATASET env)');
if (!opts.datasetId) die('missing --dataset-id (or DATASET_ID env)');
if (!opts.model) die('missing --model (or LLM_MODEL env) — the model tag to pass to the predict API');

const trials = Number.parseInt(opts.trials, 10);
if (!Number.isFinite(trials) || trials < 1) die(`invalid --trials: ${opts.trials}`);
const timeoutMs = Number.parseInt(opts.timeout, 10);
if (!Number.isFinite(timeoutMs) || timeoutMs < 1000) die(`invalid --timeout: ${opts.timeout}`);

try {
  const dashboard = await runHarness({
    predictUrl: opts.predictUrl,
    datasetPath: opts.dataset,
    datasetId: opts.datasetId,
    trials,
    model: opts.model,
    timeoutMs
  });

  await mkdir(path.dirname(path.resolve(opts.output)), { recursive: true });
  await writeFile(opts.output, JSON.stringify(dashboard, null, 2));
  console.log(
    `\nwrote ${opts.output} — ${dashboard.per_case.length} cases, ` +
      `intent_match_mean=${dashboard.overall.intent_match_mean.toFixed(3)}, ` +
      `pass_k=${dashboard.overall.pass_k_rate.toFixed(3)}, ` +
      `class_acc=${dashboard.overall.class_accuracy.toFixed(3)}`
  );
} catch (err) {
  die((err as Error).message);
}
}

// Run the CLI only when this file is executed directly (not when imported by tests).
const isMain = Boolean(process.argv[1]) && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  await main();
}
