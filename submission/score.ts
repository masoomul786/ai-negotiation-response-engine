#!/usr/bin/env node
/**
 * Cognitive Diplomat — scorer.
 *
 * Takes the dashboard JSON file(s) produced by harness.ts for a SINGLE
 * submission (typically one per required backend) and prints the spec-defined
 * score. The reviewer feeds the printed numbers into the Topcoder scorecard.
 *
 * Formula (from spec.md §Judging):
 *   predictor_score_per_backend =
 *       0.60 × IntentMatchMean (× pass_k_rate if pass_k_rate < 0.5)
 *     + 0.20 × (1 − DurationNorm)
 *     + 0.20 × (1 − TokenNorm)
 *
 *   submission_total = mean(per-backend scores across required backends)
 *
 * DurationNorm / TokenNorm use fixed spec caps — 60 000 ms, 4 000 tokens.
 *
 * DQ (from spec): IntentMatchMean < 0.30 on any backend.
 *
 * Usage:
 *   npx tsx score.ts results-gemma.json results-qwen.json
 */

import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import type { Dashboard } from './harness.js';

// Spec-level constants. Keep in sync with spec.md.
export const DURATION_CAP_MS = 60_000;
export const TOKEN_CAP = 4_000;
export const INTENT_MATCH_MIN = 0.30;
export const PASS_K_PENALTY_THRESHOLD = 0.5;

export interface ScoreWeights {
  intent: number;
  duration: number;
  tokens: number;
}
export const DEFAULT_WEIGHTS: ScoreWeights = {
  intent: 0.6,
  duration: 0.2,
  tokens: 0.2
};

export interface PerBackendScore {
  model_id: string;
  dataset_id: string;
  intent_match_mean: number;
  pass_k_rate: number;
  intent_match_penalized: number;
  duration_ms_p50: number;
  tokens_mean: number;
  duration_norm: number;
  token_norm: number;
  score: number;
  dq_reasons: string[];
}

export interface SubmissionScore {
  per_backend: PerBackendScore[];
  mean_score: number;
  dq: boolean;
  dq_reasons: string[];
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

/** Scale a value into [0, 1] using a min-max range. When min == max, returns 0. */
export function minMaxNorm(x: number, min: number, max: number): number {
  const range = max - min;
  if (range <= 0) return 0;
  return clamp01((x - min) / range);
}

/** Compute the per-backend score for a single dashboard. */
export function scoreDashboard(
  d: Dashboard,
  weights: ScoreWeights = DEFAULT_WEIGHTS
): PerBackendScore {
  const intent = d.overall.intent_match_mean;
  const passK = d.overall.pass_k_rate;

  // pass^k penalty — low reliability scales the intent contribution.
  const intentPenalized = passK < PASS_K_PENALTY_THRESHOLD ? intent * passK : intent;

  const durationNorm = minMaxNorm(d.overall.duration_ms_p50, 0, DURATION_CAP_MS);
  const tokenNorm = minMaxNorm(d.overall.tokens_mean, 0, TOKEN_CAP);

  const score =
    weights.intent * intentPenalized +
    weights.duration * (1 - durationNorm) +
    weights.tokens * (1 - tokenNorm);

  const dqReasons: string[] = [];
  if (intent < INTENT_MATCH_MIN) {
    dqReasons.push(`intent_match_mean=${intent.toFixed(3)} < ${INTENT_MATCH_MIN}`);
  }

  return {
    model_id: d.model_id,
    dataset_id: d.dataset_id,
    intent_match_mean: intent,
    pass_k_rate: passK,
    intent_match_penalized: intentPenalized,
    duration_ms_p50: d.overall.duration_ms_p50,
    tokens_mean: d.overall.tokens_mean,
    duration_norm: durationNorm,
    token_norm: tokenNorm,
    score,
    dq_reasons: dqReasons
  };
}

/** Compute the submission-level score from per-backend dashboards. */
export function scoreSubmission(
  dashboards: Dashboard[],
  weights: ScoreWeights = DEFAULT_WEIGHTS
): SubmissionScore {
  if (dashboards.length === 0) {
    throw new Error('no dashboards supplied');
  }
  const perBackend = dashboards.map((d) => scoreDashboard(d, weights));
  const mean = perBackend.reduce((s, b) => s + b.score, 0) / perBackend.length;
  const dqReasons = perBackend.flatMap((b) =>
    b.dq_reasons.map((r) => `[${b.model_id}] ${r}`)
  );
  return {
    per_backend: perBackend,
    mean_score: mean,
    dq: dqReasons.length > 0,
    dq_reasons: dqReasons
  };
}

// ──────────────────────────────────────────────────────────
// CLI
// ──────────────────────────────────────────────────────────

async function main(files: string[]): Promise<void> {
  if (files.length === 0) {
    console.error('usage: tsx score.ts <dashboard.json>...');
    process.exit(1);
  }

  const dashboards: Dashboard[] = [];
  for (const f of files) {
    const raw = await readFile(f, 'utf8');
    dashboards.push(JSON.parse(raw) as Dashboard);
  }

  const result = scoreSubmission(dashboards);

  console.log(
    `score ceilings: duration ${DURATION_CAP_MS} ms, tokens ${TOKEN_CAP} ` +
      `(above ceiling → 0 contribution on that axis; no hard cap, no DQ)`
  );
  console.log('');
  for (const b of result.per_backend) {
    console.log(
      `${b.model_id.padEnd(18)}` +
        ` score=${b.score.toFixed(3)}  intent=${b.intent_match_mean.toFixed(3)}` +
        `  pass_k=${b.pass_k_rate.toFixed(3)}` +
        `  dur_p50=${b.duration_ms_p50.toFixed(0)}ms (norm=${b.duration_norm.toFixed(2)})` +
        `  tokens=${b.tokens_mean.toFixed(0)} (norm=${b.token_norm.toFixed(2)})`
    );
  }
  console.log('');
  console.log(`mean score across backends: ${result.mean_score.toFixed(3)}`);
  console.log(`status: ${result.dq ? 'DQ' : 'OK'}`);
  if (result.dq) console.log(`DQ reasons: ${result.dq_reasons.join('; ')}`);
}

const isMain = Boolean(process.argv[1]) && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  await main(process.argv.slice(2));
}
