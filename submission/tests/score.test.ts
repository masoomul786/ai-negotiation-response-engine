/**
 * Tests for the scorer (score.ts).
 *
 * Verifies the spec scoring formula:
 *   0.60 × IntentMatchMean (with pass^k penalty)
 * + 0.20 × (1 − DurationNorm)
 * + 0.20 × (1 − TokenNorm)
 */

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

import type { Dashboard } from '../harness.js';
import {
  DEFAULT_WEIGHTS,
  DURATION_CAP_MS,
  INTENT_MATCH_MIN,
  PASS_K_PENALTY_THRESHOLD,
  TOKEN_CAP,
  minMaxNorm,
  scoreDashboard,
  scoreSubmission
} from '../score.js';

function makeDashboard(
  overrides: Partial<Dashboard['overall']> & {
    model_id?: string;
  }
): Dashboard {
  const { model_id, ...overall } = overrides;
  return {
    schema_version: 'dashboard-v1',
    model_id: model_id ?? 'model',
    dataset_id: 'ds',
    k: 4,
    overall: {
      intent_match_mean: 1,
      class_accuracy: 1,
      pass_k_rate: 1,
      behavioral_diversity_mean: 0.5,
      duration_ms_p50: 0,
      duration_ms_p95: 0,
      tokens_mean: 0,
      ...overall
    },
    per_class: {},
    confusion_matrix: { labels: [], matrix: [] },
    per_case: []
  };
}

describe('minMaxNorm', () => {
  test('linear scaling', () => {
    assert.equal(minMaxNorm(5, 0, 10), 0.5);
    assert.equal(minMaxNorm(0, 0, 10), 0);
    assert.equal(minMaxNorm(10, 0, 10), 1);
  });
  test('clamps to [0, 1]', () => {
    assert.equal(minMaxNorm(-5, 0, 10), 0);
    assert.equal(minMaxNorm(15, 0, 10), 1);
  });
  test('min == max → 0', () => {
    assert.equal(minMaxNorm(5, 5, 5), 0);
  });
});

describe('scoreDashboard', () => {
  test('perfect predictor (best on every axis) → score = 1.0', () => {
    const d = makeDashboard({
      intent_match_mean: 1,
      pass_k_rate: 1,
      duration_ms_p50: 0,
      tokens_mean: 0
    });
    const s = scoreDashboard(d);
    // 0.6*1 + 0.2*1 + 0.2*1 = 1.0
    assert.ok(Math.abs(s.score - 1.0) < 1e-9, `score=${s.score}`);
    assert.equal(s.dq_reasons.length, 0);
  });

  test('zero intent but fast/cheap → 0.40 from efficiency alone + DQ flag', () => {
    const d = makeDashboard({
      intent_match_mean: 0,
      pass_k_rate: 0,
      duration_ms_p50: 0,
      tokens_mean: 0
    });
    const s = scoreDashboard(d);
    // 0.6*0 + 0.2*1 + 0.2*1 = 0.40
    assert.ok(Math.abs(s.score - 0.4) < 1e-9, `score=${s.score}`);
    assert.ok(s.dq_reasons.some((r) => r.includes('intent_match_mean')));
  });

  test('at duration cap + token cap → efficiency contributes 0', () => {
    const d = makeDashboard({
      intent_match_mean: 1,
      pass_k_rate: 1,
      duration_ms_p50: DURATION_CAP_MS,
      tokens_mean: TOKEN_CAP
    });
    const s = scoreDashboard(d);
    // 0.6*1 + 0.2*0 + 0.2*0 = 0.60
    assert.ok(Math.abs(s.score - 0.6) < 1e-9, `score=${s.score}`);
  });

  test('pass_k penalty: pass_k_rate < 0.5 scales intent by pass_k_rate', () => {
    const d = makeDashboard({
      intent_match_mean: 0.8,
      pass_k_rate: 0.25,
      duration_ms_p50: 0,
      tokens_mean: 0
    });
    const s = scoreDashboard(d);
    // intentPenalized = 0.8 * 0.25 = 0.2
    assert.ok(Math.abs(s.intent_match_penalized - 0.2) < 1e-9);
    // 0.6*0.2 + 0.2*1 + 0.2*1 = 0.52
    assert.ok(Math.abs(s.score - 0.52) < 1e-9, `score=${s.score}`);
  });

  test('pass_k_rate == 0.5 → no penalty applied (threshold is strict-less-than)', () => {
    const d = makeDashboard({
      intent_match_mean: 0.8,
      pass_k_rate: 0.5,
      duration_ms_p50: 0,
      tokens_mean: 0
    });
    const s = scoreDashboard(d);
    assert.equal(s.intent_match_penalized, 0.8);
  });

  test('DQ when intent_match_mean < 0.30', () => {
    const d = makeDashboard({ intent_match_mean: 0.29 });
    const s = scoreDashboard(d);
    assert.ok(s.dq_reasons.some((r) => r.includes(`< ${INTENT_MATCH_MIN}`)));
  });

  test('DQ does NOT trigger at exactly 0.30', () => {
    const d = makeDashboard({ intent_match_mean: 0.3 });
    const s = scoreDashboard(d);
    assert.equal(s.dq_reasons.length, 0);
  });

  test('weights sum to 1 in defaults', () => {
    const w = DEFAULT_WEIGHTS;
    assert.ok(Math.abs(w.intent + w.duration + w.tokens - 1) < 1e-9);
  });

  test('PASS_K_PENALTY_THRESHOLD matches spec', () => {
    assert.equal(PASS_K_PENALTY_THRESHOLD, 0.5);
  });
});

describe('scoreSubmission', () => {
  test('mean of per-backend scores', () => {
    const ds = [
      makeDashboard({
        model_id: 'gemma',
        intent_match_mean: 1,
        pass_k_rate: 1,
        duration_ms_p50: 0,
        tokens_mean: 0
      }),
      makeDashboard({
        model_id: 'qwen',
        intent_match_mean: 0.5,
        pass_k_rate: 1,
        duration_ms_p50: 0,
        tokens_mean: 0
      })
    ];
    const s = scoreSubmission(ds);
    // gemma: 0.6*1 + 0.2*1 + 0.2*1 = 1.0
    // qwen:  0.6*0.5 + 0.2*1 + 0.2*1 = 0.7
    // mean = 0.85
    assert.ok(Math.abs(s.mean_score - 0.85) < 1e-9, `mean=${s.mean_score}`);
    assert.equal(s.dq, false);
  });

  test('DQ if any backend fails the intent floor', () => {
    const ds = [
      makeDashboard({ model_id: 'gemma', intent_match_mean: 1 }),
      makeDashboard({ model_id: 'qwen', intent_match_mean: 0.1 })
    ];
    const s = scoreSubmission(ds);
    assert.equal(s.dq, true);
    assert.ok(s.dq_reasons.some((r) => r.includes('[qwen]')));
  });

  test('errors on empty input', () => {
    assert.throws(() => scoreSubmission([]), /no dashboards/);
  });
});
