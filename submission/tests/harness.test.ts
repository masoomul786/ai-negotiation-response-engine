/**
 * Tests for the benchmark harness.
 *
 * Run with:
 *   npm test
 */

import { test, describe, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer, type Server, type IncomingMessage, type ServerResponse } from 'node:http';
import { writeFile, mkdtemp, rm } from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';

import {
  INTENT_CLASSES,
  isIntentClass,
  classMatch,
  cosine,
  percentiles,
  mean,
  embed,
  embedSimilarity,
  behavioralDiversity,
  callPredict,
  loadDataset,
  runHarness,
  type PredictResponse,
  type DatasetRecord,
  type IntentClass
} from '../harness.js';


describe('classMatch', () => {
  test('exact match → 1', () => {
    assert.equal(classMatch('accept', 'accept'), 1);
    assert.equal(classMatch('counter_offer', 'counter_offer'), 1);
  });
  test('adjacent match → 0.5', () => {
    assert.equal(classMatch('offer', 'counter_offer'), 0.5); // counter_offer → offer
    assert.equal(classMatch('counter_offer', 'offer'), 0.5); // offer → counter_offer
    assert.equal(classMatch('counter_offer', 'reject'), 0.5); // reject → counter_offer
  });
  test('non-adjacent → 0', () => {
    assert.equal(classMatch('inquiry', 'accept'), 0);
    assert.equal(classMatch('accept', 'reject'), 0);
    assert.equal(classMatch('inquiry', 'offer'), 0);
  });
});

describe('cosine', () => {
  test('identical unit vectors → 1', () => {
    assert.ok(Math.abs(cosine([1, 0, 0], [1, 0, 0]) - 1) < 1e-9);
  });
  test('orthogonal → 0', () => {
    assert.ok(Math.abs(cosine([1, 0], [0, 1])) < 1e-9);
  });
  test('anti-parallel → -1', () => {
    assert.ok(Math.abs(cosine([1, 0], [-1, 0]) + 1) < 1e-9);
  });
  test('length mismatch → 0', () => {
    assert.equal(cosine([1, 2], [1, 2, 3]), 0);
  });
  test('zero magnitude → 0', () => {
    assert.equal(cosine([0, 0], [1, 0]), 0);
  });
  test('known value: 45° vectors', () => {
    const c = cosine([1, 0], [1, 1]);
    assert.ok(Math.abs(c - Math.SQRT1_2) < 1e-9);
  });
});

describe('percentiles', () => {
  test('empty → 0/0', () => {
    assert.deepEqual(percentiles([]), { p50: 0, p95: 0 });
  });
  test('single value', () => {
    assert.deepEqual(percentiles([42]), { p50: 42, p95: 42 });
  });
  test('10 values: p50=60, p95=100', () => {
    const r = percentiles([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    assert.equal(r.p50, 60);
    assert.equal(r.p95, 100);
  });
  test('unsorted input is sorted before indexing', () => {
    const r = percentiles([100, 10, 50, 20, 30]);
    assert.equal(r.p50, 30);
  });
});

describe('mean', () => {
  test('empty → 0', () => {
    assert.equal(mean([]), 0);
  });
  test('normal', () => {
    assert.equal(mean([1, 2, 3, 4, 5]), 3);
  });
});

describe('isIntentClass', () => {
  test('valid intent', () => {
    assert.equal(isIntentClass('accept'), true);
    assert.equal(isIntentClass('counter_offer'), true);
    assert.equal(isIntentClass('inquiry'), true);
  });
  test('invalid intent', () => {
    assert.equal(isIntentClass('ask_info'), false); // old taxonomy
    assert.equal(isIntentClass(''), false);
    assert.equal(isIntentClass('ACCEPT'), false); // case-sensitive
  });
});

describe('embed / embedSimilarity / behavioralDiversity (real MiniLM)', () => {
  test('embed returns a normalised 384-dim vector', async () => {
    const v = await embed('hello world');
    assert.equal(v.length, 384);
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    assert.ok(Math.abs(norm - 1) < 0.01, `expected unit norm, got ${norm}`);
  });

  test('identical strings → similarity ≈ 1 (after scaling)', async () => {
    const s = await embedSimilarity('hello world', 'hello world');
    assert.ok(s > 0.95, `expected close to 1, got ${s}`);
  });

  test('semantically similar > dissimilar', async () => {
    const sim = await embedSimilarity(
      "I'll take it at $300",
      "Sure, 300 works for me"
    );
    const diss = await embedSimilarity(
      "I'll take it at $300",
      "What color is the bike?"
    );
    assert.ok(sim > diss, `expected sim (${sim.toFixed(3)}) > diss (${diss.toFixed(3)})`);
  });

  test('empty inputs -> similarity 0', async () => {
    assert.equal(await embedSimilarity('', 'something'), 0);
    assert.equal(await embedSimilarity('something', ''), 0);
  });

  test('behavioralDiversity: identical texts ≈ 0', async () => {
    const d = await behavioralDiversity(['hello', 'hello', 'hello']);
    assert.ok(d < 0.01, `expected ~0, got ${d}`);
  });

  test('behavioralDiversity: diverse texts > 0.1', async () => {
    const d = await behavioralDiversity([
      "I'll pay $100, final offer.",
      "How many miles does it have?",
      "Can you deliver it this weekend?"
    ]);
    assert.ok(d > 0.1, `expected > 0.1, got ${d}`);
  });

  test('behavioralDiversity: < 2 texts → 0', async () => {
    assert.equal(await behavioralDiversity([]), 0);
    assert.equal(await behavioralDiversity(['only one']), 0);
  });

  test('behavioralDiversity: empty strings contribute 0 (not 1) — fix for blank-persona exploit', async () => {
    const allBlank = await behavioralDiversity(['', '', '']);
    assert.equal(allBlank, 0, `3 empty strings should score 0 diversity, got ${allBlank}`);
    const oneBlank = await behavioralDiversity([
      '',
      "I'll pay $100 and that's my final offer.",
      'Can you deliver it this weekend?'
    ]);
    assert.ok(oneBlank < 0.5, `one-blank case should score < 0.5, got ${oneBlank}`);
  });
});


function startMockServer(
  handler: (req: IncomingMessage, res: ServerResponse) => void | Promise<void>
): Promise<{ server: Server; url: string }> {
  return new Promise((resolve) => {
    const server = createServer(async (req, res) => {
      try {
        await handler(req, res);
      } catch {
        res.statusCode = 500;
        res.end();
      }
    });
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      if (typeof addr === 'string' || addr === null) throw new Error('unexpected address');
      resolve({ server, url: `http://127.0.0.1:${addr.port}/v1/predict` });
    });
  });
}

function cannedResponse(
  predicted: IntentClass,
  message: string,
  personaTexts: string[] = [
    "Let's settle at 300, final.",
    'What condition is it in overall?',
    'I can come pick it up this weekend.'
  ]
): PredictResponse {
  return {
    predicted_next_message: message,
    predicted_intent_class: predicted,
    confidence: 0.7,
    persona_predictions: personaTexts.map((p, i) => ({
      persona: `persona_${i}`,
      prediction: p,
      reasoning: 'r'
    })),
    metadata: {
      duration_ms: 10,
      tokens_used: { prompt: 100, completion: 20, total: 120 },
      model: 'mock-model'
    }
  };
}

describe('callPredict', () => {
  test('successful response', async () => {
    const inst = await startMockServer((_req, res) => {
      res.statusCode = 200;
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify(cannedResponse('accept', "Sounds good, we have a deal at 300")));
    });
    try {
      const r = await callPredict(inst.url, [{ role: 'seller', content: 'Can you do 300?' }], 'mock-model');
      assert.equal(r.predicted_intent_class, 'accept');
      assert.equal(r.persona_predictions.length, 3);
    } finally {
      inst.server.close();
    }
  });

  test('HTTP 500 → throws', async () => {
    const inst = await startMockServer((_req, res) => {
      res.statusCode = 500;
      res.end('internal error');
    });
    try {
      await assert.rejects(
        callPredict(inst.url, [{ role: 'seller', content: 'hi' }], 'mock-model'),
        /HTTP 500/
      );
    } finally {
      inst.server.close();
    }
  });

  test('malformed JSON → throws', async () => {
    const inst = await startMockServer((_req, res) => {
      res.statusCode = 200;
      res.setHeader('content-type', 'application/json');
      res.end('not json{{{');
    });
    try {
      await assert.rejects(callPredict(inst.url, [{ role: 'seller', content: 'hi' }], 'mock-model'));
    } finally {
      inst.server.close();
    }
  });

  test('invalid intent class in response → throws', async () => {
    const inst = await startMockServer((_req, res) => {
      res.statusCode = 200;
      res.setHeader('content-type', 'application/json');
      res.end(
        JSON.stringify({
          ...cannedResponse('accept', 'x'),
          predicted_intent_class: 'escalate' // no longer a valid class
        })
      );
    });
    try {
      await assert.rejects(
        callPredict(inst.url, [{ role: 'seller', content: 'hi' }], 'mock-model'),
        /invalid intent class/
      );
    } finally {
      inst.server.close();
    }
  });

  test('timeout → throws', async () => {
    const inst = await startMockServer(() => {
      // never responds
    });
    try {
      await assert.rejects(
        callPredict(inst.url, [{ role: 'seller', content: 'hi' }], 'mock-model', 100)
      );
    } finally {
      inst.server.close();
    }
  });

  test('rejects response with persona_predictions.length !== 3', async () => {
    const inst = await startMockServer((_req, res) => {
      res.statusCode = 200;
      res.setHeader('content-type', 'application/json');
      const bad = cannedResponse('accept', 'ok');
      bad.persona_predictions = bad.persona_predictions.slice(0, 2); // only 2
      res.end(JSON.stringify(bad));
    });
    try {
      await assert.rejects(
        callPredict(inst.url, [{ role: 'seller', content: 'hi' }], 'mock-model'),
        /exactly 3 entries/
      );
    } finally {
      inst.server.close();
    }
  });

  test('rejects response with duplicate persona names', async () => {
    const inst = await startMockServer((_req, res) => {
      res.statusCode = 200;
      res.setHeader('content-type', 'application/json');
      const bad = cannedResponse('accept', 'ok');
      bad.persona_predictions[1]!.persona = bad.persona_predictions[0]!.persona;
      res.end(JSON.stringify(bad));
    });
    try {
      await assert.rejects(
        callPredict(inst.url, [{ role: 'seller', content: 'hi' }], 'mock-model'),
        /3 distinct persona names/
      );
    } finally {
      inst.server.close();
    }
  });

  test('rejects response with empty persona.prediction', async () => {
    const inst = await startMockServer((_req, res) => {
      res.statusCode = 200;
      res.setHeader('content-type', 'application/json');
      const bad = cannedResponse('accept', 'ok');
      bad.persona_predictions[0]!.prediction = '';
      res.end(JSON.stringify(bad));
    });
    try {
      await assert.rejects(
        callPredict(inst.url, [{ role: 'seller', content: 'hi' }], 'mock-model'),
        /prediction must be a non-empty string/
      );
    } finally {
      inst.server.close();
    }
  });

  test('rejects response with missing predicted_next_message', async () => {
    const inst = await startMockServer((_req, res) => {
      res.statusCode = 200;
      res.setHeader('content-type', 'application/json');
      const bad = cannedResponse('accept', 'ok') as Partial<PredictResponse>;
      delete (bad as Record<string, unknown>).predicted_next_message;
      res.end(JSON.stringify(bad));
    });
    try {
      await assert.rejects(
        callPredict(inst.url, [{ role: 'seller', content: 'hi' }], 'mock-model'),
        /predicted_next_message/
      );
    } finally {
      inst.server.close();
    }
  });

  test('rejects response with confidence out of [0, 1]', async () => {
    const inst = await startMockServer((_req, res) => {
      res.statusCode = 200;
      res.setHeader('content-type', 'application/json');
      const bad = cannedResponse('accept', 'ok');
      bad.confidence = 2.5;
      res.end(JSON.stringify(bad));
    });
    try {
      await assert.rejects(
        callPredict(inst.url, [{ role: 'seller', content: 'hi' }], 'mock-model'),
        /confidence must be a number in \[0, 1\]/
      );
    } finally {
      inst.server.close();
    }
  });
});

describe('loadDataset', () => {
  let tmp: string;
  before(async () => {
    tmp = await mkdtemp(path.join(tmpdir(), 'harness-test-'));
  });
  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  test('valid JSONL', async () => {
    const file = path.join(tmp, 'valid.jsonl');
    await writeFile(
      file,
      [
        JSON.stringify({
          id: 'r1',
          conversation: [{ role: 'seller', content: 'Can you do 300?' }],
          actual_next_message: 'Yes, 300 works for me.',
          actual_intent_class: 'accept'
        }),
        JSON.stringify({
          id: 'r2',
          conversation: [{ role: 'buyer', content: 'How old is the bike?' }],
          actual_next_message: 'About 2 years, barely used.',
          actual_intent_class: 'offer'
        })
      ].join('\n')
    );
    const recs = await loadDataset(file);
    assert.equal(recs.length, 2);
    assert.equal(recs[0]!.id, 'r1');
    assert.equal(recs[1]!.actual_intent_class, 'offer');
  });

  test('malformed JSON line → throws', async () => {
    const file = path.join(tmp, 'broken.jsonl');
    await writeFile(file, 'this is not json');
    await assert.rejects(loadDataset(file), /parse error/);
  });

  test('invalid intent class → throws', async () => {
    const file = path.join(tmp, 'bad-intent.jsonl');
    await writeFile(
      file,
      JSON.stringify({
        id: 'r1',
        conversation: [{ role: 'seller', content: 'hi' }],
        actual_next_message: 'ok',
        actual_intent_class: 'not_a_real_intent'
      })
    );
    await assert.rejects(loadDataset(file), /validation failed/);
  });

  test('malformed turn (bad role) → throws', async () => {
    const file = path.join(tmp, 'bad-turn-role.jsonl');
    await writeFile(
      file,
      JSON.stringify({
        id: 'r1',
        conversation: [{ role: 'user', content: 'hi' }], // 'user' no longer valid
        actual_next_message: 'ok',
        actual_intent_class: 'accept'
      })
    );
    await assert.rejects(loadDataset(file), /malformed turn/);
  });

  test('malformed turn (missing content) → throws', async () => {
    const file = path.join(tmp, 'bad-turn-content.jsonl');
    await writeFile(
      file,
      JSON.stringify({
        id: 'r1',
        conversation: [{ role: 'seller' }],
        actual_next_message: 'ok',
        actual_intent_class: 'accept'
      })
    );
    await assert.rejects(loadDataset(file), /malformed turn/);
  });

  test('malformed turn (non-object) → throws', async () => {
    const file = path.join(tmp, 'non-object-turn.jsonl');
    await writeFile(
      file,
      JSON.stringify({
        id: 'r1',
        conversation: ['not-a-turn-object'],
        actual_next_message: 'ok',
        actual_intent_class: 'accept'
      })
    );
    await assert.rejects(loadDataset(file), /malformed turn/);
  });

  test('empty conversation → throws', async () => {
    const file = path.join(tmp, 'empty-convo.jsonl');
    await writeFile(
      file,
      JSON.stringify({
        id: 'r1',
        conversation: [],
        actual_next_message: 'ok',
        actual_intent_class: 'accept'
      })
    );
    await assert.rejects(loadDataset(file), /validation failed/);
  });

  test('skips blank lines', async () => {
    const file = path.join(tmp, 'blanks.jsonl');
    const record = JSON.stringify({
      id: 'r1',
      conversation: [{ role: 'seller', content: 'hi' }],
      actual_next_message: 'ok',
      actual_intent_class: 'accept'
    });
    await writeFile(file, `\n\n${record}\n\n`);
    const recs = await loadDataset(file);
    assert.equal(recs.length, 1);
  });
});

describe('runHarness end-to-end', () => {
  let tmp: string;
  let dataset: string;
  let inst: { server: Server; url: string };

  before(async () => {
    tmp = await mkdtemp(path.join(tmpdir(), 'harness-run-'));
    dataset = path.join(tmp, 'dataset.jsonl');
    const records: DatasetRecord[] = [
      {
        id: 'c1',
        conversation: [{ role: 'seller', content: "Lowest I'll go is 350." }],
        actual_next_message: 'Alright, 350 it is.',
        actual_intent_class: 'accept'
      },
      {
        id: 'c2',
        conversation: [{ role: 'seller', content: 'Here is the chair and ottoman.' }],
        actual_next_message: 'What condition is the wood in?',
        actual_intent_class: 'inquiry'
      }
    ];
    await writeFile(dataset, records.map((r) => JSON.stringify(r)).join('\n'));

    // Mock always returns "accept" — matches c1's intent but not c2's.
    inst = await startMockServer((_req, res) => {
      res.statusCode = 200;
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify(cannedResponse('accept', 'Alright, 350 it is.')));
    });
  });

  after(async () => {
    inst.server.close();
    await rm(tmp, { recursive: true, force: true });
  });

  test('produces dashboard with expected shape and aggregates', async () => {
    const dashboard = await runHarness({
      predictUrl: inst.url,
      datasetPath: dataset,
      datasetId: 'test-v1',
      trials: 2,
      model: 'test-model'
    });

    // Shape
    assert.equal(dashboard.schema_version, 'dashboard-v1');
    // Dashboard reflects what was REQUESTED (opts.model), not what the mock echoed back.
    assert.equal(dashboard.model_id, 'test-model');
    assert.equal(dashboard.dataset_id, 'test-v1');
    assert.equal(dashboard.k, 2);
    assert.equal(dashboard.per_case.length, 2);
    assert.equal(dashboard.confusion_matrix.labels.length, INTENT_CLASSES.length);
    assert.equal(
      dashboard.confusion_matrix.matrix.length,
      INTENT_CLASSES.length,
      'confusion matrix rows = #classes'
    );

    const c1 = dashboard.per_case.find((c) => c.id === 'c1');
    const c2 = dashboard.per_case.find((c) => c.id === 'c2');
    assert.ok(c1 && c2, 'both cases present');
    assert.equal(c1.trials.length, 2);
    assert.equal(c2.trials.length, 2);

    // c1: mock always returns accept → class_match = 1 on every trial
    for (const tr of c1.trials) {
      assert.equal(tr.class_match, 1);
      assert.equal(tr.predicted_intent, 'accept');
    }
    // c2: mock returns accept, ground truth inquiry; not adjacent → class_match = 0
    for (const tr of c2.trials) {
      assert.equal(tr.class_match, 0);
      assert.equal(tr.predicted_intent, 'accept');
    }

    // class_accuracy = correct/total = 2/4 = 0.5
    assert.ok(
      Math.abs(dashboard.overall.class_accuracy - 0.5) < 0.01,
      `class_accuracy=${dashboard.overall.class_accuracy}`
    );

    const acceptClass = dashboard.per_class['accept'];
    assert.ok(acceptClass, 'accept class present');
    assert.ok(Math.abs(acceptClass.precision - 0.5) < 0.01);
    assert.ok(Math.abs(acceptClass.recall - 1) < 0.01);
  });

  test('handles predict-API failures by skipping trials', async () => {
    const flaky = await startMockServer((_req, res) => {
      res.statusCode = 500;
      res.end();
    });
    try {
      const dashboard = await runHarness({
        predictUrl: flaky.url,
        datasetPath: dataset,
        datasetId: 'flaky-v1',
        trials: 2,
        model: 'flaky'
      });
      assert.equal(dashboard.per_case.length, 2);
      for (const c of dashboard.per_case) assert.equal(c.trials.length, 0);
      assert.equal(dashboard.overall.pass_k_rate, 0);
    } finally {
      flaky.server.close();
    }
  });
});


// Sanity tests that exercise the harness against mock predict APIs with known
// behaviours. Useful for eyeballing what a "perfect" vs "wrong" submission
// would score. Each test logs the resulting dashboard so you can inspect it.

describe('runHarness — perfect predictor (mock returns ground truth)', () => {
  let tmp: string;
  let dataset: string;
  const records: DatasetRecord[] = [
    {
      id: 'g1',
      conversation: [
        { role: 'buyer', content: 'How much for the sofa?' },
        { role: 'seller', content: "I'm asking 400. It's barely used." }
      ],
      actual_next_message: 'How about 350 and I pick it up?',
      actual_intent_class: 'counter_offer'
    },
    {
      id: 'g2',
      conversation: [
        { role: 'seller', content: 'The bike is $150 as listed.' },
        { role: 'buyer', content: "I'll take it at 120 if it's clean." }
      ],
      actual_next_message: "Deal. 120 works, I'll meet you at 5pm.",
      actual_intent_class: 'accept'
    },
    {
      id: 'g3',
      conversation: [
        { role: 'seller', content: 'Let me know which color you want.' },
        { role: 'buyer', content: 'Is the phone unlocked and in good condition?' }
      ],
      actual_next_message: 'Yes, unlocked and barely used, no scratches.',
      actual_intent_class: 'offer'
    }
  ];

  before(async () => {
    tmp = await mkdtemp(path.join(tmpdir(), 'harness-perfect-'));
    dataset = path.join(tmp, 'data.jsonl');
    await writeFile(dataset, records.map((r) => JSON.stringify(r)).join('\n'));
  });

  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  test('scores 1.0 on intent_match, class_accuracy, and pass_k_rate', async () => {
    const byKey = new Map<string, DatasetRecord>(
      records.map((r) => [JSON.stringify(r.conversation), r])
    );

    const inst = await startMockServer(async (req, res) => {
      const chunks: Buffer[] = [];
      for await (const c of req) chunks.push(c as Buffer);
      const body = JSON.parse(Buffer.concat(chunks).toString()) as { conversation: unknown };
      const rec = byKey.get(JSON.stringify(body.conversation));
      if (!rec) {
        res.statusCode = 500;
        res.end('unknown conversation');
        return;
      }
      const resp: PredictResponse = {
        predicted_next_message: rec.actual_next_message,
        predicted_intent_class: rec.actual_intent_class,
        confidence: 1.0,
        persona_predictions: [
          {
            persona: 'literal',
            prediction: rec.actual_next_message,
            reasoning: 'echo the ground truth'
          },
          {
            persona: 'aggressive',
            prediction: "That's way too high, I'm walking away.",
            reasoning: 'aggressive stance'
          },
          {
            persona: 'analytical',
            prediction: 'Let me think about the total cost including delivery.',
            reasoning: 'analytical stance'
          }
        ],
        metadata: {
          duration_ms: 12,
          tokens_used: { prompt: 400, completion: 60, total: 460 },
          model: 'oracle-perfect'
        }
      };
      res.statusCode = 200;
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify(resp));
    });

    try {
      const dashboard = await runHarness({
        predictUrl: inst.url,
        datasetPath: dataset,
        datasetId: 'perfect-v1',
        trials: 3,
        model: 'mock-perfect'
      });

      console.log('\n── Perfect predictor dashboard ──────────────────');
      console.log(JSON.stringify(dashboard.overall, null, 2));
      console.log('per_class:', JSON.stringify(dashboard.per_class, null, 2));
      console.log('pass_k per case:', dashboard.per_case.map((c) => `${c.id}:${c.pass_k.passed}`).join(' '));
      console.log('─────────────────────────────────────────────────');

      assert.equal(dashboard.overall.class_accuracy, 1, 'class_accuracy should be 1.0');
      assert.ok(
        dashboard.overall.intent_match_mean > 0.99,
        `intent_match_mean=${dashboard.overall.intent_match_mean}`
      );
      assert.equal(dashboard.overall.pass_k_rate, 1, 'pass_k_rate should be 1.0');
      assert.ok(
        dashboard.overall.behavioral_diversity_mean > 0.15,
        `behavioral_diversity_mean=${dashboard.overall.behavioral_diversity_mean}`
      );

      for (const c of dashboard.per_case) {
        assert.equal(c.pass_k.passed, true, `case ${c.id} should pass`);
        for (const tr of c.trials) {
          assert.equal(tr.class_match, 1);
          assert.ok(tr.embed_similarity > 0.99, `embed_similarity=${tr.embed_similarity}`);
        }
      }

      for (const cls of ['counter_offer', 'accept', 'offer'] as const) {
        const pc = dashboard.per_class[cls];
        assert.ok(pc, `per_class missing ${cls}`);
        assert.equal(pc.precision, 1);
        assert.equal(pc.recall, 1);
        assert.equal(pc.f1, 1);
      }
    } finally {
      inst.server.close();
    }
  });
});

describe('runHarness — wrong predictor (mock returns a non-adjacent intent)', () => {
  let tmp: string;
  let dataset: string;
  const records: DatasetRecord[] = [
    {
      id: 'w1',
      conversation: [{ role: 'seller', content: 'Final offer: $200.' }],
      actual_next_message: 'Alright, $200 works.',
      actual_intent_class: 'accept'
    },
    {
      id: 'w2',
      conversation: [{ role: 'seller', content: 'Priced at $150.' }],
      actual_next_message: 'How many miles does it have?',
      actual_intent_class: 'inquiry'
    }
  ];

  before(async () => {
    tmp = await mkdtemp(path.join(tmpdir(), 'harness-wrong-'));
    dataset = path.join(tmp, 'data.jsonl');
    await writeFile(dataset, records.map((r) => JSON.stringify(r)).join('\n'));
  });

  after(async () => {
    await rm(tmp, { recursive: true, force: true });
  });

  test('scores near zero on class accuracy and pass_k_rate', async () => {
    const inst = await startMockServer((_req, res) => {
      res.statusCode = 200;
      res.setHeader('content-type', 'application/json');
      // Always predict `reject` with an unrelated message.
      // reject is not adjacent to accept or inquiry, so class_match = 0 for both.
      res.end(
        JSON.stringify(
          cannedResponse('reject', "No way, you're trying to rip me off here.")
        )
      );
    });
    try {
      const dashboard = await runHarness({
        predictUrl: inst.url,
        datasetPath: dataset,
        datasetId: 'wrong-v1',
        trials: 2,
        model: 'mock-wrong'
      });

      console.log('\n── Wrong predictor dashboard ────────────────────');
      console.log(JSON.stringify(dashboard.overall, null, 2));
      console.log('─────────────────────────────────────────────────');

      assert.equal(dashboard.overall.class_accuracy, 0, 'class_accuracy should be 0');
      assert.equal(dashboard.overall.pass_k_rate, 0, 'pass_k_rate should be 0');
      assert.ok(
        dashboard.overall.intent_match_mean < 0.5,
        `intent_match_mean=${dashboard.overall.intent_match_mean}`
      );
    } finally {
      inst.server.close();
    }
  });
});
