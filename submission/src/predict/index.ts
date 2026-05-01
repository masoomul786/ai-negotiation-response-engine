// index.ts
// Express server — three endpoints, startup sequence.

import express from 'express';
import { z } from 'zod';
import { initRAG } from '../rag.js';
import { warmModel } from '../ollama.js';
import { predict } from '../predict.js';
import { recent, stats } from '../memory.js';

const app = express();
app.use(express.json({ limit: '1mb' }));

const requestSchema = z.object({
  conversation: z.array(
    z.object({
      role: z.enum(['buyer', 'seller']),
      content: z.string().min(1).max(2000),
    })
  ).min(1).max(50),
  model: z.string().default('gemma4:e2b'),
});

// POST /v1/predict — main prediction endpoint
app.post('/v1/predict', async (req, res) => {
  const parsed = requestSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({
      error: 'Invalid input',
      details: parsed.error.message,
    });
  }
  try {
    const result = await predict(parsed.data);
    return res.status(200).json(result);
  } catch (err) {
    console.error('predict error:', err);
    return res.status(500).json({
      error: 'Prediction failed',
      details: String(err),
    });
  }
});

// GET /v1/memory — inspect per-request memory ring buffer
app.get('/v1/memory', (_req, res) => {
  res.json({ stats: stats(), recent: recent(5) });
});

// GET /v1/health — liveness check
app.get('/v1/health', (_req, res) => {
  res.json({ status: 'ok', ts: new Date().toISOString() });
});

async function start() {
  const port = parseInt(process.env.PORT ?? '3000', 10);

  console.log('=== Cognitive Diplomat — Negotiation Predictor ===');
  console.log('Startup sequence:');
  console.log('  1. Loading RAG index...');
  await initRAG();

  console.log('  2. Warming models...');
  await Promise.all([warmModel('gemma4:e2b'), warmModel('qwen3.5:2b')]);

  app.listen(port, () => {
    console.log(`  3. Server listening on http://localhost:${port}`);
    console.log('Ready.');
    console.log('  POST /v1/predict');
    console.log('  GET  /v1/memory');
    console.log('  GET  /v1/health');
  });
}

start().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
