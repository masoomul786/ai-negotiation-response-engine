// rag.ts
// Local semantic retrieval over the negotiation playbook.
// Uses all-MiniLM-L6-v2 — same model as the scoring harness for aligned semantics.

import * as fs from 'fs';
import * as path from 'path';
import type { Turn } from './analysis.js';

type Extractor = (text: string, opts: Record<string, unknown>) => Promise<{ data: Float32Array }>;

interface Chunk { text: string; vec: number[]; }

let extractor: Extractor | null = null;
let chunks: Chunk[] = [];
let ready = false;

export async function initRAG(): Promise<void> {
  console.log('RAG: loading all-MiniLM-L6-v2 ...');
  try {
    const { pipeline, cos_sim } = await import('@xenova/transformers') as {
      pipeline: (task: string, model: string, opts: Record<string, unknown>) => Promise<Extractor>;
      cos_sim: (a: number[], b: number[]) => number;
    };

    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      cache_dir: path.join(process.cwd(), 'models'),
      local_files_only: false,
    });

    (globalThis as Record<string, unknown>).__cos_sim = cos_sim;

    const pbPath = path.join(process.cwd(), 'data', 'domain', 'conversations_playbook.md');
    const text = fs.existsSync(pbPath)
      ? fs.readFileSync(pbPath, 'utf8')
      : fallbackPlaybook();

    const raw = text.split(/\n\n+/).map(c => c.trim()).filter(c => c.length > 40);
    console.log(`RAG: indexing ${raw.length} chunks ...`);

    chunks = await Promise.all(raw.map(async t => ({ text: t, vec: await embed(t) })));
    ready = true;
    console.log(`RAG: ready (${chunks.length} chunks)`);
  } catch (e) {
    console.error('RAG init failed:', e);
    ready = false;
  }
}

async function embed(text: string): Promise<number[]> {
  const out = await extractor!(text, { pooling: 'mean', normalize: true });
  return Array.from(out.data);
}

function cosineSim(a: number[], b: number[]): number {
  const fn = (globalThis as Record<string, unknown>).__cos_sim as (a: number[], b: number[]) => number;
  return fn(a, b);
}

// Retrieval: embed last 5 turns, return top-3 most relevant playbook chunks.
// This aligns retrieval semantics with the evaluation embedding model.
export async function retrieve(conv: Turn[]): Promise<string> {
  if (!ready || !extractor) return fallbackPlaybook().split('\n\n').slice(0, 2).join('\n\n');
  try {
    const query = conv.slice(-5).map(t => t.content).join(' ').slice(0, 600);
    const qvec = await embed(query);
    const scored = chunks.map(c => ({ text: c.text, score: cosineSim(qvec, c.vec) }));
    return scored.sort((a, b) => b.score - a.score).slice(0, 3).map(s => s.text).join('\n\n');
  } catch {
    return fallbackPlaybook().split('\n\n').slice(0, 2).join('\n\n');
  }
}

function fallbackPlaybook(): string {
  return `Counter-offer tactics: acknowledge budget constraints, move 10-20% incrementally, anchor to a specific price.

Price resistance: "too high/low" signals willingness to continue. Propose a price that splits the gap by 30-40%.

Closing signals: "final offer", "last price", "deal", or gap under 12% means negotiation is nearly done.

Inquiry handling: questions about condition or availability early on are exploratory. Late questions signal readiness to close.

Rejection vs soft rejection: hard rejection with no counter means breakdown. Soft rejection means they want a better offer.

Acceptance language: "deal", "okay", "works for me", "I'll take it", "sold" after 2-5 rounds means acceptance.`;
}
