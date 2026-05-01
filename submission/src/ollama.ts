// ollama.ts
// Thin wrapper around the local Ollama HTTP API.
// Uses /api/chat so think:false is respected by gemma4:e2b and qwen3.5:2b.

const BASE_URL = process.env.OLLAMA_URL ?? 'http://localhost:11434';
const CALL_TIMEOUT_MS = 90_000;
const MAX_RETRIES = 2;

export interface OllamaCallParams {
  model: string;
  prompt: string;
  system?: string;
  max_tokens?: number;
}

export interface OllamaCallResult {
  text: string;
  promptTokens: number;
  completionTokens: number;
}

async function callOnce(p: OllamaCallParams): Promise<OllamaCallResult> {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), CALL_TIMEOUT_MS);

  try {
    const messages: Array<{ role: string; content: string }> = [];
    if (p.system) messages.push({ role: 'system', content: p.system });
    messages.push({ role: 'user', content: p.prompt });

    const body: Record<string, unknown> = {
      model: p.model,
      messages,
      stream: false,
      think: false,
      options: {
        temperature: 0,
        seed: 42,
        num_predict: p.max_tokens ?? 200,
      },
    };

    const res = await fetch(`${BASE_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: ac.signal,
    });

    if (!res.ok) throw new Error(`Ollama ${res.status}: ${await res.text()}`);

    const data = await res.json() as {
      message?: { content: string };
      prompt_eval_count?: number;
      eval_count?: number;
    };

    return {
      text: data.message?.content ?? '',
      promptTokens: data.prompt_eval_count ?? 0,
      completionTokens: data.eval_count ?? 0,
    };
  } finally {
    clearTimeout(timer);
  }
}

export async function callOllama(p: OllamaCallParams): Promise<OllamaCallResult> {
  let lastErr: Error = new Error('unknown');
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      if (attempt > 0) await new Promise(r => setTimeout(r, 1500 * attempt));
      return await callOnce(p);
    } catch (err) {
      lastErr = err as Error;
      console.warn(`  [ollama] attempt ${attempt + 1} failed: ${lastErr.message}`);
    }
  }
  throw lastErr;
}

// Strips markdown fences, <think> blocks, finds outermost {...} block.
export function parseJSON(raw: string): Record<string, unknown> | null {
  let cleaned = raw.replace(/<think>[\s\S]*?<\/think>/gi, '');
  cleaned = cleaned.replace(/```(?:json)?/gi, '').trim();
  try {
    return JSON.parse(cleaned);
  } catch {
    const start = cleaned.indexOf('{');
    if (start === -1) return null;
    let depth = 0;
    for (let i = start; i < cleaned.length; i++) {
      if (cleaned[i] === '{') depth++;
      else if (cleaned[i] === '}') {
        depth--;
        if (depth === 0) {
          try { return JSON.parse(cleaned.slice(start, i + 1)); } catch { break; }
        }
      }
    }
    return null;
  }
}

export async function warmModel(model: string): Promise<void> {
  try {
    await callOllama({ model, prompt: 'hi', max_tokens: 1 });
    console.log(`  warmed: ${model}`);
  } catch (e) {
    console.warn(`  warm failed for ${model}:`, e);
  }
}
