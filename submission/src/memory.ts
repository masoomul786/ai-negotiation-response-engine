// memory.ts
// Per-request memory ring buffer (last 100 requests).
// Spec: "per-request memory layer storing persona outputs and synthesis trace"

export interface RequestRecord {
  id: string;
  ts: number;
  model: string;
  convLen: number;
  hardRuleFired: boolean;
  hardRuleIntent: string | null;
  personaOutputs: Array<{ persona: string; intent: string; prediction: string; reasoning: string }>;
  lockedIntent: string;
  finalIntent: string;
  finalMessage: string;
  selectedPersona: string | null;
  overriddenByHardRule: boolean;
  synthesisTrace: string;
  durationMs: number;
  totalTokens: number;
}

const MAX = 100;
const store: RequestRecord[] = [];

export function newId(): string { return crypto.randomUUID(); }

export function save(rec: RequestRecord): void {
  store.push(rec);
  if (store.length > MAX) store.shift();
}

export function recent(n = 5): RequestRecord[] {
  return store.slice(-n);
}

export function stats() {
  if (store.length === 0) return { total: 0, hardRuleRate: 0, avgTokens: 0, avgMs: 0 };
  const hr = store.filter(r => r.hardRuleFired).length;
  return {
    total: store.length,
    hardRuleRate: hr / store.length,
    avgTokens: Math.round(store.reduce((s, r) => s + r.totalTokens, 0) / store.length),
    avgMs: Math.round(store.reduce((s, r) => s + r.durationMs, 0) / store.length),
  };
}
