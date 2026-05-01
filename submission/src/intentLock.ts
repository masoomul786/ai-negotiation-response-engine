// intentLock.ts — Fallback intent resolution when no hard rule fires.
// Used only when hardRule() returns null (~5-10% of cases).

import type { PersonaOutput, NegContext } from './personas.js';

const VALID = new Set(['accept', 'counter_offer', 'reject', 'offer', 'inquiry']);

export function lockIntent(personas: PersonaOutput[], ctx: NegContext): string {
  // Tally votes
  const votes: Record<string, number> = {};
  for (const p of personas) {
    const intent = VALID.has(p.intent) ? p.intent : 'counter_offer';
    votes[intent] = (votes[intent] ?? 0) + 1;
  }

  const ranked = Object.entries(votes).sort((a, b) => b[1] - a[1]);
  const topIntent = ranked[0]![0];
  const topCount = ranked[0]![1];
  const bothPrices = ctx.latestBuyer !== null && ctx.latestSeller !== null;

  // Guard: block inquiry during active price negotiation with both prices present
  if (topIntent === 'inquiry' && bothPrices && !ctx.closing) {
    const fallback = ranked.find(([i]) => i !== 'inquiry');
    return fallback ? fallback[0] : 'counter_offer';
  }

  // Guard: block accept when price gap is too large
  if (topIntent === 'accept' && bothPrices) {
    const gap = Math.abs(ctx.latestSeller! - ctx.latestBuyer!) / Math.max(ctx.latestSeller!, ctx.latestBuyer!);
    if (gap > 0.35) return 'counter_offer'; // absolute hard block
    if (gap > 0.10 && !ctx.closing) return 'counter_offer';
  }

  // Clear majority (2 or 3 of 3 agree)
  if (topCount >= 2) return topIntent;

  // Three-way tie — use contextual tiebreaker
  const lastTurn = ctx.lastThreeTurns.split('\n').at(-1)?.toLowerCase() ?? '';

  // Closing signals override
  if (ctx.closing) {
    if (ctx.next === 'buyer' && ctx.firm) return 'accept';
    if (ctx.next === 'seller' && ctx.firm) return 'reject';
    return 'accept';
  }

  // Strong accept signals in last turn
  if (/\b(deal|sold|agreed|i.?ll\s+take|works\s+for\s+me|sounds\s+good)\b/.test(lastTurn))
    return 'accept';

  // Strong reject signals in last turn
  if (/\b(price\s+is\s+firm|not\s+going\s+lower|take\s+it\s+or\s+leave|not\s+interested|walking\s+away)\b/.test(lastTurn))
    return 'reject';

  if (ctx.firm) return 'reject';

  // Both prices exist → negotiation is live → counter_offer
  if (bothPrices) return 'counter_offer';

  // No prices, question present → offer or inquiry
  if (lastTurn.includes('?') && ctx.latestBuyer === null && ctx.latestSeller === null)
    return ctx.convLen >= 6 ? 'inquiry' : 'offer';

  // Safe default
  return 'counter_offer';
}
