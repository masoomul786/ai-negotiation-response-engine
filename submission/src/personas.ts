// personas.ts — Three parallel persona agents with distinct analytical priors.
// Diversity is structural: each persona uses a DIFFERENT primary signal.
// Token budget per persona: max_tokens=120 (down from 180) — saves ~180 tokens/request.

import { callOllama, parseJSON } from './ollama.js';
import type { Turn, PriceData } from './analysis.js';

export interface NegContext {
  next: 'buyer' | 'seller';
  latestBuyer: number | null;
  latestSeller: number | null;
  predictedPrice: number | null;
  gapPct: string;
  closing: boolean;
  firm: boolean;
  ragContext: string;
  lastFiveTurns: string;
  lastThreeTurns: string;
  fullConvText: string;
  vocab: string;
  convLen: number;
}

export interface PersonaOutput {
  persona: string;
  intent: string;
  prediction: string;
  reasoning: string;
  promptTokens: number;
  completionTokens: number;
}

const VALID = new Set(['accept', 'counter_offer', 'reject', 'offer', 'inquiry']);

function safeIntent(v: unknown): string {
  if (typeof v !== 'string') return 'counter_offer';
  const n = v.toLowerCase().trim().replace(/\s+/g, '_');
  return VALID.has(n) ? n : 'counter_offer';
}

const INTENT_GUIDE = `INTENT (predict NEXT speaker's response):
- accept: agrees to deal ("deal/I'll take it/works for me/very good X it is")
- counter_offer: proposes different price OR confirms logistics time ("I'll be there around 5")
- reject: refuses without new number ("can't go beyond/no problem good luck/look elsewhere")
- offer: names FIRST price when NO prior price in conversation
- inquiry: asks logistics AFTER price is agreed (when/where/address/time)
KEY: deal/sold→accept | walkaway→reject | prices far apart→counter_offer | no price yet→offer | price settled+logistics→inquiry`;

const STYLE_NOTE = `CRAIGSLIST STYLE — short, casual, specific:
counter_offer buyer: "It is still used though. Will you do 300?"
counter_offer seller: "I can meet you halfway. 114."
counter_offer seller: "285 is too low. I bought it for $500. I can go to $300, take it or leave it."
accept buyer: "I'd be okay with 60. And yes, I'll pick up today."
accept seller: "You've got yourself a deal...a great deal! $9000 it is!"
reject seller: "no problem good luck"
offer buyer: "I was thinking $100 if there are no extra bells or whistles."
inquiry seller: "That works, can you come by in an hour?"
OUTPUT ONLY valid JSON: {"intent":"<class>","message":"<10-40 word casual message>","reasoning":"<6 words>"}`;

// ── Persona 1: Price Gap Analyst — focuses purely on numbers ─────────────
function pricePersonaPrompt(ctx: NegContext): { system: string; user: string } {
  const priceInfo = [
    ctx.latestBuyer  ? `buyer=$${ctx.latestBuyer}`  : 'buyer=none',
    ctx.latestSeller ? `seller=$${ctx.latestSeller}` : 'seller=none',
    ctx.predictedPrice ? `mid=$${ctx.predictedPrice}` : null,
  ].filter(Boolean).join(' ');

  const system = `You are a PRICE GAP ANALYST predicting the next Craigslist negotiation message.
Primary signal: numerical gap between buyer/seller prices.
- Gap >20%, no movement → counter_offer or reject
- Gap <12% → very likely accept  
- No prices yet → offer
- Prices settled + logistics question → inquiry
- Logistics time confirmation → counter_offer

${INTENT_GUIDE}

${STYLE_NOTE}`;

  const user = `Prices: ${priceInfo} | Gap: ${ctx.gapPct} | Closing: ${ctx.closing} | Firm: ${ctx.firm}
Last 3 turns:
${ctx.lastThreeTurns}
Next speaker: ${ctx.next}
JSON:`;

  return { system, user };
}

// ── Persona 2: Tone & Emotion Mirror — ignores price math ────────────────
function mirrorPersonaPrompt(ctx: NegContext): { system: string; user: string } {
  const lastLine = ctx.lastThreeTurns.split('\n').at(-1)?.replace(/^(BUYER|SELLER):\s*/i, '') ?? '';
  const emotional = /\b(sorry|strenuous|baby|divorce|moving|sick|broke|budget|struggling|tough|difficult|desperate|really\s+need)\b/i.test(ctx.fullConvText);

  const system = `You are a TONE & EMOTION ANALYST predicting the next Craigslist negotiation message.
Primary signal: emotional register, social reciprocity, conversational pragmatics — NOT arithmetic.
Emotional context: ${emotional ? 'sympathetic/personal circumstances present' : 'neutral transactional tone'}

Rules:
- Sympathetic context + fair offer → lean accept
- Aggressive/dismissive language → lean reject  
- Friendly back-and-forth with gap → counter_offer
- Question after price settlement → inquiry
- Do NOT agree to accept if gap is still large (>15%)

${INTENT_GUIDE}

${STYLE_NOTE}`;

  const user = `Responding to: "${lastLine.slice(0, 140)}"
${ctx.lastFiveTurns}
Next speaker: ${ctx.next} | Firm: ${ctx.firm} | buyer=$${ctx.latestBuyer ?? '?'} seller=$${ctx.latestSeller ?? '?'}
JSON:`;

  return { system, user };
}

// ── Persona 3: Stage Strategist — uses RAG + conversation arc ────────────
function patternPersonaPrompt(ctx: NegContext): { system: string; user: string } {
  const stage = ctx.convLen <= 3 ? 'EARLY (info-gathering, no prices yet)'
    : ctx.convLen <= 7 ? 'MID (active bargaining, prices being exchanged)'
    : 'LATE (closing zone, one side will accept or walk)';

  const system = `You are a NEGOTIATION STAGE STRATEGIST predicting the next Craigslist negotiation message.
Primary signal: conversation stage and arc patterns — not arithmetic, not tone.
Current stage: ${stage}

Stage heuristics:
- EARLY → offer or inquiry (first price being named)
- MID → counter_offer is most common (active price discovery)  
- LATE → accept or reject (deal closes or falls apart)
- "I'll be there at [time]" even late-stage → counter_offer (CraigslistBargain dataset convention)
- Seller "make me an offer" + prices exist → buyer responds with counter_offer

Playbook context:
${ctx.ragContext.slice(0, 350)}

${INTENT_GUIDE}

${STYLE_NOTE}`;

  const user = `Full conversation:
${ctx.fullConvText}
Next speaker: ${ctx.next} | Stage: ${stage} | Gap: ${ctx.gapPct} | Closing: ${ctx.closing}
JSON:`;

  return { system, user };
}

export async function runPersonas(ctx: NegContext, model: string): Promise<PersonaOutput[]> {
  const configs = [
    { name: 'Price Analyst',      ...pricePersonaPrompt(ctx) },
    { name: 'Emotion Mirror',     ...mirrorPersonaPrompt(ctx) },
    { name: 'Stage Strategist',   ...patternPersonaPrompt(ctx) },
  ];

  // Strict parallel fan-out — all 3 simultaneously, never serial
  const results = await Promise.all(configs.map(async (cfg): Promise<PersonaOutput> => {
    try {
      const r = await callOllama({ model, system: cfg.system, prompt: cfg.user, max_tokens: 120 });
      const parsed = parseJSON(r.text);
      const intent = safeIntent(parsed?.intent);
      const msg = typeof parsed?.message === 'string' && parsed.message.length > 1
        ? (parsed.message as string).slice(0, 300) : '';
      const reasoning = typeof parsed?.reasoning === 'string'
        ? (parsed.reasoning as string).slice(0, 80) : 'pattern match';
      return {
        persona: cfg.name, intent, prediction: msg, reasoning,
        promptTokens: r.promptTokens, completionTokens: r.completionTokens,
      };
    } catch {
      return {
        persona: cfg.name, intent: 'counter_offer', prediction: '',
        reasoning: 'error fallback', promptTokens: 0, completionTokens: 0,
      };
    }
  }));

  return results;
}

export function buildCtx(
  conv: Turn[],
  prices: PriceData,
  closing: boolean,
  firm: boolean,
  ragContext: string,
  vocab: string,
  next: 'buyer' | 'seller',
): NegContext {
  const lastThreeTurns = conv.slice(-3).map(t => `${t.role.toUpperCase()}: ${t.content}`).join('\n');
  const lastFiveTurns  = conv.slice(-5).map(t => `${t.role.toUpperCase()}: ${t.content}`).join('\n');
  const fullConvText   = conv.slice(-8).map(t => `${t.role.toUpperCase()}: ${t.content}`).join('\n');
  const gapPct = prices.latestBuyer !== null && prices.latestSeller !== null
    ? (Math.abs(prices.latestSeller - prices.latestBuyer) / Math.max(prices.latestSeller, prices.latestBuyer) * 100).toFixed(1) + '%'
    : 'unknown';

  return {
    next, latestBuyer: prices.latestBuyer, latestSeller: prices.latestSeller,
    predictedPrice: prices.predictedPrice, gapPct, closing, firm, ragContext,
    lastThreeTurns, lastFiveTurns, fullConvText, vocab, convLen: conv.length,
  };
}
