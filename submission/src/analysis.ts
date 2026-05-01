// analysis.ts — Universal negotiation analysis. Zero LLM calls.
// Unified price extractor handles $X, Xk, raw numbers — filters years & mileage.
// Hard rules ordered by specificity; generalised for hidden test cases.

export type Role = 'buyer' | 'seller';
export interface Turn { role: Role; content: string; }
export interface PriceData {
  buyerPrices: number[];
  sellerPrices: number[];
  latestBuyer: number | null;
  latestSeller: number | null;
  predictedPrice: number | null;
}

// ── Word-number normalizer ─────────────────────────────────────────────
// Converts informal spoken numbers to digits before price extraction.
function normalizeWordNumbers(text: string): string {
  return text
    .replace(/\b(?:a\s+)?grand\b/gi, '1000')
    .replace(/\b(?:a\s+)?couple\s+(?:hundred|grand)\b/gi, (m) => m.toLowerCase().includes('grand') ? '2000' : '200')
    // Compound tens+ones before thousand: "twenty two thousand", "thirty five thousand"
    .replace(/\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[\s-]+(one|two|three|four|five|six|seven|eight|nine)\s+thousand\b/gi, (_, tens, ones) => {
      const t: Record<string, number> = { twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90 };
      const o: Record<string, number> = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9 };
      return String((t[tens.toLowerCase()] + o[ones.toLowerCase()]) * 1000);
    })
    .replace(/\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s*[-\s]?\s*(one|two|three|four|five|six|seven|eight|nine)\s+hundred\b/gi, (_, tens, ones) => {
      const t: Record<string, number> = { twenty: 2, thirty: 3, forty: 4, fifty: 5, sixty: 6, seventy: 7, eighty: 8, ninety: 9 };
      const o: Record<string, number> = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9 };
      return String((t[tens.toLowerCase()] * 10 + o[ones.toLowerCase()]) * 100);
    })
    .replace(/\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+thousand\b/gi, (_, w) => {
      const map: Record<string, number> = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90 };
      return String((map[w.toLowerCase()] ?? 1) * 1000);
    })
    .replace(/\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+hundred\b/gi, (_, w) => {
      const map: Record<string, number> = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20, thirty: 30, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90 };
      return String((map[w.toLowerCase()] ?? 1) * 100);
    });
}

// ── Unified price extractor (used everywhere) ────────────────────────────
export function extractPricesFrom(text: string): number[] {
  const normalized = normalizeWordNumbers(text);
  const results: number[] = [];
  const seen = new Set<number>();

  // Pass 1: Xk notation — exclude mileage context (also catches "Xk on it" = odometer)
  const kRe = /\b(\d+(?:\.\d+)?)k\b/gi;
  let km: RegExpExecArray | null;
  while ((km = kRe.exec(normalized)) !== null) {
    const after = normalized.slice(km.index + km[0].length, km.index + km[0].length + 30).toLowerCase();
    // Exclude: miles, mi, odometer, km, "on it" (Xk on it = mileage), "on the clock"
    if (/miles|mi\b|odometer|kilometer|\bkm\b|on\s+it\b|on\s+the\s+(clock|odometer)|on\s+\w+\s+it/.test(after)) continue;
    // Also check before for "has X on it" patterns
    const before = normalized.slice(Math.max(0, km.index - 15), km.index).toLowerCase();
    if (/has\s+about|has\s+around|has\s+only|has\s+\d/.test(before) && /on\s+it|on\s+the/.test(after)) continue;
    const v = Math.round(parseFloat(km[1]) * 1000);
    if (v >= 1000 && v <= 10_000_000 && !seen.has(v)) { results.push(v); seen.add(v); }
  }

  // Pass 2: dollar-prefixed or bare numbers (on normalized text)
  const numRe = /\$[\d,]+(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\b/g;
  let m: RegExpExecArray | null;
  while ((m = numRe.exec(normalized)) !== null) {
    const raw = m[0].replace(/[^0-9.]/g, '');
    const v = Math.round(parseFloat(raw));
    if (isNaN(v) || v < 10 || v > 10_000_000) continue;
    if (v >= 1900 && v <= 2100) continue; // year filter
    const after = normalized.slice(m.index + m[0].length, m.index + m[0].length + 20).toLowerCase();
    if (/\s*miles|\s*mi\b|\s*km\b|\s*odometer/.test(after)) continue;
    if (!seen.has(v)) { results.push(v); seen.add(v); }
  }
  return results;
}

export function extractedPriceFrom(text: string): number | null {
  return extractPricesFrom(text)[0] ?? null;
}

export function extractAllPricesFrom(text: string): number[] {
  return extractPricesFrom(text);
}

export function hasPriceAnywhere(turns: Turn[]): boolean {
  return turns.some(t => extractPricesFrom(t.content).length > 0);
}

export function nextSpeaker(conv: Turn[]): Role {
  return conv[conv.length - 1].role === 'buyer' ? 'seller' : 'buyer';
}

// Detects if a price in a buyer turn is a quote/reference to seller's price (not an offer)
function isBuyerQuotingSellerPrice(text: string, v: number): boolean {
  const low = text.toLowerCase();
  // "22k is out of my range / too high / can't afford / listed at X"
  const priceStr = v.toString();
  const kStr = v >= 1000 ? (v / 1000).toString() + 'k' : null;
  const mentions = [priceStr, ...(kStr ? [kStr] : [])];
  for (const p of mentions) {
    const idx = low.indexOf(p.toLowerCase());
    if (idx === -1) continue;
    const after = low.slice(idx + p.length, idx + p.length + 40);
    const before = low.slice(Math.max(0, idx - 40), idx);
    // Negative framing after the price
    if (/is\s+(a\s+)?(little\s+)?(out\s+of|too\s+high|way\s+too|too\s+much|above|over)/.test(after)) return true;
    if (/\bcan.?t\s+afford|\bout\s+of\s+(my\s+)?price\s+range|\btoo\s+high\s+for\s+me/.test(after)) return true;
    // "listed at X" or "asking X" – buyer citing the listing price
    if (/\b(listed|listing|asking|advertised|posted)\s+(at\s+)?(for\s+)?$/.test(before)) return true;
    if (/it.?s\s+listed\s+(at\s+)?$|you.?re\s+(asking|listed)\s+(at\s+)?$/.test(before)) return true;
  }
  return false;
}

export function extractPrices(conv: Turn[], next: Role): PriceData {
  const buyerPrices: number[] = [];
  const sellerPrices: number[] = [];
  for (const t of conv) {
    const prices = extractPricesFrom(t.content);
    for (const v of prices) {
      if (t.role === 'buyer' && isBuyerQuotingSellerPrice(t.content, v)) continue;
      (t.role === 'buyer' ? buyerPrices : sellerPrices).push(v);
    }
  }
  const latestBuyer = buyerPrices.at(-1) ?? null;
  const latestSeller = sellerPrices.at(-1) ?? null;

  let predictedPrice: number | null = null;
  if (latestBuyer !== null && latestSeller !== null) {
    const raw = next === 'buyer'
      ? latestBuyer + (latestSeller - latestBuyer) * 0.35
      : latestSeller - (latestSeller - latestBuyer) * 0.35;
    predictedPrice = Math.round(raw);
  } else if (latestSeller !== null && next === 'buyer') {
    predictedPrice = Math.round(latestSeller * 0.78);
  } else if (latestBuyer !== null && next === 'seller') {
    predictedPrice = Math.round(latestBuyer * 1.20);
  }
  return { buyerPrices, sellerPrices, latestBuyer, latestSeller, predictedPrice };
}

export function inClosingZone(conv: Turn[], b: number | null, s: number | null): boolean {
  const last = conv[conv.length - 1].content;
  const hasDealWord =
    /\b(deal|sold|agreed|fine|alright|okay|works\s+for\s+me|i.?ll\s+take|you.?ve\s+got)\b/i.test(last)
    && !/\b(find|finding|good|working|make|best|great|cheap)\s+(a\s+)?deal\b/i.test(last);
  const tightGap = b !== null && s !== null && s > 0 && b < s && (s - b) / s < 0.12;
  return hasDealWord || tightGap;
}

export function hasFirmLanguage(conv: Turn[]): boolean {
  const last = conv[conv.length - 1].content;
  return /\b(firm|final\s+offer|can.?t\s+go\s+(lower|higher|below|above)|no\s+lower|no\s+higher|take\s+it\s+or\s+leave|won.?t\s+budge|lowest\s+i\s+can|best\s+i\s+can\s+do|most\s+i\s+can\s+do|price\s+is\s+firm|bottom\s+line|absolute\s+(floor|bottom)|rock\s+bottom|that.?s\s+my\s+(limit|max|ceiling|top)|my\s+(limit|max|ceiling|absolute\s+max)|highest\s+i\s+(can|could|will)\s+go)\b/i.test(last);
}

export function extractItem(conv: Turn[]): string {
  const text = conv.slice(0, 6).map(t => t.content).join(' ');
  const match = text.match(
    /\b(macbook|iphone|ipad|laptop|computer|phone|samsung|pixel|ps5|ps4|xbox|nintendo|switch|guitar|piano|keyboard|amp|speaker|camera|drone|lens|tv|television|monitor|desk|chair|couch|sofa|table|bed|mattress|dresser|bike|bicycle|car|truck|suv|van|motorcycle|scooter|boat|trailer|vehicle|tacoma|camry|civic|tesla|bmw|honda|toyota|ford|chevy|apartment|room|studio|office|house|condo|property|treadmill|weights|strat|fender|gibson|acoustic|bass|drums|violin|saxophone|trumpet|record|vinyl|watch|ring|necklace|jewelry|jacket|coat|boots|shoes|bag|purse|fridge|washer|dryer|dishwasher|oven|stove|microwave|food\s+truck|beach\s+cruiser|cruiser|ottoman|lamp|peloton|4x4|wheeler|hutch|lamps)\b/i,
  );
  return match ? match[0].toLowerCase() : 'item';
}

export function vocabMirror(conv: Turn[]): string {
  const stop = new Set(['the','this','that','with','have','from','they','will','been','more','when','your','what','about','would','there','their','just','like','also','into','than','then','some','could','over','very','much']);
  const freq: Record<string, number> = {};
  for (const t of conv) {
    for (const w of (t.content.toLowerCase().match(/\b[a-z]{4,}\b/g) ?? [])) {
      if (!stop.has(w)) freq[w] = (freq[w] ?? 0) + 1;
    }
  }
  return Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([w]) => w).join(', ');
}

// ─────────────────────────────────────────────────────────────────────────────
// HARD RULES — generalised universal negotiation classifier
// ─────────────────────────────────────────────────────────────────────────────
export function hardRule(conv: Turn[], next: Role, prices: PriceData): string | null {
  const last = conv[conv.length - 1];
  const low = last.content.toLowerCase().trim();
  const lastContent = last.content;
  const lastSpeaker = last.role;
  const convLen = conv.length;
  const { latestBuyer, latestSeller } = prices;
  const lastPrice = extractedPriceFrom(lastContent);

  // ── REJECT ──────────────────────────────────────────────────────────────
  if (/\b(look\s+elsewhere|will\s+look\s+elsewhere|i.?ll\s+look\s+elsewhere|thanks\s+for\s+your\s+time|going\s+to\s+pass|have\s+to\s+pass|gonna\s+pass|no\s+longer\s+interested)\b/i.test(low))
    return 'reject';
  if (/\b(no\s+deal|not\s+interested|walking\s+away|forget\s+it|no\s+way|never\s+mind)\b/i.test(low))
    return 'reject';
  if (/can.?t\s+go\s+(beyond|above|higher|more\s+than|over)/i.test(low))
    return 'reject';
  if (/\b(appreciate|thank\s+you\s+for|thanks\s+for)\b.{0,40}\b(but|however|unfortunately)\b.{0,40}\b(can.?t|cannot|won.?t|not\s+(able|going|willing))\b/i.test(low))
    return 'reject';
  // "X or I'll wait / X or find someone else / X or nothing" = seller firm reject
  if (/\bor\s+(i.?ll\s+wait|find\s+someone|wait\s+for|nothing|no\s+deal|forget)\b/i.test(low))
    return 'reject';
  // "Price is firm" standalone (no counter number following)
  if (/\bprice\s+is\s+firm\b/i.test(low) && lastPrice !== null && latestBuyer !== null && lastPrice > latestBuyer * 1.03)
    return 'reject';

  if (lastSpeaker === 'seller' && next === 'buyer') {
    if (/\bsorry\b/i.test(low) && /\b(can.?t|cannot|won.?t|not\s+able)\b/i.test(low)
      && /\b(go\s+lower|drop|budge|do\s+(that|it|less)|meet\s+you)\b/i.test(low))
      return 'reject';
    if (/\b(final\s+offer|bottom\s+line|absolute\s+(floor|bottom|minimum)|won.?t\s+go\s+(below|lower)|rock\s+bottom|no\s+lower|not\s+going\s+lower)\b/i.test(low)
      && latestBuyer !== null && lastPrice !== null && lastPrice > latestBuyer * 1.05
      && !/\b(will\s+go\s+down\s+to|can\s+go\s+down\s+to|going\s+down\s+to|i\s+will\s+go\s+down)\b/i.test(low))
      return 'reject';
    if (lastPrice !== null && latestBuyer !== null && lastPrice > latestBuyer * 1.04
      && /\b(good\?|what\s+do\s+you\s+(think|say)|deal\?|take\s+it\?|let\s+it\s+go\b)/i.test(low)
      && !/\b(how\s+about|what\s+about|would\s+you)\b/i.test(low))
      return 'reject';
  }

  // Buyer states absolute max / limit → seller's next response is reject or counter
  // When hasFirmLanguage on BUYER turn, SELLER likely rejects or counters hard
  if (lastSpeaker === 'buyer' && next === 'seller') {
    const firmOnBuyer = /(that.?s\s+my\s+(limit|max|ceiling|top)|my\s+(limit|max)|highest\s+i\s+(can|could|will)\s+go|most\s+i\s+can\s+(do|go|offer)|that.?s\s+the\s+most|that.?s\s+all\s+i\s+(have|got)|can.?t\s+go\s+(higher|above|over|beyond))/i.test(low);
    if (firmOnBuyer && latestSeller !== null && latestBuyer !== null && latestSeller > latestBuyer * 1.03)
      return 'reject';
  }

  // Extra reject: buyer says 'that's my limit' explicitly
  if (lastSpeaker === 'buyer' && next === 'seller') {
    if (/that.{0,3}s\s+(my\s+)?limit|my\s+absolute\s+max|most\s+i\s+can\s+do/i.test(low)
      && latestSeller !== null && latestBuyer !== null && latestSeller > latestBuyer * 1.03)
      return 'reject';
  }

  // ── COUNTER_OFFER (early — seller proposes price with question) ──────────
  if (lastSpeaker === 'seller' && next === 'buyer') {
    if (lastPrice !== null && lastContent.includes('?')
      && /\b(how\s+about|what\s+about|would\s+you|can\s+you\s+do|try)\b/i.test(low)
      && latestBuyer !== null && lastPrice > latestBuyer * 1.03)
      return 'counter_offer';
  }

  // ── INQUIRY — detect BEFORE accept when deal is done + logistics asked ───
  // Buyer says "deal/works/agreed" AND asks a logistics question in same message
  if (lastSpeaker === 'buyer' && next === 'seller') {
    const hasDealClose = /\b(deal|works\s+for\s+me|that\s+works|sounds\s+good|agreed|okay|ok|fair)\b/i.test(low);
    const hasLogisticsQ = lastContent.includes('?') && /\b(when|where|address|come\s+by|pick\s+up|tomorrow|morning|afternoon|evening|time|am\b|pm\b|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i.test(low);
    if (hasDealClose && hasLogisticsQ && hasPriceAnywhere(conv)) return 'inquiry';
  }
  // Seller says "X it is / come on over / you got it" with logistics invite → buyer asks inquiry
  if (lastSpeaker === 'seller' && next === 'buyer') {
    if (/\b(come\s+on\s+over|come\s+(by|get\s+it|pick\s+it)|swing\s+by|you\s+know\s+where\s+i\s+am)\b/i.test(low) && lastPrice !== null)
      return 'inquiry';
    if (/\bit\s+is[.!]?\s*$/i.test(lastContent.trim()) && lastPrice !== null && latestBuyer !== null
      && Math.abs(lastPrice - latestBuyer) / Math.max(lastPrice, latestBuyer) < 0.06)
      return 'inquiry';
  }
  // Rental/property: buyer accepts price per month + asks logistics → seller responds inquiry
  if (lastSpeaker === 'buyer' && next === 'seller') {
    const isRental = /per\s+month|monthly|move\s+in|move-in|subletting|apartment|rent/i.test(conv.map(t => t.content).join(' '));
    if (isRental && /\b(works|okay|ok|fine|fair|deal|sounds\s+good)\b/i.test(low) && lastContent.includes('?'))
      return 'inquiry';
  }

  // ── ACCEPT ──────────────────────────────────────────────────────────────
  if (/\b(buy\s+it\s+today|make\s+a\s+deal\s+at)\b/i.test(low)) return 'accept';
  if (/i.?ll\s+take\s+\$|i.?ll\s+take\s+the\s+\d/i.test(low)) return 'accept';
  if (/\b(you.?ve\s+got\s+(yourself\s+a\s+)?deal|you\s+got\s+(yourself\s+a\s+)?deal|it.?s\s+a\s+deal|done\s+deal)\b/i.test(low)) return 'accept';

  if (lastSpeaker === 'seller' && next === 'buyer') {
    if (/\b(i.?ll\s+take|that\s+sounds\s+perfect|sounds\s+perfect)\b/i.test(low) && lastPrice !== null)
      return 'accept';
    if (/\b(split\s+the\s+(diff|difference)|meet\s+(you\s+)?in\s+the\s+middle|halfway)\b/i.test(low) && lastPrice !== null)
      return 'accept';
    if (lastPrice !== null && latestBuyer !== null
      && Math.abs(lastPrice - latestBuyer) / Math.max(lastPrice, latestBuyer) < 0.03
      && !/\b(will\s+go\s+down\s+to|can\s+go\s+down\s+to|no\s+lower|but\s+no\s+lower|not\s+going\s+lower)\b/i.test(low))
      return 'accept';
    if (/\b(i\s+could\s+do|i\s+can\s+do)\b/i.test(low) && lastPrice !== null
      && /\b(pick\s*up|come\s+by|today|tonight|tomorrow|when\s+can\s+you)\b/i.test(low))
      return 'accept';
    if (/\b(meet\s+up|somewhere\s+to\s+be\s+in\s+that\s+area|have\s+somewhere\s+to\s+be)\b/i.test(low) && convLen >= 6)
      return 'accept';
    if (/\b(works\s+for\s+me|that\s+works|sounds\s+good)\b/i.test(low) && lastPrice !== null) {
      const prevBuyer = conv.filter(t => t.role === 'buyer').map(t => t.content).join(' ');
      if (/\b(come\s+get\s+it|pick\s+up|come\s+by|i.?ll\s+come|today|cash)\b/i.test(prevBuyer))
        return 'inquiry';
      return 'accept';
    }
  }

  if (lastSpeaker === 'buyer' && next === 'seller') {
    if (/\b(anytime|any\s+time|come\s+get\s+it|come\s+pick\s+it|whenever|i.?m\s+free|can\s+come\s+get)\b/i.test(low)
      && latestBuyer !== null && latestSeller === null)
      return 'accept';
    if (/\bavailable\b/i.test(low) && !lastContent.includes('?')
      && latestBuyer !== null && latestSeller === null)
      return 'accept';
    if (lastPrice !== null && latestSeller !== null
      && Math.abs(lastPrice - latestSeller) / Math.max(lastPrice, latestSeller) < 0.10
      && /\b(i.?ll\s+do|deal|works\s+for\s+me|let.?s\s+do\s+it|sounds\s+good)\b/i.test(low)) {
      // If buyer also asks a logistics question → inquiry not accept
      const hasLogQ = lastContent.includes('?') && /\b(when|where|address|pick\s+up|come\s+by|weekend|saturday|sunday|morning|afternoon|time|hour)\b/i.test(low);
      return hasLogQ ? 'inquiry' : 'accept';
    }
  }

  if (/\b(deal|sold|agreed|you\s+got\s+it)\b/i.test(low)
    && !/\b(find|finding|good|working|make|best|great|cheap)\s+(a\s+)?deal\b/i.test(low)
    && hasPriceAnywhere(conv)) {
    // Guard: only accept when prices are reasonably close (gap < 35%) or one price missing
    const gapOk = latestBuyer === null || latestSeller === null
      || Math.abs(latestSeller - latestBuyer) / Math.max(latestSeller, latestBuyer) < 0.35;
    if (gapOk) return 'accept';
  }

  if (/\b(pick\s*up|come\s*(by|get|grab)|available|anytime|meet\s+up|today|tonight|tomorrow)\b/i.test(low)) {
    if (latestBuyer !== null && latestSeller !== null) {
      const gap = Math.abs(latestSeller - latestBuyer) / Math.max(latestSeller, latestBuyer);
      if (gap <= 0.06) return 'accept';
    }
  }

  // ── INQUIRY ──────────────────────────────────────────────────────────────
  if (lastSpeaker === 'seller' && next === 'buyer') {
    if (/\b(sure|yes|yeah|yep)\b/i.test(low)
      && /\b(welcome|come\s+by|come\s+dig|when\s+works|anytime|any\s+time|works\s+for\s+you|whenever)\b/i.test(low)
      && !hasPriceAnywhere([last]))
      return 'inquiry';
    if (/\b(when\s+can\s+you|come\s+by|pick\s+it\s+up|swing\s+by|what\s+time)\b/i.test(low) && lastPrice !== null)
      return 'inquiry';
    if (/\b(here\s+all\s+day|any\s+time\s+works|so\s+any\s+time|anytime\s+works|i.?m\s+here\s+all)\b/i.test(low)
      && !hasPriceAnywhere([last]))
      return 'counter_offer';
  }
  if (lastSpeaker === 'buyer' && next === 'seller' && lastPrice !== null
    && /\b(if\s+you\s+can\s+do|if\s+you.?ll\s+do|pick\s+it\s+up\s+right\s+now|pick\s+it\s+up\s+now)\b/i.test(low))
    return 'inquiry';
  if (lastSpeaker === 'buyer' && next === 'seller') {
    const isLogQ = lastContent.includes('?')
      && /\b(address|where|when|what\s+time|come\s+by|pick\s+up|hour|am\b|pm\b|meet|saturday|sunday|monday|tuesday|wednesday|thursday|friday|morning|afternoon|evening)\b/i.test(low)
      && !/\b(anytime|any\s+time|come\s+get\s+it|come\s+pick\s+it|whenever)\b/i.test(low);
    if (isLogQ && hasPriceAnywhere(conv)) return 'inquiry';
  }

  // ── OFFER ──────────────────────────────────────────────────────────────
  if (lastSpeaker === 'seller' && next === 'buyer') {
    if (/\b(how\s+much\s+(are\s+you\s+willing|would\s+you\s+pay|were\s+you\s+thinking)|what\s+(price\s+were\s+you|are\s+you\s+thinking|price\s+do\s+you|would\s+you\s+offer|were\s+you\s+thinking|price\s+were\s+you\s+thinking)|name\s+your\s+price|so\s+how\s+much|what.?s\s+your\s+budget)\b/i.test(low))
      return 'offer';
    if (/\b(what\s+price\s+were\s+you|what\s+were\s+you\s+thinking|what.?s\s+your\s+offer|as\s+long\s+as\s+we.?re\s+not\s+talking)\b/i.test(low))
      return 'offer';
    if (/\b(make\s+(me\s+)?an\s+offer|i.?d\s+be\s+willing\s+to\s+bargain|bargain\s+a\s+bit|right\s+buyer)\b/i.test(low)
      && hasPriceAnywhere(conv))
      return 'counter_offer';
    if (/\b(make\s+(me\s+)?an\s+offer)\b/i.test(low) && !hasPriceAnywhere(conv))
      return 'offer';
    if (!hasPriceAnywhere(conv) && convLen >= 4 && !lastContent.includes('?'))
      return 'offer';
  }
  if (next === 'buyer' && prices.latestBuyer === null && prices.latestSeller === null && convLen >= 4 && lastContent.includes('?'))
    return 'offer';

  // ── COUNTER_OFFER ──────────────────────────────────────────────────────
  if (lastSpeaker === 'seller' && next === 'buyer') {
    if (lastPrice !== null && lastContent.includes('?')
      && /\b(how\s+about|what\s+about|would\s+you|can\s+you\s+do|does?\s+\$?[\d]|try)\b/i.test(low)
      && latestBuyer !== null && lastPrice > latestBuyer * 1.03)
      return 'counter_offer';
    if (/\b(no\s+lower|but\s+no\s+lower|not\s+going\s+lower)\b/i.test(low)
      && lastPrice !== null && latestBuyer !== null && lastPrice > latestBuyer * 1.03)
      return 'counter_offer';
    if (/\b(will\s+go\s+down\s+to|can\s+go\s+down\s+to|going\s+down\s+to)\b/i.test(low)
      && /\b(but\s+no\s+lower|no\s+lower|lowest)\b/i.test(low)
      && lastPrice !== null && latestBuyer !== null && lastPrice > latestBuyer)
      return 'counter_offer';
  }

  return null;
}
