// templates.ts

export function cleanMessage(msg: string): string {
  return msg
    .replace(/<think>[\s\S]*?<\/think>/gi, '')
    .replace(/^(I understand your position[^.]*\.\s*)/i, '')
    .replace(/^(Of course[,!]\s*|Certainly[,!]\s*|Absolutely[,!]\s*)/i, '')
    .replace(/^(BUYER|SELLER|Message|Response|Reply|Output|Next|Buyer|Seller):\s*/i, '')
    .replace(/^[""\u201c\u201d''`]|[""\u201c\u201d''`]$/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

export function fallbackMessage(
  intent: string,
  role: 'buyer' | 'seller',
  price: number | null,
): string {
  const p = price ? `$${price}` : null;
  const t: Record<string, Record<string, string>> = {
    counter_offer: {
      buyer: p ? `That's still high. Would you take ${p}?` : `Can you come down a bit more?`,
      seller: p ? `Best I can do is ${p}.` : `That's too low, need more.`,
    },
    accept: {
      buyer: p ? `Deal, ${p}. When can I pick it up?` : `Works for me. When can I come?`,
      seller: p ? `Deal, ${p}. When can you come by?` : `Deal. When can you pick it up?`,
    },
    reject: {
      buyer: `Can't go higher, sorry. Good luck.`,
      seller: `no problem good luck`,
    },
    offer: {
      buyer: p ? `I was thinking ${p}.` : `What's your best price?`,
      seller: p ? `Asking ${p} for it.` : `Make me an offer.`,
    },
    inquiry: {
      buyer: `Perfect, what's your address? I can be there by 2pm.`,
      seller: `That works, can you come by in an hour?`,
    },
  };
  return t[intent]?.[role] ?? `Let me get back to you.`;
}
