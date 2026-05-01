# Negotiation Playbook — Buyer / Seller Two-Party Bargaining

Domain context for the RAG layer. Patterns below are drawn from marketplace bargaining (Craigslist-style single-item sales across cars, furniture, electronics, housing, bikes, phones). Use them to ground persona reasoning when predicting the next turn.

## Intent signals in the most recent turn

- **`inquiry`** — questions about the product, its condition, provenance, warranty, delivery, or terms. Examples: "How many miles?", "Does it come with the charger?", "Is the title clean?", "Can I pick it up this weekend?", "Can you come by in an hour?".
- **`offer`** — stating a price or set of terms. Can be the opening move by either side, or a side offer ("I'll take it for $X if you deliver"). Includes first-time asks, but not reactions to a prior number.
- **`counter_offer`** — explicitly proposing a different number than the one just stated ("You said $500, I'll do $400"). Always a reaction to a prior `offer` or `counter_offer`. Also covers logistics confirmations that implicitly accept a price range ("Alright, I'll be there around 5").
- **`reject`** — pushing back without proposing a new number. "I can't go any lower.", "That won't work for me.", "I already told you my bottom line.". Often appears as insistence on a previously-stated price. Also covers polite walkaway responses like "no problem good luck".
- **`accept`** — agreeing to the current terms and moving toward closing. "Deal.", "Alright, $X works.", "Let's do it.", "Then I'll take the 1000.", "very good 3000$ it is.". Frequently terse.

## Common arc patterns

- **Info → offer → counter → counter → accept.** The classic 4-step bargain: inquire, opening offer, one or two counters, meet in the middle.
- **Info → offer → reject → counter → accept.** Seller holds firm once, buyer concedes.
- **Info → offer → counter → reject → walkaway.** Deal fails; last turn often reads like `reject` or a curt sign-off.
- **Info-heavy opening.** Some dialogues stretch 4-6 turns of pure `inquiry` before any number is floated. The next turn after a long info stretch is almost always `offer`.
- **Logistics close.** When seller says "come by anytime" or "I'm here all day", the buyer confirms a time. This is often labelled `counter_offer` in the dataset even though it reads like acceptance.

## Textual cues by intent

| Intent | Typical cue words / patterns |
|---|---|
| `inquiry` | "how", "what", "is it", "does it", "when can", "any chance", "?", "can you come by", "in an hour" |
| `offer` | First price mentioned by either side; phrases like "I'm asking", "I'll pay", "would you take", "my offer is", "willing to pay $X" |
| `counter_offer` | "how about $X", "what if I", "meet at $X", "let's split the difference", "lowest I can do is", "I'll be there around [time]", "Alright" confirming logistics |
| `reject` | "can't go", "won't go", "no way", "absolutely not", "final offer", "firm at", "no problem good luck", "Cant go beyond" |
| `accept` | "deal", "done", "works for me", "alright", "sold", "I'll take it", "very good X it is", "you've got yourself a deal", "Then I'll take" |

## Seller vs buyer dynamics

- **Sellers tend to** anchor high on the opening `offer`, concede in small steps, and reject once or twice before accepting.
- **Buyers tend to** open with `inquiry`, counter-offer aggressively, and either accept if the seller moves, or walk away with a `reject` if the seller holds firm.
- **Walkaways** (final turn = `reject`) are most common when the two sides' numbers differ by more than ~30% and neither has moved in the last two turns.
- **Buyer walkaway phrasing**: "look elsewhere", "thanks for your time", "good luck" — these are `reject` from the buyer's side.
- **Seller responding to walkaway**: "no problem good luck" — this is also `reject` sentiment.

## Red flags for a failing negotiation

- Buyer repeats the same number without new justification → next turn likely `reject` or walkaway.
- Seller adds unrelated product features in response to a price push → low trust, may signal vague stalling.
- Either side tells the other "what you paid is irrelevant" → emotional escalation, next turn often `reject`.
- A clear side offer ("$X and I pick up myself") with no price movement → the other side will usually accept or make one final counter.

## Price quanta and language

- Most Craigslist-style negotiations converge in round-number increments ($5, $10, $25, $50, $100). Expect the counter-offer to be roughly halfway between the last two numbers, rounded to a quantum.
- Explicit "final offer" language often precedes acceptance within 1–2 turns.
- "I could come down to $X" almost always signals the seller's floor for this round; the buyer's accept / counter usually follows immediately.
- When both sides are within 10-12% of each other, the next move is almost always `accept` or a tiny `counter_offer`.

## Acceptance patterns — terse and direct

Real Craigslist acceptance messages are SHORT. Examples from actual conversations:
- "very good 3000$ it is." — seller accepting
- "You've got yourself a deal...a great deal! $9000 it is!" — seller accepting enthusiastically
- "Then I'll take the 1000." — seller accepting buyer's offer after logistics confirmed
- "I'd be okay with 60. And yes, I'll pick up today." — buyer accepting seller's price
- "Ok great. Let's meet at 4PM. $15 thanks my man" — buyer confirming meeting and price
- "deal, come pick it up" — seller confirming

## Counter-offer patterns — specific, grounded, brief

Real counter-offers reference the specific item and conversation:
- "You have it listed for $400. So you're already trying to sell it for half, practically." — references listing price
- "I see your point, but brand new you must pay 5X the price. Let's be fair and settle for $45" — references original price
- "I can meet you halfway. 114." — explicit midpoint
- "$4300 for to pick it up" — discount in exchange for pickup
- "285 is too low, I cannot turn a profit. I bought this item for $500. I can go to $300, take it or leave it." — cost justification
- "My rock bottom price is $11,000. If you can see your way to make that great." — sets floor
- "Alright, I'll be there around 5, cya then" — logistics confirmation (labelled counter_offer)
- "I have 11k in cash that I could give you today. I also have 2 Honda 4-wheelers I could throw in." — creative counter with barter

## Rejection patterns — brief, polite or firm

- "no problem good luck" — seller responding to buyer walkaway
- "Cant go beyond $240 sorry" — buyer holding at their stated max
- "I'll look elsewhere, thanks" — buyer walking away

## Offer patterns — first price, specific terms

When seller asks "how much are you willing to pay?" or "make me an offer", the buyer responds with:
- A specific dollar amount: "4800 per month.", "$100 if there are no extra bells or whistles."
- Sometimes with conditions or personal context: "I am willing to pay $900 and sign a two year lease."
- Sometimes creative: "I have 11k in cash and two Honda 4-wheelers."

## Inquiry patterns — logistics after price agreement

When a price is agreed upon but logistics remain:
- "That works, can you come by in an hour?" — seller confirming price and asking logistics
- "When can you come?" — seller asking after acceptance

## Closing zone behaviour

When the price gap is under 12% or explicit closing language appears:
- Next buyer turn → almost always `accept`
- Next seller turn → either `accept` or one final small `counter_offer`
- If seller used "firm" language → possible `reject` if buyer pushes

## Barter and creative offers

Some buyers offer non-cash items alongside cash ("I have cash + two Honda 4-wheelers"). The seller's response is usually `counter_offer` (negotiate the terms) or `accept` (if the overall value works).
