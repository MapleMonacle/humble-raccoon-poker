"""
Tournament bot, built from the poker-prep kit by build_bot.py. Do not edit this
file by hand: edit the modules in poker-prep/ and rebuild.

Sections, in order (each starts with its module's own explanation):
  evaluator  - who has the better hand (card ints, hand scores, fast lookups)
  equity     - Monte Carlo: how often our hand wins against the opponent's range
  decision   - pot odds, EV, thresholds and bet sizing -> fold/check/call/raise
  opponents  - VPIP, PFR, aggression, fold-to-bet, bluff share + exploit rules
  interface  - neutral data types (Observation, Action, ...)
  bot        - EquityBot: model -> ranges -> equity -> decide
  hackathon  - decide(game_state): the tournament adapter, push/fold, safety
No LLMs, no network: standard library only.
"""
# No `from __future__ import annotations` here, on purpose: the tournament
# loads this file without registering it in sys.modules, and dataclasses
# crash on postponed (string) annotations in that situation.


##############################################################################
# evaluator.py
##############################################################################
"""
evaluator.py: variant-agnostic poker hand evaluator (standard library only).

CARD ENCODING
    A card is a small int:   card = rank * 4 + suit
      rank 0..12  means 2,3,4,5,6,7,8,9,T,J,Q,K,A  (always the *global* rank,
                  even in short deck, so "6s" is the same int in every variant)
      suit 0..3   means c,d,h,s
    So "2c" = 0 and "As" = 12*4 + 3 = 51. Ints hash, compare and add quickly.

HAND SCORES
    Evaluator.evaluate(hole, board) returns an int. HIGHER IS BETTER, and two
    equal ints are a split pot. Layout:
        score = category_strength << (4 * hand_size)  |  ranks packed 4 bits each
    category_strength comes from the variant's category_order (weakest first),
    so "flush beats full house" in short deck is only a different ordering.

THE SPEED TRICK (a lazily built lookup table)
    For the usual "best 5-card hand" rules, each card maps to one integer key:
        5**rank  +  1 << (40 + 4*suit)
    Adding up the keys of a hand gives:
      * low 40 bits:  base-5 digits = how many cards of each rank. That number
        identifies the RANK MULTISET, which is all you need for every hand
        except flushes. Its score is computed once and cached in a dict
        (~50k distinct 7-card multisets, so the cache fills fast).
      * high bits:    4 bits per suit = how many cards of that suit. A flush is
        possible only if some suit has >= 5 cards; one add + mask detects it.
    So a typical evaluation is: one sum, one dict lookup, one bit test.

VARIANTS
    Everything about the game lives in a Variant: which ranks/suits are in the
    deck, how many hole cards, how the board is dealt, how many cards make a
    hand, whether you must use N hole cards (Omaha) and the category order.
    Presets: HOLDEM, OMAHA, SHORT_DECK, ROYAL, LEDUC, KUHN, THREE_CARD.
"""


from collections import Counter
from dataclasses import dataclass
from itertools import combinations

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

# Category names, weakest first. Variants pick a subset and an order.
STANDARD_ORDER = ("high_card", "pair", "two_pair", "trips", "straight",
                  "flush", "full_house", "quads", "straight_flush")
# Short deck (6+): flushes are rarer than full houses with 36 cards, so most
# rule sets rank flush ABOVE full house. Some also put trips above straight;
# if the organisers say so, swap those two names.
SHORT_DECK_ORDER = ("high_card", "pair", "two_pair", "trips", "straight",
                    "full_house", "flush", "quads", "straight_flush")
# 3-card poker: with 3 cards a straight is rarer than a flush.
THREE_CARD_ORDER = ("high_card", "pair", "flush", "straight", "trips",
                    "straight_flush")
ALL_CATEGORIES = set(STANDARD_ORDER)


# --------------------------------------------------------------------------
# Card parsing / formatting
# --------------------------------------------------------------------------
def make_card(rank_char: str, suit_char: str) -> int:
    return RANK_CHARS.index(rank_char.upper()) * 4 + SUIT_CHARS.index(suit_char.lower())


def parse_card(text: str) -> int:
    """'As', 'td', '10h', 'Ah' -> int. Accepts '10' for ten."""
    t = text.strip()
    if t[:2] == "10":
        t = "T" + t[2:]
    if len(t) != 2:
        raise ValueError(f"bad card {text!r}")
    return make_card(t[0], t[1])


def parse_cards(text) -> list[int]:
    """'As Kd 7c' or 'AsKd7c' or ['As','Kd'] or [51, 46] -> list of ints."""
    if isinstance(text, (list, tuple)):
        return [c if isinstance(c, int) else parse_card(c) for c in text]
    t = text.replace(",", " ").replace("10", "T").split()
    out = []
    for chunk in t:
        if len(chunk) % 2:
            raise ValueError(f"bad card string {text!r}")
        out.extend(parse_card(chunk[i:i + 2]) for i in range(0, len(chunk), 2))
    return out


def card_str(card: int) -> str:
    return RANK_CHARS[card >> 2] + SUIT_CHARS[card & 3]


def cards_str(cards) -> str:
    return " ".join(card_str(c) for c in cards)


# --------------------------------------------------------------------------
# Variant description
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    name: str = "holdem"
    ranks: str = RANK_CHARS              # ranks in the deck, low to high
    suits: str = SUIT_CHARS              # suits in the deck (subset of "cdhs")
    hole_cards: int = 2                  # private cards per player
    street_cards: tuple = (0, 3, 1, 1)   # board cards dealt before each betting round
    hand_size: int = 5                   # cards in a made hand
    hole_used: tuple | None = None       # (min, max) hole cards the hand must use; None = any
    category_order: tuple = STANDARD_ORDER
    ace_low_straight: bool = True        # A-2-3-4-5 (or A-6-7-8-9 in short deck)

    def __post_init__(self):
        bad = set(self.category_order) - ALL_CATEGORIES
        if bad:
            raise ValueError(f"unknown categories {bad}")
        if not set(self.ranks) <= set(RANK_CHARS) or not set(self.suits) <= set(SUIT_CHARS):
            raise ValueError("ranks/suits must come from '23456789TJQKA' / 'cdhs'")

    @property
    def board_cards(self) -> int:
        return sum(self.street_cards)

    def deck(self) -> list[int]:
        return [make_card(r, s) for r in self.ranks for s in self.suits]


HOLDEM = Variant()
OMAHA = Variant("omaha", hole_cards=4, hole_used=(2, 2))
SHORT_DECK = Variant("short_deck", ranks="6789TJQKA", category_order=SHORT_DECK_ORDER)
ROYAL = Variant("royal", ranks="TJQKA")
LEDUC = Variant("leduc", ranks="JQK", suits="hs", hole_cards=1, street_cards=(0, 1),
                hand_size=2, category_order=("high_card", "pair"), ace_low_straight=False)
KUHN = Variant("kuhn", ranks="JQK", suits="s", hole_cards=1, street_cards=(0,),
               hand_size=1, category_order=("high_card",), ace_low_straight=False)
THREE_CARD = Variant("three_card", hole_cards=3, street_cards=(0,), hand_size=3,
                     category_order=THREE_CARD_ORDER)
PRESETS = {v.name: v for v in (HOLDEM, OMAHA, SHORT_DECK, ROYAL, LEDUC, KUHN, THREE_CARD)}

# Per-card key for the fast path (see module docstring).
_SUIT_SHIFT = 40
_RANK_KEY_MASK = (1 << _SUIT_SHIFT) - 1
_KEY = [5 ** (c >> 2) + (1 << (_SUIT_SHIFT + 4 * (c & 3))) for c in range(52)]


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------
class Evaluator:
    def __init__(self, variant: Variant = HOLDEM):
        v = variant
        self.variant = v
        self.hand_size = v.hand_size
        self._shift = 4 * v.hand_size
        # strength 1 = weakest category; categories missing from the order
        # simply do not exist in this game (e.g. no straights in Leduc).
        self._strength = {name: i + 1 for i, name in enumerate(v.category_order)}
        self._straights = self._build_straights()
        self._has_flush = ("flush" in self._strength or "straight_flush" in self._strength) \
            and v.hand_size >= 3
        # Fast path = best 5 of up to 12 cards (12 keeps suit counts in 4 bits).
        self._fast = v.hand_size == 5
        self._cache: dict[int, int] = {}

    # ---------------- public API ----------------
    def evaluate(self, hole, board=()) -> int:
        """Score of the best legal hand for these hole + board cards."""
        v = self.variant
        if v.hole_used is None:
            return self.rank(list(hole) + list(board))
        lo, hi = v.hole_used
        k = v.hand_size
        best = -1
        for h in range(lo, hi + 1):
            b = k - h
            if h > len(hole) or b < 0 or b > len(board):
                continue
            board_combos = list(combinations(board, b))
            if not self._fast:
                for hc in combinations(hole, h):
                    for bc in board_combos:
                        s = self.rank(hc + bc)
                        if s > best:
                            best = s
                continue
            # Fast Omaha path: pre-add the keys of each hole subset and each
            # board subset once, then every 5-card hand is a single addition.
            cache = self._cache
            board_keys = [(bc, sum(_KEY[c] for c in bc)) for bc in board_combos]
            for hc in combinations(hole, h):
                hk = sum(_KEY[c] for c in hc)
                for bc, bk in board_keys:
                    total = hk + bk
                    s = cache.get(total & _RANK_KEY_MASK)
                    if s is None:
                        s = cache[total & _RANK_KEY_MASK] = self._nonflush_score(hc + bc)
                    if self._has_flush and ((total >> _SUIT_SHIFT) + 0x3333) & 0x8888:
                        fs = self._flush_score(hc + bc, total >> _SUIT_SHIFT)
                        if fs > s:
                            s = fs
                    if s > best:
                        best = s
        if best < 0:
            raise ValueError("not enough cards to make a legal hand")
        return best

    def rank(self, cards) -> int:
        """Best hand from ANY hand_size cards among `cards` (no hole-card rules)."""
        n = len(cards)
        if self._fast and 5 <= n <= 12:
            total = sum(map(_KEY.__getitem__, cards))
            score = self._cache.get(total & _RANK_KEY_MASK)
            if score is None:
                score = self._nonflush_score(cards)
                self._cache[total & _RANK_KEY_MASK] = score
            suit_counts = total >> _SUIT_SHIFT
            # +3 per 4-bit field sets the field's top bit iff the count is >= 5
            if self._has_flush and (suit_counts + 0x3333) & 0x8888:
                fs = self._flush_score(cards, suit_counts)
                if fs > score:
                    score = fs
            return score
        return self._rank_slow(cards)

    def ranker_for_board(self, board):
        """Return rank(a, b) for repeated evaluations against one fixed board."""
        public = list(board)
        if not self._fast or len(public) != 5:
            return lambda a, b: self.rank([a, b] + public)

        keys = _KEY
        board_key = sum(map(keys.__getitem__, public))
        public_suits = board_key >> _SUIT_SHIFT
        can_flush = self._has_flush and any(
            ((public_suits >> shift) & 15) >= 3 for shift in (0, 4, 8, 12)
        )
        cache = self._cache
        nonflush = self._nonflush_score
        flush = self._flush_score

        def rank_pair(a, b):
            total = board_key + keys[a] + keys[b]
            rank_key = total & _RANK_KEY_MASK
            score = cache.get(rank_key)
            flush_possible = can_flush and (((total >> _SUIT_SHIFT) + 0x3333) & 0x8888)
            if score is not None and not flush_possible:
                return score
            cards = [a, b] + public
            if score is None:
                score = cache[rank_key] = nonflush(cards)
            if flush_possible:
                score = max(score, flush(cards, total >> _SUIT_SHIFT))
            return score

        return rank_pair

    def category(self, score: int) -> str:
        return self.variant.category_order[(score >> self._shift) - 1]

    def describe(self, score: int) -> str:
        ranks = []
        for i in range(self.hand_size - 1, -1, -1):
            ranks.append(RANK_CHARS[(score >> (4 * i)) & 0xF])
        cat = self.category(score)
        if cat in ("straight", "straight_flush"):
            return f"{cat} ({ranks[0]}-high)"
        return f"{cat} ({' '.join(ranks)})"

    # ---------------- reference implementation ----------------
    def score_exact(self, cards) -> int:
        """Score of EXACTLY hand_size cards. Slow but obviously correct; the
        tests check the fast path against this on thousands of random hands."""
        k = len(cards)
        ranks = [c >> 2 for c in cards]
        counts = Counter(ranks)
        # ranks ordered by (how many, rank): e.g. K K K 7 7 or Q Q 9 5 2
        groups = sorted(counts.items(), key=lambda rc: (rc[1], rc[0]), reverse=True)
        kickers = [r for r, n in groups for _ in range(n)]
        pattern = [n for _, n in groups] + [0, 0]
        is_flush = k >= 3 and len({c & 3 for c in cards}) == 1
        mask = 0
        for r in ranks:
            mask |= 1 << r
        straight_high = self._straight_high(mask) if len(counts) == k and k >= 3 else -1
        if straight_high >= 0 and (mask.bit_count() != k):
            straight_high = -1

        cands = [("high_card", sorted(ranks, reverse=True))]
        if straight_high >= 0:
            cands.append(("straight", [straight_high]))
            if is_flush:
                cands.append(("straight_flush", [straight_high]))
        if is_flush:
            cands.append(("flush", sorted(ranks, reverse=True)))
        if pattern[0] >= 4:
            cands.append(("quads", kickers))
        if pattern[0] >= 3 and pattern[1] >= 2:
            cands.append(("full_house", kickers))
        if pattern[0] >= 3:
            cands.append(("trips", kickers))
        if pattern[0] >= 2 and pattern[1] >= 2:
            cands.append(("two_pair", kickers))
        if pattern[0] >= 2:
            cands.append(("pair", kickers))
        return max(self._pack(cat, ks) for cat, ks in cands)

    def _rank_slow(self, cards) -> int:
        k = self.hand_size
        if len(cards) < k:
            raise ValueError(f"need at least {k} cards, got {len(cards)}")
        if len(cards) == k:
            return self.score_exact(cards)
        return max(self.score_exact(c) for c in combinations(cards, k))

    # ---------------- helpers ----------------
    def _pack(self, category: str, ranks) -> int:
        """category + ranks -> int. Returns 0 if the category does not exist."""
        s = self._strength.get(category, 0)
        if not s:
            return 0
        ranks = list(ranks)[:self.hand_size]
        for r in ranks:
            s = (s << 4) | r
        return s << (4 * (self.hand_size - len(ranks)))

    def _build_straights(self):
        """List of (rank_mask, high_rank) for every straight, best first."""
        v = self.variant
        k = v.hand_size
        if k < 3 or not ({"straight", "straight_flush"} & set(v.category_order)):
            return []
        vals = sorted(RANK_CHARS.index(r) for r in v.ranks)
        out = []
        for i in range(len(vals) - k + 1):
            window = vals[i:i + k]
            out.append((sum(1 << r for r in window), window[-1]))
        if v.ace_low_straight and 12 in vals and len(vals) >= k:
            low = [12] + vals[:k - 1]
            if len(set(low)) == k:
                out.append((sum(1 << r for r in low), vals[k - 2]))
        out.sort(key=lambda mh: mh[1], reverse=True)
        return out

    def _straight_high(self, mask: int) -> int:
        for smask, high in self._straights:
            if mask & smask == smask:
                return high
        return -1

    def _nonflush_score(self, cards) -> int:
        """Best 5-card hand ignoring suits (cached by rank multiset)."""
        counts = [0] * 13
        for c in cards:
            counts[c >> 2] += 1
        all_desc = [r for r in range(12, -1, -1) for _ in range(counts[r])]
        distinct = [r for r in range(12, -1, -1) if counts[r]]

        def rest(used, n):
            """Highest n ranks left after removing the `used` cards."""
            left = list(used)
            out = []
            for r in all_desc:
                if r in left:
                    left.remove(r)
                elif len(out) < n:
                    out.append(r)
            return out

        best = self._pack("high_card", all_desc[:5])
        mask = 0
        for r in distinct:
            mask |= 1 << r
        sh = self._straight_high(mask)
        if sh >= 0:
            best = max(best, self._pack("straight", [sh]))
        quads = [r for r in distinct if counts[r] >= 4]
        trips = [r for r in distinct if counts[r] >= 3]
        pairs = [r for r in distinct if counts[r] >= 2]
        if quads:
            q = quads[0]
            best = max(best, self._pack("quads", [q] * 4 + rest([q] * 4, 1)))
        if trips:
            t = trips[0]
            other = [r for r in pairs if r != t]
            if other:
                best = max(best, self._pack("full_house", [t] * 3 + [other[0]] * 2))
            best = max(best, self._pack("trips", [t] * 3 + rest([t] * 3, 2)))
        if len(pairs) >= 2:
            p1, p2 = pairs[0], pairs[1]
            used = [p1, p1, p2, p2]
            best = max(best, self._pack("two_pair", used + rest(used, 1)))
        if pairs:
            p = pairs[0]
            best = max(best, self._pack("pair", [p, p] + rest([p, p], 3)))
        return best

    def _flush_score(self, cards, suit_counts: int) -> int:
        best = 0
        for s in range(4):
            if (suit_counts >> (4 * s)) & 0xF >= 5:
                mask = 0
                for c in cards:
                    if c & 3 == s:
                        mask |= 1 << (c >> 2)
                sh = self._straight_high(mask)
                if sh >= 0:
                    best = max(best, self._pack("straight_flush", [sh]))
                top5 = [r for r in range(12, -1, -1) if mask >> r & 1][:5]
                best = max(best, self._pack("flush", top5))
        return best


_DEFAULT = {}


def get_evaluator(variant: Variant = HOLDEM) -> Evaluator:
    """Shared evaluator per variant (keeps one warm cache)."""
    ev = _DEFAULT.get(variant)
    if ev is None:
        ev = _DEFAULT[variant] = Evaluator(variant)
    return ev


##############################################################################
# equity.py
##############################################################################
"""
equity.py: Monte Carlo equity estimator (standard library only).

"Equity" = the share of the pot you win on average if all remaining cards are
dealt out and nobody folds: wins count 1, a 2-way split counts 1/2, etc.

How it works (one TRIAL):
    1. give each opponent hole cards (uniform random, from a weighted range,
       or "the best of m random hands" for a player who is showing strength),
    2. deal the rest of the board at random from the unseen cards,
    3. evaluate everyone, record our share of the pot.
Average over many trials. The standard error shrinks like 1/sqrt(trials):
    500 trials  -> about +-2.2% equity,   5000 trials -> about +-0.7%.

The loop stops at whichever comes first: max_iterations or time_budget_ms, so a
bot can never blow an action timeout. On the river against one random hand the
number of possibilities is small, so it enumerates them exactly instead.

CLI:
    python equity.py "As Ks" --board "Qs Js 2c" --opps 2 --ms 200
    python equity.py --bench
"""


import bisect
import math
import random
import time
from dataclasses import dataclass
from itertools import combinations



@dataclass
class EquityResult:
    equity: float      # expected pot share in [0, 1]
    win: float         # fraction of trials won outright
    tie: float         # fraction of trials split
    iterations: int
    stderr: float      # standard error of `equity`
    exact: bool = False

    def __str__(self):
        tag = "exact" if self.exact else f"+-{1.96 * self.stderr:.3f}"
        return (f"equity {self.equity:.3f} ({tag}), win {self.win:.3f}, "
                f"tie {self.tie:.3f}, {self.iterations} trials")


# --------------------------------------------------------------------------
# Cheap hand-strength heuristics (used to model ranges, not to pick actions)
# --------------------------------------------------------------------------
def preflop_score(hole) -> float:
    """Rough, variant-agnostic pre-board strength: higher = better.
    High cards count, the top card double; pairs get a big bonus; suited and
    connected cards a small one. Good enough to rank hands into 'top X%'."""
    ranks = sorted((c >> 2 for c in hole), reverse=True)
    score = 2.0 * ranks[0] + sum(ranks[1:])
    seen = {}
    for r in ranks:
        seen[r] = seen.get(r, 0) + 1
    for r, n in seen.items():
        if n >= 2:
            score += (16 + r) * (n - 1)
    suits = [c & 3 for c in hole]
    if len(hole) > 1 and max(suits.count(s) for s in set(suits)) >= 2:
        score += 2
    distinct = sorted(seen)
    score += sum(1 for a, b in zip(distinct, distinct[1:]) if b - a == 1)
    return score


def current_strength(ev: Evaluator, hole, board) -> float:
    """How strong `hole` is RIGHT NOW: made-hand score if the board allows
    a full hand, otherwise the preflop heuristic. Only compare like with like
    (all candidates on the same board)."""
    if board and len(hole) + len(board) >= ev.hand_size:
        try:
            return ev.evaluate(hole, board)
        except ValueError:
            pass
    return preflop_score(hole)


def preflop_threshold(variant: Variant, top_fraction: float, samples: int = 4000,
                      seed: int = 0) -> float:
    """preflop_score value that only the top `top_fraction` of hands reach.
    Works for any variant because it samples real hands from its deck."""
    rng = random.Random(seed)
    deck = variant.deck()
    scores = sorted(preflop_score(rng.sample(deck, variant.hole_cards)) for _ in range(samples))
    idx = min(len(scores) - 1, max(0, int((1.0 - top_fraction) * len(scores))))
    return scores[idx]


# --------------------------------------------------------------------------
# Ranges: ways to deal an opponent's hole cards
# --------------------------------------------------------------------------
CAP_WEIGHT = 0.25     # how likely a "capped" monster still is, relative to other hands


class TopFraction:
    """Opponent's range as two simple filters:
      1. preflop_fraction: only the top X% of starting hands (by preflop_score)
         - "he raised preflop and only raises 8% of hands";
      2. fraction: of those, only the top Y% by strength on the CURRENT board
         - "and he bets this flop only 30% of the time, so he has the best 30%".

      3. bluff_share: a bluffer's aggression is not all strong hands. With
         this probability we deal ANY hand, skipping both filters ("his bets
         tell me little"). Measured from showdowns (opponents.py).
      4. cap: "range capping". A player who only checks has usually NOT got a
         monster (most players bet those). The strongest `cap` share of hands,
         judged on the board AS IT WAS WHEN THEY CHECKED (`cap_board`), is made
         CAP_WEIGHT (4x) less likely, not impossible. A hand that only became
         strong on a later card (a flush on the river) is not touched.

    prepare() runs once per decision: it lists every hole-card combo the
    opponent could have (or a random sample of `pool` combos when there are
    too many, e.g. Omaha), applies the filters, and sample() just picks
    from what is left. Filter 2 ranks made hands and ignores draws, which is
    right against passive players; bluff_share covers the aggressive ones."""

    def __init__(self, fraction: float = 1.0, preflop_fraction: float = 1.0,
                 bluff_share: float = 0.0, pool: int = 2500, cap: float = 0.0,
                 cap_board=None):
        self.fraction = max(1e-3, min(1.0, fraction))
        self.preflop_fraction = max(1e-3, min(1.0, preflop_fraction))
        self.bluff_share = max(0.0, min(1.0, bluff_share))
        self.cap = max(0.0, min(0.5, cap))
        self.cap_board = cap_board
        self.capped = []
        self.pool = pool
        self.hands = None
        self.wide = None

    @property
    def is_random(self) -> bool:
        return self.fraction >= 0.999 and self.preflop_fraction >= 0.999 and self.cap <= 0

    def prepare(self, rng, deck, hole_n, ev, board):
        if math.comb(len(deck), hole_n) <= self.pool:
            combos = list(combinations(deck, hole_n))
        else:
            combos = [tuple(rng.sample(deck, hole_n)) for _ in range(self.pool)]
        self.wide = combos
        if self.preflop_fraction < 0.999:
            combos.sort(key=preflop_score, reverse=True)
            combos = combos[:max(1, math.ceil(self.preflop_fraction * len(combos)))]
        self.capped = []
        if self.cap > 0:                          # monsters a checker rarely has
            check_board = self.cap_board if self.cap_board is not None else board
            combos.sort(key=lambda h: current_strength(ev, h, check_board), reverse=True)
            k = min(len(combos) - 1, int(self.cap * len(combos)))
            self.capped, combos = combos[:k], combos[k:]
        if self.fraction < 0.999:
            combos.sort(key=lambda h: current_strength(ev, h, board), reverse=True)
            combos = combos[:max(1, math.ceil(self.fraction * len(combos)))]
        self.hands = combos

    def sample(self, rng, available, hole_n, ev, board):
        if self.hands is None:
            return rng.sample(available, hole_n)
        pick_from = self.hands
        if self.bluff_share and rng.random() < self.bluff_share:
            pick_from = self.wide
        elif self.capped:
            w = CAP_WEIGHT * len(self.capped)
            if rng.random() < w / (w + len(self.hands)):
                pick_from = self.capped
        avail = set(available)
        for _ in range(30):
            h = pick_from[rng.randrange(len(pick_from))]
            if all(c in avail for c in h):
                return list(h)
        return rng.sample(available, hole_n)     # every candidate blocked: give up

    def __repr__(self):
        return (f"TopFraction({self.fraction:.2f}, preflop {self.preflop_fraction:.2f}, "
                f"bluff {self.bluff_share:.2f}, cap {self.cap:.2f})")


class WeightedRange:
    """Explicit list of hole-card combos with weights, e.g. from a dataset or
    from holdem_range('QQ+,AKs'). Conflicting combos (cards already used) are
    rejected and redrawn; if every draw conflicts we fall back to random."""

    def __init__(self, combos_weights):
        items = list(combos_weights.items()) if isinstance(combos_weights, dict) else list(combos_weights)
        self.combos = [tuple(c) for c, w in items if w > 0]
        self.cum = []
        total = 0.0
        for _, w in items:
            if w > 0:
                total += w
                self.cum.append(total)
        if not self.combos:
            raise ValueError("empty range")

    def sample(self, rng, available, hole_n, ev, board):
        avail = set(available)
        for _ in range(50):
            combo = rng.choices(self.combos, cum_weights=self.cum, k=1)[0]
            if all(c in avail for c in combo):
                return list(combo)
        return rng.sample(available, hole_n)

    def __len__(self):
        return len(self.combos)


def holdem_range(text: str) -> WeightedRange:
    """Hold'em shorthand -> WeightedRange. Supports 'AA', 'QQ+', 'AKs', 'AKo',
    'AK' (suited+offsuit), 'ATs+' (ATs..AKs), '22-66' style is not supported."""
    combos = {}
    suits = "cdhs"

    def add_pair(r):
        for a, b in combinations(suits, 2):
            combos[(make_card(r, a), make_card(r, b))] = 1.0

    def add_two(hi, lo, kind):
        for a in suits:
            for b in suits:
                if (kind == "s" and a != b) or (kind == "o" and a == b):
                    continue
                combos[(make_card(hi, a), make_card(lo, b))] = 1.0

    for tok in text.replace(" ", "").split(","):
        if not tok:
            continue
        plus = tok.endswith("+")
        tok = tok.rstrip("+").upper().replace("S", "s").replace("O", "o")
        hi, lo, kind = tok[0], tok[1], tok[2:] or "so"
        if hi == lo:
            start = RANK_CHARS.index(hi)
            for r in (RANK_CHARS[start:] if plus else hi):
                add_pair(r)
            continue
        if RANK_CHARS.index(lo) > RANK_CHARS.index(hi):
            hi, lo = lo, hi
        los = RANK_CHARS[RANK_CHARS.index(lo):RANK_CHARS.index(hi)] if plus else lo
        for l in los:
            for k in kind:
                add_two(hi, l, k)
    return WeightedRange(combos)


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------
class ConditionalResponse:
    """Reuse heads-up showdown trials to price responses to each bet size."""

    def __init__(self, trials, scorer, fold_curve, reraise_curve, factor, deadline):
        self.trials, self.scorer = trials, scorer
        self.fold_curve, self.reraise_curve = fold_curve, reraise_curve
        self.factor, self.deadline = factor, deadline
        self.counts, self.sums = [0], [0.0]
        self.ready = None

    def prepare(self):
        score = self.scorer()
        if score is None:
            return False
        strengths, groups = {}, {}
        for i, (a, b, share) in enumerate(self.trials):
            if i % 16 == 0 and time.perf_counter() >= self.deadline:
                return False
            combo = (a, b) if a < b else (b, a)
            if combo not in strengths:
                strengths[combo] = score(combo)
            group = groups.setdefault(strengths[combo], [0, 0.0])
            group[0] += 1
            group[1] += share
        for strength in sorted(groups):
            count, total = groups[strength]
            self.counts.append(self.counts[-1] + count)
            self.sums.append(self.sums[-1] + total)
        return time.perf_counter() < self.deadline

    def total_at(self, count):
        i = bisect.bisect_right(self.counts, count) - 1
        if count == self.counts[i]:
            return self.sums[i]
        width = self.counts[i + 1] - self.counts[i]
        return self.sums[i] + (count - self.counts[i]) / width * (self.sums[i + 1] - self.sums[i])

    def __call__(self, fraction, can_reraise):
        if self.ready is None:
            self.ready = self.prepare()
        if not self.ready:
            return None
        n = self.counts[-1]
        fold = self.fold_curve(fraction)
        reraise = (1 - fold) * self.reraise_curve(fraction) if can_reraise else 0.0
        # Stronger hands generally continue. Mix in unfiltered trials to
        # allow weak calls and bluff raises, and to smooth thin samples.
        lower, higher = n * fold, n * (1 - reraise)
        mid, top = self.total_at(higher), self.sums[-1]
        raw = top / n
        called = self.conditioned(mid - self.total_at(lower), higher - lower, raw)
        raised = self.conditioned(top - mid, n - higher, raw)
        return called, reraise, raised

    def conditioned(self, wins, count, raw):
        sharpened = (wins + 8 * raw) / (count + 8)
        return max(0.0, min(1.0, self.factor * (0.75 * sharpened + 0.25 * raw)))


def estimate_equity(hole, board=(), evaluator: Evaluator | None = None, num_opponents: int = 1,
                    opp_ranges=None, dead=(), time_budget_ms: float | None = 50.0,
                    max_iterations: int = 200_000, min_iterations: int = 32,
                    rng=None, exact_threshold: int = 2000, heads_up_trials=None) -> EquityResult:
    """Equity of `hole` given `board`, against `num_opponents` hands.

    opp_ranges: None (all random) or a list with one entry per opponent, each
                None / TopFraction / WeightedRange.
    dead:       cards known to be out of play (e.g. folded cards you saw).
    time_budget_ms: stop after this long (None = only max_iterations counts).
    heads_up_trials: optional list for receiving (opponent card, opponent card, pot share).
    """
    ev = evaluator or get_evaluator(HOLDEM)
    v = ev.variant
    rng = rng or random
    hole = list(parse_cards(hole) if isinstance(hole, str) else hole)
    board = list(parse_cards(board) if isinstance(board, str) else board)
    if num_opponents < 1:
        return EquityResult(1.0, 1.0, 0.0, 0, 0.0, True)
    ranges = list(opp_ranges) if opp_ranges is not None else [None] * num_opponents
    if len(ranges) != num_opponents:
        raise ValueError("need one range entry per opponent")
    ranges = [None if isinstance(r, TopFraction) and r.is_random else r for r in ranges]

    # The clock starts now, so range preparation counts against the budget too.
    deadline = None if time_budget_ms is None else time.perf_counter() + time_budget_ms / 1000.0
    known = set(hole) | set(board) | set(dead)
    deck = [c for c in v.deck() if c not in known]
    need = v.board_cards - len(board)
    hn = v.hole_cards
    if need < 0:
        raise ValueError("board has more cards than the variant deals")
    if len(deck) < need + hn * num_opponents:
        raise ValueError("not enough cards left in the deck")

    # Exact enumeration on a complete board vs one random hand.
    if need == 0 and num_opponents == 1 and ranges[0] is None \
            and math.comb(len(deck), hn) <= exact_threshold:
        return _exact_vs_one(ev, hole, board, deck, heads_up_trials)

    for r in ranges:
        if r is not None and hasattr(r, "prepare"):
            r.prepare(rng, deck, hn, ev, board)
    simple = v.hole_used is None             # can use ev.rank on the concatenation
    all_random = all(r is None for r in ranges)
    k_draw = need + hn * num_opponents
    total = total_sq = 0.0
    wins = ties = 0
    it = 0
    rank = ev.rank
    evaluate = ev.evaluate
    while it < max_iterations:
        if deadline is not None and it >= min_iterations and (it & 15) == 0 \
                and time.perf_counter() > deadline:
            break
        if all_random:
            draw = rng.sample(deck, k_draw)
            full = board + draw[:need]
            opps = [draw[need + i * hn: need + (i + 1) * hn] for i in range(num_opponents)]
        else:
            used = set()
            opps = []
            for r in ranges:
                avail = [c for c in deck if c not in used] if used else deck
                h = rng.sample(avail, hn) if r is None else r.sample(rng, avail, hn, ev, board)
                used.update(h)
                opps.append(h)
            avail = [c for c in deck if c not in used]
            full = board + rng.sample(avail, need)

        if simple:
            hero = rank(hole + full)
            best = -1
            n_best = 0
            for h in opps:
                s = rank(h + full)
                if s > best:
                    best, n_best = s, 1
                elif s == best:
                    n_best += 1
        else:
            hero = evaluate(hole, full)
            best = -1
            n_best = 0
            for h in opps:
                s = evaluate(h, full)
                if s > best:
                    best, n_best = s, 1
                elif s == best:
                    n_best += 1

        if hero > best:
            share = 1.0
            wins += 1
        elif hero == best:
            share = 1.0 / (n_best + 1)
            ties += 1
        else:
            share = 0.0
        total += share
        total_sq += share * share
        if heads_up_trials is not None and num_opponents == 1:
            heads_up_trials.append((*opps[0], share))
        it += 1

    mean = total / it
    var = max(0.0, total_sq / it - mean * mean)
    return EquityResult(mean, wins / it, ties / it, it, math.sqrt(var / it))


def _exact_vs_one(ev, hole, board, deck, heads_up_trials=None):
    hero = ev.evaluate(hole, board)
    win = tie = n = 0
    for opp in combinations(deck, ev.variant.hole_cards):
        s = ev.evaluate(opp, board)
        n += 1
        if hero > s:
            win += 1
        elif hero == s:
            tie += 1
        if heads_up_trials is not None:
            heads_up_trials.append((*opp, 1.0 if hero > s else 0.5 if hero == s else 0.0))
    eq = (win + tie / 2) / n
    return EquityResult(eq, win / n, tie / n, n, 0.0, exact=True)


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
def benchmark(seconds: float = 1.0) -> None:
    rng = random.Random(42)

    def bench_eval(label, variant, n_hole, n_board):
        ev = Evaluator(variant)
        deck = variant.deck()
        hands = [rng.sample(deck, n_hole + n_board) for _ in range(20000)]
        hands = [(h[:n_hole], h[n_hole:]) for h in hands]
        for h, b in hands:                      # warm the cache once
            ev.evaluate(h, b)
        count = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            for h, b in hands[:2000]:
                ev.evaluate(h, b)
            count += 2000
        rate = count / (time.perf_counter() - t0)
        print(f"  {label:<34} {rate:>12,.0f} hands/s")

    print("Hand evaluation (warm cache):")
    bench_eval("Hold'em 7 cards", HOLDEM, 2, 5)
    bench_eval("Short deck 7 cards", SHORT_DECK, 2, 5)
    bench_eval("Omaha 4+5 (60 combos each)", OMAHA, 4, 5)

    print("Equity trials:")
    ev = Evaluator(HOLDEM)
    for label, hole, board, opps, ranges in [
        ("HU preflop vs random", "As Kd", "", 1, None),
        ("HU flop vs random", "As Kd", "Qs Jh 2c", 1, None),
        ("3-way preflop vs random", "As Kd", "", 3, None),
        ("HU flop vs TopFraction(0.25)", "As Kd", "Qs Jh 2c", 1, [TopFraction(0.25)]),
    ]:
        r = estimate_equity(hole, board, ev, opps, ranges, time_budget_ms=seconds * 1000,
                            max_iterations=10**9, rng=rng)
        print(f"  {label:<34} {r.iterations / seconds:>12,.0f} trials/s   ({r})")
    print("Rule of thumb: trials in a 50 ms budget = trials/s / 20.")


##############################################################################
# decision.py
##############################################################################
"""
decision.py: the readable decision core.

Input:  a Spot (equity + the chip situation), Params (our tunable thresholds)
        and Adjustments (what the opponent model wants changed).
Output: an Action: fold / check / call / raise (a bet is a raise from 0),
        with a human-readable `reason`.

Every rule here is one you can say out loud:

  1. POT ODDS. Calling costs `to_call` to win `pot + to_call`, so a call breaks
     even when equity == to_call / (pot + to_call).
  2. EV.  EV(fold) = 0.
          EV(call) = equity * (pot + to_call) - to_call
          EV(bet B) = P(fold) * pot + P(call) * (eq_called * (pot + 2B) - B)
     ("eq_called" < equity because people call with their better hands.)
     These are ONE-STREET approximations: they assume the hand goes to a
     showdown after the call, and ignore raises, later betting and how much
     of our equity we actually realise. They are exact for a river call or an
     all-in, and a rule of thumb earlier in the hand.
  3. STRENGTH. Equity alone is misleading multi-way (30% vs 3 players is
     above average). We normalise it: strength = (equity - fair) / (1 - fair),
     fair = 1 / players. 0 = average hand, 1 = the nuts. Thresholds use this,
     so the same numbers work heads-up and 6-handed.
  4. RULES, in order:
     a. strength >= value bar      -> bet/raise for value (bigger when stronger);
        facing a bet, re-raise only with a much stronger hand, else call.
     b. a bluff/steal is +EV -> take it some of the time (bluff_freq), but
        only as the first bet: never bluff-raise into a real bet.
     c. no bet to face             -> check.
     d. equity beats pot odds by a margin -> call, else fold.
  5. SIZING. If the opponent model has learned how often this opponent folds
     to small / medium / large / overbet sizes, try each size and keep the one
     with the best EV (best_size). Otherwise use the fixed rule below.
     A bet/raise is a fraction of the pot after we call:
        raise_to = current_bet + fraction * (pot + to_call)
     fraction slides from min_bet_fraction (just strong) to max_bet_fraction
     (the nuts), then is clamped to the legal min/max.
"""


import random
from dataclasses import dataclass, field


@dataclass
class Action:
    kind: str            # "fold" | "check" | "call" | "raise"  ("raise" from 0 = a bet)
    raise_to: int = 0    # raise: our TOTAL bet on this street after the action
    amount: int = 0      # chips we add to the pot with this action
    reason: str = ""

    def __str__(self):
        if self.kind == "raise":
            return f"raise to {self.raise_to} (+{self.amount})  [{self.reason}]"
        if self.kind == "call":
            return f"call {self.amount}  [{self.reason}]"
        return f"{self.kind}  [{self.reason}]"


@dataclass
class Spot:
    """Everything the decision needs, in chips. The adapter fills this in."""
    equity: float          # 0..1 share of the pot vs the opponents still in
    pot: int               # chips in the middle INCLUDING every bet made this street
    to_call: int           # chips needed to call (0 = we can check)
    my_bet: int            # chips we already put in on this street
    current_bet: int       # the highest total bet on this street
    stack: int             # chips we still have behind
    big_blind: int
    min_raise_to: int      # smallest legal raise-to (0 = raising not allowed)
    max_raise_to: int      # largest legal raise-to (all-in in no-limit)
    num_opponents: int = 1
    street: int = 0        # 0 = preflop / before any board card
    facing_bet: bool = False   # True if to_call comes from a real bet/raise (not just blinds)
    opponent_max_to: int | None = None

    def raise_chips(self, raise_to: int) -> tuple[int, int]:
        """Return our matched contribution and the opponent's call cost."""
        if self.num_opponents == 1 and self.opponent_max_to is not None:
            raise_to = min(raise_to, self.opponent_max_to)
        return raise_to - self.my_bet, raise_to - self.current_bet


@dataclass
class Params:
    """All tunable numbers in one place (sweep them with sim/tune.py)."""
    value_strength: float = 0.30     # strength needed to bet/raise for value
    reraise_strength: float = 0.60   # strength needed to raise when facing a bet
    call_margin: float = 0.03        # equity cushion above pot odds before calling
    bluff_freq: float = 0.10         # how often we take a +EV bluff/steal by default
    called_equity_factor: float = 0.70   # our equity when a bluff gets called, vs raw
    min_bet_fraction: float = 0.50   # bet size (fraction of pot) when just strong enough
    max_bet_fraction: float = 1.00   # bet size with the nuts
    bluff_fraction: float = 0.50     # bluff size, fraction of pot
    shove_ratio: float = 0.75        # if the bet would be >= this share of our stack, go all in


@dataclass
class Adjustments:
    """Changes requested by the opponent model (opponents.py). Neutral by default."""
    value_margin: float = 0.0        # + = need a stronger hand to value bet; - = bet thinner
    call_margin: float = 0.0         # + = call tighter; - = call looser
    bluff_freq: float | None = None  # override Params.bluff_freq
    bet_size_mult: float = 1.0       # scale value-bet sizes
    fold_prob: float = 0.40          # chance everyone folds to a bet (for bluff EV)
    fold_curve: object = None        # optional f(bet as fraction of pot) -> fold chance
    trap: bool = False               # check strong-but-not-huge hands to let a bluffer bet
    notes: list = field(default_factory=list)
    response_curve: object = None   # optional (size, can_reraise) -> response probabilities/equities


# ---------------------------------------------------------------- formulas
def pot_odds(to_call: float, pot: float) -> float:
    """Equity needed for a call to break even."""
    return to_call / (pot + to_call) if to_call > 0 else 0.0


def ev_call(equity: float, pot: float, to_call: float) -> float:
    return equity * (pot + to_call) - to_call


def heads_up_call_pot(pot: int, owed: int, cost: int) -> int:
    """Pot before calling, excluding the part of the bet the caller can't match."""
    return pot - (owed - cost)


def ev_bet(fold_prob: float, pot: float, add: float, opp_call: float, equity_called: float) -> float:
    """EV of putting `add` more chips in, if opponents fold with fold_prob and
    otherwise call `opp_call` and we win `equity_called` of the bigger pot."""
    return fold_prob * pot + (1 - fold_prob) * (equity_called * (pot + add + opp_call) - add)


def normalized_strength(equity: float, num_opponents: int) -> float:
    fair = 1.0 / (num_opponents + 1)
    return (equity - fair) / (1.0 - fair)


def raise_size(spot: Spot, fraction: float, shove_ratio: float = 0.75) -> int:
    """raise_to for a bet of `fraction` x (pot after calling), clamped to the legal range."""
    target = spot.current_bet + fraction * (spot.pot + spot.to_call)
    target = int(round(max(spot.min_raise_to, min(spot.max_raise_to, target))))
    all_in_to = spot.my_bet + spot.stack
    # Don't leave a silly stub behind: if we'd commit most of our stack, shove.
    if spot.max_raise_to >= all_in_to and (target - spot.my_bet) >= shove_ratio * spot.stack:
        target = spot.max_raise_to
    return target


SIZE_CHOICES = (0.33, 0.5, 0.75, 1.0, 1.5)


def best_size(spot: Spot, p: Params, a: Adjustments, value: bool):
    """Try each bet size and keep the one with the best EV, using what we have
    learned about how often THIS opponent folds to each size (a.fold_curve).
      value bet: the more often they fold, the stronger the hands that call,
                 so our equity when called shrinks:  eq x (1 - 0.5 x fold%)
      bluff:     when called we have eq x called_equity_factor
    Returns (ev, raise_to, fold_chance)."""
    eq = spot.equity
    cands = {raise_size(spot, f, p.shove_ratio) for f in SIZE_CHOICES}
    if value and spot.stack <= 3 * (spot.pot + spot.to_call):
        cands.add(spot.max_raise_to)                      # shallow: shoving is a real option
    best = None
    for to in sorted(cands):
        if to <= spot.current_bet:
            continue
        add, opp_call = spot.raise_chips(to)
        frac = opp_call / max(1, spot.pot + spot.to_call)
        f = a.fold_curve(frac)
        eq_called = eq * (1 - 0.5 * f) if value else eq * p.called_equity_factor
        reply = None
        extra = 0
        if a.response_curve is not None and spot.opponent_max_to is not None:
            next_total = min(to + max(opp_call, spot.big_blind), spot.my_bet + spot.stack,
                             spot.opponent_max_to)
            extra = max(0, next_total - to)
            reply = a.response_curve(frac, extra > 0)
            if reply is not None:
                eq_called, reraises, eq_reraised = reply
        # Variance check: bigger-than-pot value bets only with a big edge.
        # Against weaker bots we win the match by grinding, not by flipping.
        if value and frac > 1.05 and eq_called < 0.75:
            continue
        ev = ev_bet(f, spot.pot, add, opp_call, eq_called)
        if reply is not None and reraises > 0:
            # Price a minimum re-raise. After that we can fold or pay to see
            # the remaining cards, using equity against the re-raising range.
            called_pot = spot.pot + add + opp_call
            called = eq_called * called_pot - add
            reraised = max(-add, eq_reraised * (called_pot + 2 * extra) - add - extra)
            ev += reraises * (reraised - called)
        if best is None or ev > best[0]:
            best = (ev, to, f)
    return best


def _raise(spot: Spot, raise_to: int, reason: str) -> Action:
    return Action("raise", raise_to, raise_to - spot.my_bet, reason)


# ---------------------------------------------------------------- decision
def decide_action(spot: Spot, params: Params | None = None, adj: Adjustments | None = None,
           rng=random) -> Action:
    p = params or Params()
    a = adj or Adjustments()
    eq = spot.equity
    strength = normalized_strength(eq, spot.num_opponents)
    call_cost = min(spot.to_call, spot.stack)            # all-in for less if short
    call_pot = spot.pot
    if spot.num_opponents == 1:
        call_pot = heads_up_call_pot(call_pot, spot.to_call, call_cost)
    odds = pot_odds(call_cost, call_pot)
    can_raise = spot.min_raise_to > 0 and spot.max_raise_to > spot.current_bet \
        and spot.stack > call_cost
    value_bar = p.value_strength + a.value_margin
    info = f"eq {eq:.2f} str {strength:+.2f} odds {odds:.2f}"

    # a. Value: a strong hand wants chips in the pot.
    #    Policy: a strong hand CALLS a bet without the extra call margins
    #    (Adjustments.call_margin). Those margins are there to stop marginal
    #    calls; folding a strong hand would throw away positive-EV calls.
    if strength >= value_bar:
        if a.trap and spot.to_call == 0 and strength < p.reraise_strength + a.value_margin:
            return Action("check", reason=f"trap: let them bluff; {info}")
        if can_raise and (spot.to_call == 0 or strength >= p.reraise_strength + a.value_margin):
            if a.fold_curve is not None:                  # we know how they react to sizes
                pick = best_size(spot, p, a, value=True)
                if pick is not None:
                    return _raise(spot, pick[1], f"value, EV-best size (fold {pick[2]:.0%}); {info}")
            # Slide size from min to max fraction as strength goes value_bar -> 1.
            t = (strength - value_bar) / max(1e-9, 1.0 - value_bar)
            frac = (p.min_bet_fraction + t * (p.max_bet_fraction - p.min_bet_fraction)) * a.bet_size_mult
            return _raise(spot, raise_size(spot, frac, p.shove_ratio), f"value; {info}")
        if spot.to_call == 0:
            return Action("check", reason=f"strong but can't bet; {info}")
        return Action("call", amount=call_cost, reason=f"strong, call; {info}")

    # b. Bluff / steal: only when the fold equity makes it profitable, and only
    #    as the FIRST bet (or a raise of the blinds). Re-raising a real bet
    #    without a hand is how bots set fire to their stack.
    bluff_freq = p.bluff_freq if a.bluff_freq is None else a.bluff_freq
    if can_raise and bluff_freq > 0 and not spot.facing_bet:
        pick = best_size(spot, p, a, value=False) if a.fold_curve is not None else None
        if pick is not None:
            ev_b, to = pick[0], pick[1]
        else:
            to = raise_size(spot, p.bluff_fraction, p.shove_ratio)
            add, opp_call = spot.raise_chips(to)
            ev_b = ev_bet(a.fold_prob, spot.pot, add, opp_call, eq * p.called_equity_factor)
        ev_c = max(0.0, ev_call(eq, call_pot, call_cost)) if spot.to_call else 0.0
        if ev_b > ev_c and rng.random() < bluff_freq:
            return _raise(spot, to, f"bluff (EV {ev_b:.0f} > {ev_c:.0f}); {info}")

    # c. Free card.
    if spot.to_call == 0:
        return Action("check", reason=f"check; {info}")

    # d. Facing a bet with a medium/weak hand: the pot-odds test.
    if eq >= odds + p.call_margin + a.call_margin and ev_call(eq, call_pot, call_cost) > 0:
        return Action("call", amount=call_cost, reason=f"pot odds; {info}")
    return Action("fold", reason=f"fold; {info}")


# Short alias (the tests and docs call it decide). The tournament file defines
# its own decide(game_state), so the bot itself always calls decide_action.
decide = decide_action


##############################################################################
# opponents.py
##############################################################################
"""
opponents.py: opponent modelling from observed actions, plus exploit rules.

Feed it every action you see (yours can be skipped):
    model.start_hand(player_ids)
    model.observe(pid, street, kind, facing_bet)   # kind: fold/check/call/bet/raise
    model.end_hand()

Stats (classic HUD numbers):
    VPIP        % of hands the player voluntarily put chips in preflop (call or raise;
                posting a blind or checking the big blind does not count)
    PFR         % of hands the player raised preflop
    AF          aggression factor = (bets + raises) / calls, postflop
    agg_freq    (bets + raises) / all postflop actions: "how often do they bet?"
    fold_to_bet % of times they folded when facing a bet or raise
    bluff_share of the times they bet/raised postflop AND showed down, how
                often their hand was in the bottom half of all possible hands
                on that board (bot.showed_weak). Honest ~0, random ~0.5.

Small samples lie, so every stat is SMOOTHED toward a population prior:
    stat = (observed_count + prior * W) / (opportunities + W)
After W (default 8) opportunities the data and the prior weigh the same; after
more hands, recent data dominates. Counts fade by STAT_DECAY each finished hand;
one opportunity per hand has an effective-count limit of 49. Replace PRIORS
with numbers mined from the organisers'
dataset on the day.

Exploit rules (opponents.py -> decision.Adjustments):
    calling station (low fold_to_bet)  -> value-bet thinner and bigger, never bluff
    folder (high fold_to_bet)          -> bluff/steal much more
    maniac (bets a lot)                -> call wider (their bets mean less)
    rock (tight + passive)             -> when they DO bet, believe them: we compute
                                          equity against a narrow range (range_read)
    bluffer (high bluff_share)         -> that share of their betting range is
                                          modelled as "any hand", so we call down wider
                                          (and hackathon.py checks to them out of position)
    gives up on the turn               -> after betting the flop they rarely bet the
                                          turn: when they check it, bet (hackathon.py)
    checks                             -> range capping: a checker rarely has a monster
"""


from dataclasses import dataclass, field


# Population priors: what we assume before seeing a player act.
PRIORS = {"vpip": 0.40, "pfr": 0.20, "agg_freq": 0.30, "af": 1.5, "fold_to_bet": 0.45,
          "bluff_share": 0.25}
PRIOR_WEIGHT = 8.0
STAT_DECAY = 0.98  # Per finished hand: half-life about 34 hands; effective cap 49.

# Bet sizes are measured as a fraction of the pot after calling (the same
# yardstick decision.raise_size uses). Buckets: small < 0.4, medium < 0.8,
# large < 1.5, overbet/shove >= 1.5. Before we have data for a bucket we
# assume people fold more to bigger bets: fold_to_bet x these multipliers.
SIZE_EDGES = (0.4, 0.8, 1.5)
SIZE_FOLD_MULT = (0.8, 1.0, 1.2, 1.4)
SIZE_PRIOR_WEIGHT = 4.0


def size_bucket(frac: float) -> int:
    for i, edge in enumerate(SIZE_EDGES):
        if frac < edge:
            return i
    return len(SIZE_EDGES)


class ShoveReads:
    """Learn jam frequencies by effective depth and whether they faced a raise."""

    def __init__(self):
        self.counts = {}

    def commit(self, opponent, events):
        records = self.counts.setdefault(opponent, {})
        for context, (seen, jams) in records.items():
            records[context] = (seen * STAT_DECAY, jams * STAT_DECAY)
        credited = set()
        for event in events:
            context = event.shove_context
            if context is None or context in credited:
                continue
            credited.add(context)
            seen, jams = records.get(context, (0.0, 0.0))
            records[context] = (seen + 1, jams + event.shoved)

    def read(self, opponent, event):
        if not event.shoved:
            return None
        seen, jams = self.counts.get(opponent, {}).get(event.shove_context, (0.0, 0.0))
        rate = (jams + 1) / (seen + 2)
        # Frequency is a reliable widening signal for habitual shovers.
        # Selective jams need the ordinary card-strength read.
        if seen < 3 or rate < 0.65:
            return None
        fraction = min(1.0, 0.02 + 1.12 * rate)
        return fraction, 0.85 * seen / (seen + 4)


@dataclass
class PlayerStats:
    hands: int = 0
    vpip_hands: int = 0
    pfr_hands: int = 0
    post_bets: int = 0        # postflop bets + raises
    post_calls: int = 0
    post_passive: int = 0     # postflop checks + folds
    faced_bet: int = 0
    folded_to_bet: int = 0
    showdowns_after_bet: int = 0
    bluffs_shown: int = 0
    jam_showdowns: int = 0    # showdowns after they moved all-in postflop
    jam_bluffs: int = 0       # ... with a weak hand
    turn_chances: int = 0     # bet the flop, then first to act unopposed on the turn
    turn_barrels: int = 0     # ... and bet again
    open_chances: int = 0     # first preflop decision with no raise to face
    open_shoves: int = 0      # ... and they raised all-in
    reshove_chances: int = 0  # faced a preflop raise
    reshoves: int = 0         # ... and raised all-in over it
    size_faced: list = field(default_factory=lambda: [0, 0, 0, 0])   # our bets, by size bucket
    size_folded: list = field(default_factory=lambda: [0, 0, 0, 0])
    size_raised: list = field(default_factory=lambda: [0, 0, 0, 0])

    @staticmethod
    def _smooth(count, n, prior, w=PRIOR_WEIGHT):
        return (count + prior * w) / (n + w)

    @property
    def vpip(self):
        return self._smooth(self.vpip_hands, self.hands, PRIORS["vpip"])

    @property
    def pfr(self):
        return self._smooth(self.pfr_hands, self.hands, PRIORS["pfr"])

    @property
    def agg_freq(self):
        n = self.post_bets + self.post_calls + self.post_passive
        return self._smooth(self.post_bets, n, PRIORS["agg_freq"])

    @property
    def af(self):
        return (self.post_bets + PRIORS["af"] * 2) / (self.post_calls + 2)

    @property
    def fold_to_bet(self):
        return self._smooth(self.folded_to_bet, self.faced_bet, PRIORS["fold_to_bet"])

    @property
    def bluff_share(self):
        return self._smooth(self.bluffs_shown, self.showdowns_after_bet, PRIORS["bluff_share"], 6)

    @property
    def jam_bluff_share(self):
        """Bluff share of their postflop ALL-INS only, smoothed toward their
        overall bluff share: a bot can bet honestly and still jam as a bluff,
        or bluff small and jam only with the nuts."""
        return self._smooth(self.jam_bluffs, self.jam_showdowns, self.bluff_share, 4)

    @property
    def shove_rate(self):
        """Share of preflop chances where they raised ALL-IN (opens + reshoves;
        an all-in that only calls doesn't count)."""
        return self._smooth(self.open_shoves + self.reshoves,
                            self.open_chances + self.reshove_chances, 0.05, 4)

    @property
    def open_shove_rate(self):
        return self._smooth(self.open_shoves, self.open_chances, 0.05, 4)

    @property
    def reshove_rate(self):
        return self._smooth(self.reshoves, self.reshove_chances, 0.05, 4)

    @property
    def give_up_rate(self):
        """After betting the flop, how often they do NOT bet the turn."""
        return 1.0 - self._smooth(self.turn_barrels, self.turn_chances, 0.5, 4)

    @property
    def check_cap(self):
        """Range capping: share of the strongest hands to remove when they only
        check. The more they like betting, the less a check hides a monster
        (passive players slow-play, aggressive ones bet their big hands)."""
        return min(0.15, 0.3 * self.agg_freq)

    def fold_at(self, frac: float) -> float:
        """How often they fold when we bet `frac` of the pot: what we have SEEN
        at that size, blended with (overall fold_to_bet x size multiplier)."""
        b = size_bucket(frac)
        prior = min(0.95, max(0.02, self.fold_to_bet * SIZE_FOLD_MULT[b]))
        return (self.size_folded[b] + prior * SIZE_PRIOR_WEIGHT) / (self.size_faced[b] + SIZE_PRIOR_WEIGHT)

    def reraise_at(self, frac: float) -> float:
        """Chance of a raise among responses that continued, smoothed by size."""
        b = size_bucket(frac)
        continued = sum(self.size_faced) - sum(self.size_folded)
        prior = self._smooth(sum(self.size_raised), continued, 0.05, 6)
        return self._smooth(self.size_raised[b], self.size_faced[b] - self.size_folded[b],
                            prior, SIZE_PRIOR_WEIGHT)

    def archetype(self) -> str:
        """Which common bot design does this look like? (see ARCHETYPES)"""
        if self.hands < 6:
            return "unknown"
        if self.open_chances + self.reshove_chances >= 4 and self.shove_rate > 0.6:
            return "shover"
        if self.fold_to_bet < 0.2 and self.agg_freq < 0.3:
            return "calling-station"
        if self.showdowns_after_bet >= 3 and self.bluff_share > 0.45:
            return "bluffer"
        small = self.size_faced[0] + self.size_faced[1]
        big = self.size_faced[2] + self.size_faced[3]
        if small >= 4 and big >= 4:
            f_small = (self.size_folded[0] + self.size_folded[1]) / small
            f_big = (self.size_folded[2] + self.size_folded[3]) / big
            if f_big - f_small > 0.35:
                return "size-folder"
        if self.pfr > 0.55 and self.agg_freq > 0.45:
            return "maniac"
        if self.vpip < 0.3 and self.agg_freq < 0.35:
            return "rock"
        if self.fold_to_bet > 0.6:
            return "folder"
        return "balanced"

    def label(self) -> str:
        """Human label, handy in logs: tight/loose + passive/aggressive."""
        if self.fold_to_bet < 0.25:
            return "calling-station"
        tl = "tight" if self.vpip < 0.25 else "loose"
        pa = "aggressive" if self.agg_freq > 0.40 else "passive"
        return f"{tl}-{pa}"

    def __str__(self):
        return (f"hands {self.hands} VPIP {self.vpip:.2f} PFR {self.pfr:.2f} "
                f"AF {self.af:.1f} agg {self.agg_freq:.2f} F2B {self.fold_to_bet:.2f} "
                f"bluff {self.bluff_share:.2f} shove {self.shove_rate:.2f} "
                f"[{self.label()} / {self.archetype()}]")


class OpponentModel:
    def __init__(self):
        self.stats: dict[str, PlayerStats] = {}
        self._in_hand: list[str] = []
        self._vpip: set[str] = set()
        self._pfr: set[str] = set()

    def get(self, pid) -> PlayerStats:
        s = self.stats.get(pid)
        if s is None:
            s = self.stats[pid] = PlayerStats()
        return s

    # ---------------- feeding the model ----------------
    def start_hand(self, player_ids):
        if self._in_hand:              # previous hand never closed: close it now
            self.end_hand()
        self._in_hand = list(player_ids)
        self._vpip.clear()
        self._pfr.clear()

    def observe(self, pid, street: int, kind: str, facing_bet: bool):
        """kind: 'fold' | 'check' | 'call' | 'bet' | 'raise' (anything else is ignored)."""
        s = self.get(pid)
        if facing_bet and kind in ("fold", "call", "raise"):
            s.faced_bet += 1
            if kind == "fold":
                s.folded_to_bet += 1
        if street == 0:
            if kind in ("call", "bet", "raise"):
                self._vpip.add(pid)
            if kind in ("bet", "raise"):
                self._pfr.add(pid)
        else:
            if kind in ("bet", "raise"):
                s.post_bets += 1
            elif kind == "call":
                s.post_calls += 1
            elif kind in ("check", "fold"):
                s.post_passive += 1

    def observe_showdown(self, pid, bet_postflop: bool, weak: bool):
        """Call for each opponent who showed cards. weak = air (see bot.showed_weak)."""
        if bet_postflop:
            s = self.get(pid)
            s.showdowns_after_bet += 1
            s.bluffs_shown += weak

    def observe_jam_showdown(self, pid, weak: bool):
        """They moved all-in postflop and showed down: was the hand weak?"""
        s = self.get(pid)
        s.jam_showdowns += 1
        s.jam_bluffs += weak

    def observe_response(self, pid, frac: float, folded: bool, raised: bool = False):
        """We bet `frac` of the pot at them; how did they respond?"""
        s = self.get(pid)
        b = size_bucket(frac)
        s.size_faced[b] += 1
        s.size_folded[b] += folded
        s.size_raised[b] += raised

    def observe_preflop(self, pid, open_chance: bool, opened: bool,
                        reshove_chance: bool, reshoved: bool):
        """Once per hand, from the replayed hand (hackathon.MatchMemory.replay):
        open_chance = their first preflop decision faced no raise, opened = they
        raised all-in then; reshove_chance = they faced a raise, reshoved = they
        raised all-in over it."""
        s = self.get(pid)
        s.open_chances += open_chance
        s.open_shoves += opened
        s.reshove_chances += reshove_chance
        s.reshoves += reshoved

    def observe_turn_barrel(self, pid, barreled: bool):
        """They bet the flop and acted first, unopposed, on the turn: did they bet again?"""
        s = self.get(pid)
        s.turn_chances += 1
        s.turn_barrels += barreled

    def end_hand(self):
        if not self._in_hand:
            return  # Repeated end_hand calls must not fade the same hand twice.
        for pid in self._in_hand:
            s = self.get(pid)
            s.hands += 1
            s.vpip_hands += pid in self._vpip
            s.pfr_hands += pid in self._pfr
        for s in self.stats.values():
            for name in ("hands", "vpip_hands", "pfr_hands", "post_bets", "post_calls",
                         "post_passive", "faced_bet", "folded_to_bet", "showdowns_after_bet",
                         "bluffs_shown", "turn_chances", "turn_barrels", "open_chances",
                         "open_shoves", "reshove_chances", "reshoves",
                         "jam_showdowns", "jam_bluffs"):
                setattr(s, name, getattr(s, name) * STAT_DECAY)
            for counts in (s.size_faced, s.size_folded, s.size_raised):
                for i in range(len(counts)):
                    counts[i] *= STAT_DECAY
        self._in_hand = []

    # ---------------- using the model ----------------
    def range_read(self, pid, raised_preflop: bool, called_preflop: bool,
                   bet_this_street: bool, called_postflop_bet: bool = False
                   ) -> tuple[float, float, float]:
        """What this player can hold, given what they did THIS hand, as the
        three numbers equity.TopFraction(now, preflop, bluff_share) needs:
          preflop: raised preflop -> top PFR% of starting hands,
                   called -> top VPIP%, otherwise 1.0 (anything);
          now:     bet/raised on this postflop street -> the top agg_freq
                   share by strength on the board; called a postflop bet ->
                   the top (1 - fold_to_bet) share (calls narrow ranges too);
                   otherwise 1.0;
          bluff_share: how much of that betting/calling range is really air."""
        s = self.get(pid)
        pre = s.pfr if raised_preflop else s.vpip if called_preflop else 1.0
        now = 1.0
        if called_postflop_bet:
            now = 1.0 - s.fold_to_bet
        if bet_this_street:
            now = min(now, s.agg_freq)
        return max(0.03, min(1.0, pre)), max(0.03, min(1.0, now)), s.bluff_share

    def adjustments(self, opp_ids, calls_this_hand: dict | None = None) -> Adjustments:
        """Exploit rules vs the opponents still in the hand.
        calls_this_hand: pid -> how many bets they have already called this
        hand postflop. Each call means they liked their hand, so we assume
        they fold less: fold chance = fold_to_bet ** (1 + calls)."""
        adj = Adjustments()
        if not opp_ids:
            return adj
        calls_this_hand = calls_this_hand or {}
        stats = [self.get(p) for p in opp_ids]
        fold_all = 1.0
        for pid, s in zip(opp_ids, stats):
            # a bluff needs EVERYONE to fold
            fold_all *= s.fold_to_bet ** (1 + calls_this_hand.get(pid, 0))
        adj.fold_prob = fold_all
        # Fold chance as a function of OUR bet size (everyone must fold).
        calls = dict(calls_this_hand)

        def fold_curve(frac, _ids=tuple(opp_ids)):
            p = 1.0
            for pid in _ids:
                p *= self.get(pid).fold_at(frac) ** (1 + calls.get(pid, 0))
            return p
        adj.fold_curve = fold_curve
        adj.notes.append("archetype: " + ", ".join(self.get(p).archetype() for p in opp_ids))
        min_fold = min(s.fold_to_bet for s in stats)
        max_agg = max(s.agg_freq for s in stats)

        # Rule 1: someone never folds -> no bluffs; bet thinner and bigger for value.
        if min_fold < 0.30:
            adj.value_margin -= 0.08
            adj.bet_size_mult *= 1.25
            adj.bluff_freq = 0.0
            adj.notes.append("station: thin value, no bluffs")
        # Rule 2: they fold too much -> bluff / steal a lot.
        elif fold_all > 0.55:
            adj.bluff_freq = 0.70
            adj.notes.append("folder: bluff more")
        elif fold_all > 0.45:
            adj.bluff_freq = 0.35
            adj.notes.append("fairly foldy: bluff a bit more")
        # Rule 3: maniac or proven bluffer -> their bets mean less, call wider.
        bluffer = any(s.archetype() == "bluffer" for s in stats)
        if bluffer:
            adj.call_margin -= 0.05
            adj.notes.append("bluffer: call wider")
        elif max_agg > 0.45:
            adj.call_margin -= 0.04
            adj.notes.append("maniac: call wider")
        # Rule 4: rocks are handled by range_fraction (equity vs a narrow range).
        return adj


##############################################################################
# interface.py
##############################################################################
"""
interface.py: the neutral data types our bots speak.

On the day the starter repo will hand your bot its own kind of game state.
Write ONE adapter function that turns it into an `Observation`, and one that
turns our `Action` into whatever the engine wants. Everything else (equity,
decision, opponent model, the simulator) only ever sees these types.
"""


from dataclasses import dataclass, field


__all__ = ["Action", "ActionEvent", "HandStart", "Observation", "HandResult", "Bot"]


@dataclass
class ActionEvent:
    player_id: str
    seat: int
    street: int        # 0 = preflop
    kind: str          # "post" | "fold" | "check" | "call" | "bet" | "raise"
    amount: int        # chips this action added to the pot
    raise_to: int      # the player's total bet on this street afterwards
    facing_bet: bool   # did they face a voluntary bet/raise when acting?


@dataclass
class HandStart:
    player_id: str          # who YOU are
    seat: int
    button: int             # seat of the dealer button
    seat_ids: list          # player id in each seat
    hole: tuple
    stacks: list            # stacks per seat at the start of the hand (before blinds)
    small_blind: int
    big_blind: int


@dataclass
class Observation:
    player_id: str
    seat: int
    hole: tuple
    board: tuple
    street: int                 # 0 = preflop
    pot: int                    # all chips in the middle, INCLUDING this street's bets
    to_call: int
    my_bet: int                 # my chips in on this street
    current_bet: int            # highest bet this street
    min_raise_to: int           # 0 if I may not raise
    max_raise_to: int
    stack: int                  # chips I still have behind
    big_blind: int
    active_opponents: list      # ids of opponents who have not folded
    history: list               # ActionEvents of this hand so far
    seat_ids: list = field(default_factory=list)
    button: int = 0
    opponent_max_to: int | None = None  # heads-up: what the opponent can match this street

    @property
    def can_check(self) -> bool:
        return self.to_call == 0


@dataclass
class HandResult:
    deltas: dict          # player id -> chips won (negative = lost)
    board: tuple
    shown: dict           # player id -> hole cards, for players who reached showdown


class Bot:
    """Base class. Only act() is required."""
    name = "bot"

    def new_hand(self, info: HandStart) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def observe(self, event: ActionEvent) -> None:
        pass

    def end_hand(self, result: HandResult) -> None:
        pass


##############################################################################
# bot.py
##############################################################################
"""
bot.py: EquityBot = equity.py + opponents.py + decision.py.

Per decision:
  1. For each opponent still in, ask the model how wide their range is given
     what they did this hand (raised preflop? bet this street?) -> TopFraction.
  2. Monte Carlo our equity against those ranges, within the time budget.
  3. Ask the model for exploit Adjustments vs those opponents.
  4. decision.decide_action() turns equity + chips into fold/check/call/raise.

Construct with a Variant so it plays any game the evaluator can score.
"""


import math
import random
import time
from itertools import combinations



def showed_weak(ev, hole, board, samples: int = 400) -> bool:
    """Did they bet with a weak hand? 'Weak' = in the bottom half of all
    hands anyone could hold on this final board. A player whose bets are
    honest almost never shows one; a random bettor shows one half the time."""
    try:
        score = ev.evaluate(hole, board)
    except ValueError:
        return False
    known = set(hole) | set(board)
    deck = [c for c in ev.variant.deck() if c not in known]
    hn = len(hole)
    if math.comb(len(deck), hn) <= samples:
        others = combinations(deck, hn)
    else:
        rng = random.Random(score)                  # deterministic per hand
        others = (rng.sample(deck, hn) for _ in range(samples))
    beaten = total = 0
    for other in others:
        s = ev.evaluate(other, board)
        total += 1
        beaten += score > s or (score == s) * 0.5
    return total > 0 and beaten / total < 0.5


class EquityBot(Bot):
    name = "equity"
    response_scorer = None

    def __init__(self, variant: Variant = HOLDEM, params: Params | None = None,
                 time_budget_ms: float | None = 40.0, max_iterations: int = 3000,
                 use_model: bool = True, seed: int | None = None, verbose: bool = False):
        self.ev = get_evaluator(variant)
        self.params = params or Params()
        self.time_budget_ms = time_budget_ms
        self.max_iterations = max_iterations
        self.use_model = use_model
        self.model = OpponentModel()
        self.rng = random.Random(seed)
        self.verbose = verbose
        self.me = None
        self.hand_events: list[ActionEvent] = []

    # ---------------- event hooks ----------------
    def new_hand(self, info: HandStart) -> None:
        self.me = info.player_id
        self.hand_events = []
        self.model.start_hand([p for p in info.seat_ids if p != info.player_id])

    def observe(self, e: ActionEvent) -> None:
        self.hand_events.append(e)
        if e.player_id != self.me and e.kind != "post":
            self.model.observe(e.player_id, e.street, e.kind, e.facing_bet)

    def end_hand(self, result: HandResult) -> None:
        # Showdown = free information: did the players who bet have it?
        for pid, hole in result.shown.items():
            if pid == self.me:
                continue
            bet_post = any(e.player_id == pid and e.street > 0 and e.kind in ("bet", "raise")
                           for e in self.hand_events)
            if bet_post:
                self.model.observe_showdown(pid, True, showed_weak(self.ev, hole, result.board))
        self.model.end_hand()

    # ---------------- the decision ----------------
    def opponent_ranges(self, obs: Observation):
        """One TopFraction (or None = random) per active opponent."""
        if not self.use_model:
            return None
        ranges = []
        for pid in obs.active_opponents:
            raised = called = bet_now = called_post = checked_post = bet_post = False
            for e in obs.history:
                if e.player_id != pid:
                    continue
                if e.street == 0 and e.kind in ("bet", "raise"):
                    raised = True
                elif e.street == 0 and e.kind == "call":
                    called = True
                elif e.street > 0 and e.kind == "call":
                    called_post = True
                elif e.street > 0 and e.kind == "check":
                    checked_post = True
                if e.street > 0 and e.kind in ("bet", "raise"):
                    bet_post = True
                if e.street == obs.street and obs.street > 0 and e.kind in ("bet", "raise"):
                    bet_now = True
            pre, now, bluff = self.model.range_read(pid, raised, called, bet_now, called_post)
            # Range capping: no bets after the flop, and a check -> probably no
            # monster, judged on the board as it was at their LAST check.
            cap, cap_board = 0.0, None
            if checked_post and not bet_post:
                last_check = max(e.street for e in obs.history
                                 if e.player_id == pid and e.street > 0 and e.kind == "check")
                cap = self.model.get(pid).check_cap
                cap_board = tuple(obs.board[:2 + last_check])
            ranges.append(TopFraction(now, pre, bluff, cap=cap, cap_board=cap_board)
                          if min(pre, now) < 0.95 or cap > 0 else None)
        return ranges

    def tweak(self, adj: Adjustments, obs: Observation, spot: Spot) -> Adjustments:
        """Hook for game-specific changes to the exploit settings (see hackathon.py)."""
        return adj

    def act(self, obs: Observation) -> Action:
        n_opp = max(1, len(obs.active_opponents))
        budget = self.time_budget_ms
        trials = None
        if (self.response_scorer is not None and self.use_model and n_opp == 1
                and obs.opponent_max_to is not None and obs.min_raise_to > 0
                and budget is not None and budget >= 75):
            trials = []
            deadline = time.perf_counter() + budget / 1000
            budget -= min(35.0, 0.15 * budget)
        eq = estimate_equity(obs.hole, obs.board, self.ev, n_opp,
                             opp_ranges=self.opponent_ranges(obs),
                             time_budget_ms=budget,
                             max_iterations=self.max_iterations, rng=self.rng,
                             heads_up_trials=trials)
        # A "real" bet = anything above the blinds preflop, any bet postflop.
        facing_bet = obs.to_call > 0 and (obs.street > 0 or obs.current_bet > obs.big_blind)
        spot = Spot(equity=eq.equity, pot=obs.pot, to_call=obs.to_call, my_bet=obs.my_bet,
                    current_bet=obs.current_bet, stack=obs.stack, big_blind=obs.big_blind,
                    min_raise_to=obs.min_raise_to, max_raise_to=obs.max_raise_to,
                    num_opponents=n_opp, street=obs.street, facing_bet=facing_bet,
                    opponent_max_to=obs.opponent_max_to)
        if self.use_model:
            calls = {}
            for e in obs.history:
                if e.street > 0 and e.kind == "call":
                    calls[e.player_id] = calls.get(e.player_id, 0) + 1
            adj = self.model.adjustments(obs.active_opponents, calls)
        else:
            adj = Adjustments()
        adj = self.tweak(adj, obs, spot)
        if trials is not None and len(trials) >= 96 and adj.fold_curve is not None:
            stat = self.model.get(obs.active_opponents[0])
            factor = spot.equity / eq.equity if eq.equity else 1.0
            adj.response_curve = ConditionalResponse(
                trials, lambda: self.response_scorer(obs.board, deadline), adj.fold_curve,
                stat.reraise_at, factor, deadline)
        action = decide_action(spot, self.params, adj, self.rng)
        if self.verbose:
            print(f"[{self.me}] st{obs.street} pot {obs.pot} call {obs.to_call} -> {action} "
                  f"{'; '.join(adj.notes)}")
        return action


##############################################################################
# reverse.py
##############################################################################
"""
reverse.py: "reverse Monte Carlo". Guess the opponent's cards, and how they
play, from what they have done this hand.

For each possible holding, estimate the equity the opponent would see against
one random hand. Replay the pot, stacks and legal sizes they faced, and estimate
the chip value a player with that equity might assign to each choice. For every
candidate holding H and policy P:

    weight(H, P)  ~  prior(P)  x  product over their actions this hand of
                                  P(action and size | equity, pot, stacks, P)

"H's equity at that moment" is H's equity against a random hand on the board
AS IT WAS when they acted, so later cards never leak into earlier decisions.
Our own hole cards are NOT dead cards in that calculation (the opponent can't
see them), but the opponent can't hold them, so they are removed from the
candidates.

Two kinds of uncertainty are kept apart:
  - HOW they play (the policy): PolicyBelief, learned across hands. Each
    finished hand is counted once, showdowns count most (we then know the
    cards), and old hands slowly fade so a bot that changes style is noticed.
  - WHAT they hold now: RevHand.infer(), recomputed from the whole hand at
    every decision, always starting from the start-of-hand policy belief, so
    the same action is never counted twice.

"unknown" spreads its weight over legal actions and raise sizes. When the
opponent doesn't look like any equity-threshold bot, "unknown" wins and the
bot leans on its ordinary range model (TopFraction) instead.

All thresholds, noise levels and shares below are starting guesses. They
still need calibrating against the arena.
"""

import bisect
import math
import random
import time
from itertools import combinations
from operator import add as _reverse_add


# Policies differ in their preferred equity for raising and in how much of an
# edge they want before committing chips. A higher margin also penalizes risking
# more of their remaining stack. None is the unknown policy.
REV_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
REV_MARGINS = (0.00, 0.04, 0.08, 0.12)
REV_POLICIES = [(t, m) for t in REV_THRESHOLDS for m in REV_MARGINS] + [None]
REV_UNKNOWN_PRIOR = 0.30   # starting belief that they are not an equity bot
REV_NOISE = 0.06           # how soft the thresholds are (their Monte Carlo is noisy too)
REV_FLOOR = 0.15           # every policy keeps this much "any action" (a mismatch isn't proof)
REV_LEARN_CANDIDATES = 200 # sample used only to learn policy from unseen hands
REV_SAMPLES = 24           # shared random deals per flop/turn board
REV_MIN_LIVE = 25          # fewer usable candidates than this: no inference
REV_DECAY = 0.97           # each finished hand, older evidence keeps 97% of its weight
REV_MAX_SHARE = 0.65       # most the inferred range can count for, vs the ordinary range
REV_WARMUP_HANDS = 8       # hands of evidence before the inferred range is fully trusted
REV_STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
REV_BOARD_LEN = (0, 3, 4, 5)
# Approximate sizing choices, measured after calling. A bet that puts either
# player all-in effectively has its own category, even if it is small.
REV_SIZE_PRIORS = (0.36, 0.36, 0.21, 0.07)


def _sig(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------
# 1. Their decisions this hand, each with the situation they faced
# --------------------------------------------------------------------------
class RevEvent:
    """One decision by the opponent.
    action:    fold / check / call / raise, normalised the way the engine plays
               it ("check" while owing chips is a call; an all-in that doesn't
               raise is a call);
    street:    0-3, so the board at that moment is board[:REV_BOARD_LEN[street]];
    facing:    did they owe chips?
    pot_odds:  equity a call needed to break even, with the call capped at
               their stack and chips they could never win taken out;
    can_raise: could they raise?
    frac:      raw raise size as a share of the pot after calling (raises only);
    pot/cost:  contestable pot before calling and chips they would pay to call;
    stack:     chips behind before the decision, if available;
    min_extra, max_extra: matchable raise increments above the call, if known;
    extra:     their observed matchable raise increment."""
    __slots__ = ("action", "street", "facing", "pot_odds", "can_raise", "frac",
                 "pot", "cost", "stack", "sizes", "size_index", "call_risk",
                 "shove_context", "shoved")

    def __init__(self, action, street, facing, pot_odds, can_raise, frac=0.0,
                 *, pot=0, cost=0, stack=None, min_extra=1, max_extra=None, extra=0,
                 shove_context=None, shoved=False):
        self.action, self.street, self.facing = action, street, facing
        self.pot_odds, self.can_raise, self.frac = pot_odds, can_raise, frac
        self.pot, self.cost, self.stack = pot, cost, stack
        self.shove_context, self.shoved = shove_context, shoved
        self.call_risk = cost / stack if stack else 0.0
        self.sizes, self.size_index = (), None
        if can_raise and max_extra and pot + cost > 0:
            after_call = pot + cost
            half = after_call // 2
            min_extra = min(min_extra, max_extra)
            bounds = ((min_extra, min(half, max_extra - 1), 0.35),
                      (max(min_extra, half + 1), min(after_call, max_extra - 1), 0.75),
                      (max(min_extra, after_call + 1), max_extra - 1, 1.25),
                      (max_extra, max_extra, 0.0))
            sizes = []
            for (lo, hi, target), prior in zip(bounds, REV_SIZE_PRIORS):
                if lo > hi:
                    continue
                if lo <= extra <= hi:
                    self.size_index = len(sizes)
                increment = max(lo, min(hi, round(target * after_call)))
                pressure = increment / (after_call + increment)
                fold_chance = 0.08 + 0.30 * pressure
                called_discount = 0.18 + 0.16 * pressure
                risk = (cost + increment) / stack if stack else 0.0
                sizes.append((prior, increment, fold_chance, called_discount,
                              risk ** 2 - self.call_risk ** 2))
            self.sizes = tuple(sizes)

    def likelihood_key(self):
        return (self.action, self.facing, self.pot_odds, self.can_raise,
                self.pot, self.cost, self.call_risk, self.sizes, self.size_index)

    def __repr__(self):
        return (f"RevEvent({self.action}, st{self.street}, facing={self.facing}, "
                f"odds={self.pot_odds:.2f}, can_raise={self.can_raise})")


def rev_events(entries, opp, start_stacks=None):
    """The opponent's decisions in one hand's log entries (see rev_replay)."""
    return rev_replay(entries, opp, start_stacks)[0]


def rev_replay(entries, opp, start_stacks=None):
    """Replay one hand's log entries. Returns (opponent's RevEvents, chips each
    player has put in this hand, blinds and antes included).
    Log amounts: blinds/antes and calls are chips added; raise/all_in are the
    player's TOTAL bet on the street. start_stacks (player -> chips at the start
    of the hand) lets us cap calls at their stack; without it we assume deep."""
    out = []
    players = {e.get("bot_id") for e in entries if e.get("bot_id") is not None}
    unavailable = set()
    bets, put, current, street, pot = {}, {}, 0, "preflop", 0
    big_blind = min_raise = 1
    for e in entries:
        pid, act, st = e.get("bot_id"), e.get("action"), e.get("street")
        amt = e.get("amount") or 0
        if act == "show":
            continue
        if st != street:
            street, bets, current = st, {}, 0
            min_raise = big_blind
        if act in ("small_blind", "big_blind", "ante"):
            pot += amt
            put[pid] = put.get(pid, 0) + amt
            if act != "ante":
                bets[pid] = bets.get(pid, 0) + amt
                current = max(current, bets[pid])
            if act == "big_blind":
                big_blind = min_raise = max(1, amt)
            if start_stacks is not None and put[pid] >= start_stacks.get(pid, math.inf):
                unavailable.add(pid)
            continue
        if st not in REV_STREETS:
            continue
        mine = bets.get(pid, 0)
        owed = max(0, current - mine)
        left = None
        if start_stacks is not None and pid in start_stacks:
            left = max(0, start_stacks[pid] - put.get(pid, 0))
        total = amt if act in ("raise", "all_in") else 0
        if act == "fold":
            kind = "fold"
        elif act in ("raise", "all_in") and total > current:
            kind = "raise"
        elif act in ("check", "call", "raise", "all_in"):
            kind = "call" if owed > 0 else "check"
        else:
            continue
        if pid == opp:
            cost = owed if left is None else min(owed, left)
            contest = heads_up_call_pot(pot, owed, cost)
            odds = cost / (contest + cost) if cost > 0 else 0.0
            others = players - {pid}
            can_raise = (left is None or left > owed) and bool(others - unavailable)
            max_extra = None
            if can_raise and left is not None and len(others) == 1:
                other = next(iter(others))
                if other in start_stacks:
                    other_left = start_stacks[other] - put.get(other, 0)
                    other_extra = bets.get(other, 0) + other_left - current
                    max_extra = max(0, min(left - owed, other_extra))
                    can_raise = max_extra > 0
            frac = (total - current) / max(1, pot + owed) if kind == "raise" else 0.0
            extra = min(total - current, max_extra) if kind == "raise" and max_extra else 0
            shove_context = None
            shoved = False
            if st == "preflop" and max_extra:
                effective_bb = (current + max_extra) / big_blind
                shove_context = (current > big_blind, bisect.bisect_left((12, 30), effective_bb))
                shoved = kind == "raise" and (act == "all_in" or extra == max_extra)
            out.append(RevEvent(kind, REV_STREETS[st], owed > 0, odds, can_raise, frac,
                                pot=contest, cost=cost, stack=left, min_extra=min_raise,
                                max_extra=max_extra, extra=extra, shove_context=shove_context,
                                shoved=shoved))
        # Apply the chips.
        if kind == "call":
            add = amt if act == "call" and amt else (max(0, total - mine) if total else owed)
            if left is not None:
                add = min(add, left)
            bets[pid] = mine + add
        elif kind == "raise":
            add = max(0, total - mine)
            bets[pid] = total
            min_raise = max(min_raise, total - current)
            current = total
        else:
            add = 0
        pot += add
        put[pid] = put.get(pid, 0) + add
        if act in ("all_in", "fold") or (left is not None and add >= left):
            unavailable.add(pid)
    return out, put


# --------------------------------------------------------------------------
# 2. How likely each action is, for a hand with this equity, under each policy
# --------------------------------------------------------------------------
def rev_likelihoods(eq: float, e: RevEvent) -> list:
    """Joint probability of their action and observed size for each policy."""
    n_legal = (2 if e.facing else 1) + (1 if e.can_raise else 0)
    unknown = 1.0 / n_legal
    legal = (e.action == "raise" and e.can_raise) or (e.action in ("call", "fold") and e.facing) \
        or (e.action == "check" and not e.facing)
    if not legal:
        return [1e-6] * len(REV_POLICIES)
    rs = [_sig((eq - t) / REV_NOISE) if e.can_raise else 0.0 for t in REV_THRESHOLDS]
    cs = ([_sig((eq - e.pot_odds - m * (1 + e.call_risk)) / REV_NOISE)
           for m in REV_MARGINS] if e.facing else None)
    sized = e.action == "raise" and e.size_index is not None
    if sized:
        unknown /= len(e.sizes)

    if e.sizes:
        # All EVs are from their decision point, where fold is worth zero.
        # Estimate equity when called by discounting their random-hand equity.
        after_call = e.pot + e.cost
        passive_ev = max(0.0, eq * after_call - e.cost)
        variance = eq * (1.0 - eq)
        prior_sum = sum(size[0] for size in e.sizes)
        quotes = []
        for prior, extra, folds, discount, risk in e.sizes:
            called_pot = after_call + 2 * extra
            called_equity = eq - discount * variance
            ev = folds * e.pot + (1 - folds) * (called_equity * called_pot - e.cost - extra)
            # EV per chip of final pot keeps the odds in a similar range
            # across bet sizes. Risk aversion reduces the value of committing.
            quotes.append((prior / prior_sum, (ev - passive_ev) / called_pot, risk))
        raise_weights = []
        for margin in REV_MARGINS:
            weights = [prior * math.exp((edge - margin * risk) / (2 * REV_NOISE))
                       for prior, edge, risk in quotes]
            total = sum(weights)
            raise_weights.append((total, weights[e.size_index] if sized else total))
    else:
        raise_weights = [(1.0, 1.0)] * len(REV_MARGINS)
    out = []
    for r in rs:
        for j, (total, observed) in enumerate(raise_weights):
            passive = 1.0 - r
            norm = passive + r * total
            if e.action == "raise":
                p = r * observed / norm
            elif e.action == "check":
                p = passive / norm
            elif e.action == "call":
                p = passive * cs[j] / norm
            else:                                   # fold
                p = passive * (1.0 - cs[j]) / norm
            out.append((1.0 - REV_FLOOR) * p + REV_FLOOR * unknown)
    out.append(unknown)
    return out


def rev_loglik_known(eqs_by_street: dict, events) -> list:
    """Log-likelihood of their actions for each policy, when we know their cards
    (showdown). eqs_by_street: street -> their equity then."""
    ll = [0.0] * len(REV_POLICIES)
    for e in events:
        if e.street not in eqs_by_street:
            continue
        for i, p in enumerate(rev_likelihoods(eqs_by_street[e.street], e)):
            ll[i] += math.log(p)
    return ll


# --------------------------------------------------------------------------
# 3. Equity against a random hand, the number an equity bot acts on
# --------------------------------------------------------------------------
class RevBoards:
    """Per-board helpers, built once and shared by every candidate hand.
      river:      every possible hand ranked once; a hand's equity vs random is
                  the share of hands it beats (ties count half);
      flop, turn: REV_SAMPLES shared random (run-out, opponent hand) deals with
                  the opponent's score pre-computed (common random numbers:
                  every candidate faces the same deals, so comparisons between
                  candidates are fair even with few samples)."""

    def __init__(self, ev, pre_eq):
        self.ev, self.pre_eq = ev, pre_eq
        self.cache = {}
        self.preflop = {}

    def _build(self, board, deadline):
        ev = self.ev
        deck = [c for c in ev.variant.deck() if c not in board]
        if len(board) == 5:
            rank_pair = ev.ranker_for_board(board)
            ranks = []
            by_card = {card: [] for card in deck}
            pair_ranks = {}
            for a, b in combinations(deck, 2):
                if deadline is not None and time.perf_counter() >= deadline:
                    return None
                rank = rank_pair(a, b)
                pair_ranks[min(a, b), max(a, b)] = rank
                ranks.append(rank)
                by_card[a].append(rank)
                by_card[b].append(rank)
            ranks.sort()
            for blocked in by_card.values():
                blocked.sort()

            def river_score(combo):
                a, b = combo
                rank = pair_ranks[(a, b) if a < b else (b, a)]
                blocked_a, blocked_b = by_card[a], by_card[b]

                def wins_with_ties(scores):
                    return (bisect.bisect_left(scores, rank) + bisect.bisect_right(scores, rank)) / 2

                # Our combo appears in both blocked lists, so add its tie back once.
                wins = wins_with_ties(ranks) - wins_with_ties(blocked_a) - wins_with_ties(blocked_b) + 0.5
                legal = len(ranks) - len(blocked_a) - len(blocked_b) + 1
                return wins / legal

            return river_score
        rng = random.Random(hash(tuple(sorted(board))) & 0xFFFFFFFF)
        need = 5 - len(board)
        deals = []
        rankers = {}
        for _ in range(REV_SAMPLES):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            d = rng.sample(deck, need + 2)
            runout = tuple(sorted(d[:need]))
            rank_pair = rankers.get(runout)
            if rank_pair is None:
                rank_pair = rankers[runout] = ev.ranker_for_board(board + runout)
            deals.append((frozenset(d), rank_pair, rank_pair(d[-2], d[-1])))

        def sampled_score(combo):
            win = n = 0.0
            a, b = combo
            for used, rank_pair, opponent_rank in deals:
                if a in used or b in used:
                    continue
                mine = rank_pair(a, b)
                win += 1.0 if mine > opponent_rank else 0.5 if mine == opponent_rank else 0.0
                n += 1
            return win / n if n else 0.5

        return sampled_score

    def _preflop_score(self, combo):
        if combo not in self.preflop:
            self.preflop[combo] = self.pre_eq(combo)
        return self.preflop[combo]

    def scorer(self, board, deadline=None):
        """Prepare a single board scorer for a batch of compatible hands."""
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        if not board:
            return self._preflop_score
        key = tuple(board)
        if key not in self.cache:
            if len(self.cache) > 12:
                self.cache.clear()
            score = self._build(key, deadline)
            if score is None:
                return None
            self.cache[key] = score
        return self.cache[key]

    def equity(self, combo, board, deadline=None):
        score = self.scorer(board, deadline)
        return None if score is None else score(combo)


# --------------------------------------------------------------------------
# 4. Across hands: which policy are they using?
# --------------------------------------------------------------------------
class PolicyBelief:
    """Log-weights over REV_POLICIES for one opponent."""

    def __init__(self):
        k = len(REV_POLICIES) - 1
        self.prior = [math.log((1 - REV_UNKNOWN_PRIOR) / k)] * k + [math.log(REV_UNKNOWN_PRIOR)]
        self.log = list(self.prior)
        self.hands = 0                      # finished hands with at least one decision
        self.showdowns = 0

    def probs(self) -> list:
        m = max(self.log)
        w = [math.exp(x - m) for x in self.log]
        s = sum(w)
        return [x / s for x in w]

    def commit(self, loglik, showdown=False):
        """Add one finished hand's evidence; older evidence fades towards the prior."""
        self.log = [p + REV_DECAY * (x - p) + l for p, x, l in zip(self.prior, self.log, loglik)]
        m = max(self.log)
        self.log = [x - m for x in self.log]
        self.hands += 1
        self.showdowns += showdown

    def best(self):
        pr = self.probs()
        i = max(range(len(pr)), key=pr.__getitem__)
        return REV_POLICIES[i], pr[i], pr[-1]


# --------------------------------------------------------------------------
# 5. Within a hand: which cards do they hold?
# --------------------------------------------------------------------------
class RevHand:
    """Keep all legal combos for live reads and reuse work while a hand unfolds."""

    def __init__(self, ev, dead, seed, n=None):
        dead = set(dead)
        deck = [c for c in ev.variant.deck() if c not in dead]
        combos = list(combinations(deck, 2))
        rng = random.Random(seed)
        self.cands = combos if n is None or n >= len(combos) else rng.sample(combos, n)
        self.learn_cands = self.cands
        if len(self.cands) > REV_LEARN_CANDIDATES:
            self.learn_cands = rng.sample(self.cands, REV_LEARN_CANDIDATES)
        self.eq = {}
        self.complete_boards = set()
        self.likelihoods = {}
        self.history = ()
        self.board = ()
        self.rows = {}

    def fill(self, boards: RevBoards, board, deadline=None, candidates=None):
        """Work out equities on `board` for every live candidate, until the deadline."""
        key = tuple(board)
        if deadline is not None and time.perf_counter() >= deadline:
            return False
        if key in self.complete_boards:
            return True
        if candidates is None:
            candidates = self.cands
        score = boards.scorer(key, deadline)
        if score is None:
            return False
        table = self.eq.setdefault(key, {})
        bset = set(board)
        for c in candidates:
            if deadline is not None and time.perf_counter() >= deadline:
                return False
            if c not in table and c[0] not in bset and c[1] not in bset:
                table[c] = score(c)
        if candidates is self.cands:
            self.complete_boards.add(key)
        return True

    def loglik_rows(self, boards: RevBoards, events, board, deadline=None, candidates=None):
        """Score the complete sample, or return no result if the budget runs out."""
        if candidates is None:
            candidates = self.cands
        board = tuple(board)
        streets = sorted({e.street for e in events})
        prefixes = {s: board[:REV_BOARD_LEN[s]] for s in streets}
        for s in streets:
            if len(prefixes[s]) < REV_BOARD_LEN[s]:
                return []                     # board for that street not known
            if not self.fill(boards, prefixes[s], deadline, candidates):
                return []
        # Hands with equal equity on the same board share a likelihood table
        # when the observed action and the decision context are the same.
        history = tuple((prefixes[e.street], e.likelihood_key()) for e in events)
        reused = (board[:len(self.board)] == self.board
                  and history[:len(self.history)] == self.history)
        previous = self.rows if reused else {}
        full_actions = [(self.eq[key[0]], self.likelihoods.setdefault(key, {}), event)
                        for key, event in zip(history, events)]
        new_actions = full_actions[len(self.history):] if reused else full_actions
        bset = set(board)
        rows = {}
        for c in candidates:
            if deadline is not None and time.perf_counter() >= deadline:
                return []
            if c[0] in bset or c[1] in bset:
                continue
            row = previous.get(c)
            actions = full_actions if row is None else new_actions
            for equities, cache, event in actions:
                equity = equities[c]
                terms = cache.get(equity)
                if terms is None:
                    terms = cache[equity] = list(map(math.log, rev_likelihoods(equity, event)))
                row = terms if row is None else list(map(_reverse_add, row, terms))
            if row is not None:
                rows[c] = row
        if candidates is self.cands:
            self.history, self.board, self.rows = history, board, rows
        return list(rows.items())

    def infer(self, boards, events, board, belief: PolicyBelief, deadline=None):
        """Returns (WeightedRange, policy posterior for this hand) or None."""
        if not events:
            return None
        rows = self.loglik_rows(boards, events, board, deadline)
        if len(rows) < REV_MIN_LIVE:
            return None
        prior = [math.log(max(1e-12, p)) for p in belief.probs()]
        joint = [list(map(_reverse_add, prior, ll)) for _, ll in rows]
        peak = max(map(max, joint))
        joint = [[math.exp(x - peak) for x in r] for r in joint]
        hand_w = list(map(sum, joint))
        total = sum(hand_w)
        post = [sum(column) / total for column in zip(*joint)]
        result = WeightedRange([(c, w) for (c, _), w in zip(rows, hand_w)]), post
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        return result

    def marginal_loglik(self, boards, events, board, deadline=None):
        """Policy log-likelihood of a finished hand whose cards we never saw:
        log of the average over candidate hands (they are a uniform sample)."""
        rows = self.loglik_rows(boards, events, board, deadline, self.learn_cands)
        if len(rows) < REV_MIN_LIVE:
            return None
        out = []
        for i in range(len(REV_POLICIES)):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            col = [ll[i] for _, ll in rows]
            m = max(col)
            out.append(m + math.log(sum(math.exp(x - m) for x in col) / len(col)))
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        return out


class RevMixture:
    """Opponent range for estimate_equity(): the inferred range with chance
    `share`, otherwise the ordinary range (TopFraction, or random if None)."""

    def __init__(self, inferred, baseline, share):
        self.inferred, self.baseline, self.share = inferred, baseline, share

    def prepare(self, rng, deck, hole_n, ev, board):
        if self.baseline is not None and hasattr(self.baseline, "prepare"):
            self.baseline.prepare(rng, deck, hole_n, ev, board)

    def sample(self, rng, available, hole_n, ev, board):
        if rng.random() < self.share:
            return self.inferred.sample(rng, available, hole_n, ev, board)
        if self.baseline is None:
            return rng.sample(available, hole_n)
        return self.baseline.sample(rng, available, hole_n, ev, board)

    def __repr__(self):
        return f"RevMixture(share {self.share:.2f}, inferred {len(self.inferred)}, base {self.baseline!r})"


def rev_share(belief: PolicyBelief, post_unknown: float) -> float:
    """How much to trust the inferred range. A starting heuristic, not a
    calibrated confidence: less when 'unknown' fits this hand well, and less
    until we have seen a few hands."""
    warm = min(1.0, belief.hands / REV_WARMUP_HANDS)
    return REV_MAX_SHARE * (1.0 - post_unknown) * warm


##############################################################################
# hackathon.py
##############################################################################
"""
hackathon.py: our EquityBot wired into the LARUS x IEEE uOttawa tournament
(github.com/FreddyyAndrews/poker-hackathon-kit).

The tournament calls one function, decide(game_state), with a dict, and wants
a dict back:  {"action": "fold" | "check" | "call" | "all_in"}  or
              {"action": "raise", "amount": <our TOTAL bet this street>}

Format: heads-up No-Limit Hold'em freezeout, 10,000 chips each, blinds that
rise every 20 hands (antes from hand 101), 2 s per decision on half a CPU
core. The bot process lives for exactly one match, so module-level variables
are our memory for that match.

What this file adds on top of the kit (bot.py / equity.py / decision.py /
opponents.py):
  1. TRANSLATION   their dict  <->  our Observation / Action.
  2. MEMORY        the opponent model is rebuilt from `match_action_log`,
                   including hands where we never had to act and showdowns.
  3. PUSH / FOLD   when stacks get short (<= 12 big blinds) the small blind
                   goes all-in or folds, from a heuristic chart (see NASH_PUSH),
                   plus extra shoves when the EV maths says they fold enough.
  4. EXPLOITS      archetype counters, bet-size learning, position, traps for
                   bluffers, turn stabs vs players who give up (TournamentBot).
  5. SAFETY        never "check" when we owe chips (the engine would CALL),
                   never fold when checking is free, any error -> check/fold,
                   and the thinking time is a fraction of the deadline.
"""


import os
import random
import time


STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

# Thinking time: a share of the deadline, capped. Half a CPU core still runs
# thousands of Monte Carlo trials in this time. BOT_THINK_MS / BOT_MAX_ITERS
# only exist so test matches can run faster; the tournament doesn't set them.
THINK_SHARE = 0.30
THINK_CAP_MS = float(os.environ.get("BOT_THINK_MS", "600"))
MAX_ITERS = int(os.environ.get("BOT_MAX_ITERS", "6000"))

PUSH_FOLD_BB = 12          # effective stack (in big blinds) where push/fold starts
SEVEN_TWO_STEAL = True     # raise 7-2 from the button: cheap shot at the 7-2 prize

# Reverse Monte Carlo is on for heads-up play. BOT_REVERSE=0 disables it
# for controlled arena comparisons.
REVERSE = os.environ.get("BOT_REVERSE", "0") == "1"
REVERSE_BUDGET_MS = 150    # most thinking time the inference may take per decision
REVERSE_SHARE_MIN = 0.05   # below this the inferred range isn't worth mixing in
# Arena experiments, enabled independently for A/B tests.
RESPONSE_EV = os.environ.get("BOT_RESPONSE_EV", "0") == "1"
SHOVE_READ = os.environ.get("BOT_SHOVE_READ", "1") == "1"
# Price a postflop all-in with the bluff share seen on their all-ins only.
JAM_BLUFF = os.environ.get("BOT_JAM_BLUFF", "0") == "1"
JAM_MIN_SHOWDOWNS = 3

# All 169 starting hands, best first, with their heads-up equity against a
# random hand (computed with equity.estimate_equity, 25,000 trials each).
HAND_TABLE = (
    "AA:0.850 KK:0.822 QQ:0.802 JJ:0.774 TT:0.747 99:0.718 88:0.694 AKs:0.671 77:0.666 "
    "AQs:0.666 AJs:0.657 AKo:0.654 ATs:0.646 AQo:0.641 66:0.636 KQs:0.636 AJo:0.630 "
    "KJs:0.630 A9s:0.627 ATo:0.626 A8s:0.623 KQo:0.617 KTs:0.616 A7s:0.610 A9o:0.607 "
    "A6s:0.605 KJo:0.605 55:0.604 A8o:0.603 QTs:0.599 K9s:0.598 A5s:0.597 KTo:0.597 "
    "QJs:0.597 A4s:0.592 K8s:0.585 A3s:0.584 A6o:0.584 A7o:0.583 K9o:0.580 A5o:0.579 "
    "QJo:0.579 Q9s:0.578 A2s:0.576 JTs:0.575 QTo:0.574 44:0.571 A4o:0.571 K7s:0.571 "
    "K8o:0.567 K6s:0.563 A3o:0.561 Q8s:0.560 J9s:0.559 Q9o:0.559 K5s:0.557 JTo:0.554 "
    "K7o:0.552 A2o:0.550 K6o:0.548 K4s:0.545 Q7s:0.544 J8s:0.543 K3s:0.543 33:0.538 "
    "Q6s:0.538 T9s:0.534 K2s:0.532 K5o:0.531 Q8o:0.531 J9o:0.530 Q5s:0.526 K4o:0.525 "
    "J7s:0.523 Q4s:0.522 T8s:0.521 T9o:0.518 Q7o:0.516 Q3s:0.514 K3o:0.513 Q6o:0.513 "
    "J6s:0.512 J8o:0.511 98s:0.510 Q2s:0.504 T7s:0.504 22:0.503 Q5o:0.502 K2o:0.500 "
    "J5s:0.498 T6s:0.494 Q4o:0.493 97s:0.492 J7o:0.492 J4s:0.491 T8o:0.491 Q3o:0.481 "
    "87s:0.477 T7o:0.476 J5o:0.475 J6o:0.475 96s:0.474 J3s:0.474 Q2o:0.474 98o:0.473 "
    "J2s:0.471 T5s:0.465 86s:0.463 T4s:0.463 T6o:0.461 76s:0.460 95s:0.460 97o:0.460 "
    "J4o:0.460 T3s:0.456 J3o:0.455 85s:0.450 87o:0.450 96o:0.447 T2s:0.447 T5o:0.446 "
    "94s:0.440 75s:0.439 J2o:0.439 86o:0.438 93s:0.436 T4o:0.435 65s:0.434 84s:0.429 "
    "76o:0.425 95o:0.425 T3o:0.424 54s:0.419 92s:0.419 64s:0.417 74s:0.417 T2o:0.417 "
    "85o:0.415 94o:0.409 83s:0.406 75o:0.404 82s:0.403 63s:0.400 73s:0.399 93o:0.398 "
    "53s:0.396 84o:0.396 65o:0.393 43s:0.392 92o:0.391 74o:0.389 54o:0.384 64o:0.382 "
    "52s:0.380 72s:0.378 83o:0.374 62s:0.372 82o:0.367 73o:0.366 42s:0.364 53o:0.364 "
    "63o:0.359 32s:0.357 72o:0.348 43o:0.345 52o:0.343 62o:0.343 42o:0.327 32o:0.325"
).split()
HAND_EQUITY = {h.split(":")[0]: float(h.split(":")[1]) for h in HAND_TABLE}
HAND_ORDER = [h.split(":")[0] for h in HAND_TABLE]


def _build_percentiles():
    """Hand class -> share of all 1,326 starting hands that are at least as
    good. 'AA' -> 0.5%, 'AKs' -> ~3%, '32o' -> 100%."""
    out, seen = {}, 0
    for name in HAND_ORDER:
        seen += 6 if len(name) == 2 else (4 if name[2] == "s" else 12)
        out[name] = seen / 1326
    return out


PERCENTILE = _build_percentiles()


def hand_class(cards: list) -> str:
    """['7h', 'As'] -> 'A7o'."""
    order = "23456789TJQKA"
    a, b = sorted(cards, key=lambda c: order.index(c[0]), reverse=True)
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def rev_pre_equity(combo) -> float:
    """Preflop equity vs a random hand for a combo of card ints (table lookup)."""
    return HAND_EQUITY[hand_class([card_str(c) for c in combo])]


# Short-stack shove chart. A HEURISTIC, not a solved Nash table: the share of
# hands the small blind shoves at each effective stack depth (in big blinds),
# set to roughly match published heads-up push/fold frequencies, with hands
# picked by their equity against a random hand (HAND_TABLE). Interpolated.
NASH_PUSH = [(1, 1.00), (2, 0.95), (3, 0.85), (4, 0.75), (5, 0.68), (6, 0.63),
             (8, 0.58), (10, 0.53), (12, 0.48), (15, 0.42), (20, 0.35), (30, 0.25)]


def nash_push_share(eff_bb: float) -> float:
    pts = NASH_PUSH
    if eff_bb <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if eff_bb <= x1:
            return y0 + (y1 - y0) * (eff_bb - x0) / (x1 - x0)
    return pts[-1][1]


# --------------------------------------------------------------------------
# The tournament bot: extra-edge rule, position, traps, turn stabs
# --------------------------------------------------------------------------
def edge_margin(m_ratio: float) -> float:
    """Extra equity we demand before calling off a big part of our stack.
    M = our chips / (blinds + antes per hand) = how many hands we could sit
    out. With a deep M, blinds are cheap: wait for a big edge instead of
    flipping coins (we expect to outplay most bots later anyway). As blinds
    bite, take smaller edges. M >= 50: +15 points; M <= 12: none."""
    return 0.15 * max(0.0, min(1.0, (m_ratio - 12) / 38))


# Equity realization: after the flop the button acts last and usually wins a
# bit more than its raw equity; the big blind acts first and wins less. A rule
# of thumb (1.05 / 0.85), applied on the flop and turn only, since the river
# has no later streets. Tested: ahead on two separate sets of arena seeds.
REALIZE_IN_POSITION = 1.05
REALIZE_OUT_OF_POSITION = 0.85


class TournamentBot(EquityBot):
    """EquityBot plus heads-up tournament tweaks (all in tweak()):
      1. calling off a big share of our stack needs an extra edge while blinds
         are cheap (big_call_margin, see edge_margin). This applies to
         marginal calls only: strong hands always call (decision.py, rule a);
      2. equity realization by position on the flop and turn, but NOT for a
         final decision (calling puts us all-in, or they are all-in): with no
         betting left, position can't change how much equity we realise;
      3. against a bluffer or maniac, when we act first after the flop, check
         good-but-not-huge hands and let them bet into us (trap);
      4. against a player who usually bets the flop and then gives up on the
         turn: once we have SEEN them check the turn after betting the flop,
         bet (on the turn after their check, or on the river). We never lead
         into them on the turn on this basis, before they have acted."""

    big_call_margin = 0.0
    reverse = None          # (WeightedRange, share) from reverse.py for THIS decision, or None
    shove_read = None       # (estimated jam fraction, trust), for a preflop shove
    jam_read = None         # bluff share of their postflop all-ins, when facing one

    def opponent_ranges(self, obs):
        """The ordinary range, mixed with the reverse-Monte-Carlo range when
        there is one for this decision (heads-up only)."""
        ranges = super().opponent_ranges(obs)
        if self.shove_read is not None and len(obs.active_opponents) == 1:
            fraction, trust = self.shove_read
            base = ranges[0] if ranges else None
            old_fraction = base.preflop_fraction if base is not None else 1.0
            bluff = base.bluff_share if base is not None else 0.12
            ranges = [TopFraction(1.0, old_fraction + trust * (fraction - old_fraction), bluff)]
        if self.jam_read is not None and ranges and ranges[0] is not None:
            b = ranges[0]
            ranges = [TopFraction(b.fraction, b.preflop_fraction, self.jam_read,
                                  cap=b.cap, cap_board=b.cap_board)] + list(ranges[1:])
        if self.reverse is None or len(obs.active_opponents) != 1:
            return ranges
        inferred, share = self.reverse
        base = ranges[0] if ranges else None
        return [RevMixture(inferred, base, share)]

    def tweak(self, adj, obs, spot):
        in_position = obs.seat == obs.button          # heads-up: button acts last postflop
        opp = obs.active_opponents[0] if len(obs.active_opponents) == 1 else None
        st = self.model.get(opp) if (opp and self.use_model) else None

        if self.big_call_margin and spot.to_call >= 0.4 * spot.stack:
            adj.call_margin += self.big_call_margin
            adj.notes.append(f"big call: +{self.big_call_margin:.2f} equity needed")

        final = spot.to_call > 0 and (spot.to_call >= spot.stack or spot.min_raise_to == 0)
        if spot.street in (1, 2) and not final:
            factor = REALIZE_IN_POSITION if in_position else REALIZE_OUT_OF_POSITION
            spot.equity = max(0.0, min(1.0, spot.equity * factor))
            adj.notes.append(f"position x{factor}")

        if st is None:
            return adj
        kind = st.archetype()
        if not in_position and spot.street > 0 and kind in ("bluffer", "maniac"):
            adj.trap = True
            adj.notes.append("trap: check to the bluffer")

        if (spot.street in (2, 3) and spot.to_call == 0 and st.turn_chances >= 4
                and st.give_up_rate > 0.6 and st.fold_to_bet >= 0.3):
            theirs = [e for e in obs.history if e.player_id == opp]
            bet_flop = any(e.street == 1 and e.kind in ("bet", "raise") for e in theirs)
            checked_turn = any(e.street == 2 and e.kind == "check" for e in theirs)
            bet_turn = any(e.street == 2 and e.kind in ("bet", "raise") for e in theirs)
            bet_river = any(e.street == 3 and e.kind in ("bet", "raise") for e in theirs)
            gave_up = bet_flop and checked_turn and not bet_turn      # seen, not assumed
            if gave_up and (spot.street == 2 or not bet_river):
                adj.bluff_freq = max(adj.bluff_freq or 0.0, 0.8)
                adj.notes.append(f"they checked the turn after betting the flop "
                                 f"(give up {st.give_up_rate:.0%}): bet")
        return adj


# --------------------------------------------------------------------------
# Match memory: turns match_action_log into opponent-model updates
# --------------------------------------------------------------------------
class MatchMemory:
    def __init__(self):
        self.bot = TournamentBot(HOLDEM, time_budget_ms=THINK_CAP_MS, max_iterations=MAX_ITERS,
                                 seed=None)
        self.bot.rng = random      # the tournament seeds `random`, so matches replay exactly
        self.me = None
        self.done_hands = set()    # hand numbers already fed to the model
        self.boards = {}           # hand_num -> last board we saw (to judge showdown bluffs)
        # Reverse Monte Carlo state (used only when REVERSE is on).
        self.beliefs = {}          # opponent -> PolicyBelief (how they play)
        self.rev_hands = {}        # hand_num -> RevHand (candidate hands + cached equities)
        self.holes = {}            # hand_num -> our hole cards (they can't hold them)
        self.rev_stacks = {}       # hand_num -> starting stacks recovered from the action log
        self.rev_boards = RevBoards(self.bot.ev, rev_pre_equity)
        self.bot.response_scorer = self.rev_boards.scorer if RESPONSE_EV else None
        self.shove_reads = ShoveReads()
        self.reverse_status = None

    def remember_stacks(self, s):
        """Starting stacks for this hand, recovered once from the action log.
        Used to size our bets by what the opponent could match."""
        h = s.get("hand_num", 0)
        if h not in self.rev_stacks:
            put = rev_replay(s.get("action_log") or [], None)[1]
            self.rev_stacks[h] = {p["bot_id"]: p["stack"] + put.get(p["bot_id"], 0)
                                  for p in s["players"]}
        return self.rev_stacks[h]

    def remember_reverse_state(self, s, hole):
        h = s.get("hand_num", 0)
        self.holes.setdefault(h, hole)
        return self.remember_stacks(s)

    def reverse_range(self, s, hole, board, opp, budget_ms):
        """Infer the opponent's hand for this decision. Returns
        (WeightedRange, share, posterior) or None when there's nothing useful."""
        deadline = time.perf_counter() + budget_ms / 1000.0
        entries = s.get("action_log") or []
        start = self.remember_reverse_state(s, hole)
        h = s.get("hand_num", 0)
        events = rev_events(entries, opp, start)
        if not events:
            self.reverse_status = "no_actions"
            return None
        belief = self.beliefs.setdefault(opp, PolicyBelief())
        rh = self.rev_hands.get(h)
        if rh is None:
            rh = self.rev_hands[h] = RevHand(self.bot.ev, hole, seed=h * 1_000_003 + sum(hole))
        res = rh.infer(self.rev_boards, events, board, belief, deadline)
        if res is None:
            self.reverse_status = "budget" if time.perf_counter() >= deadline else "unavailable"
            return None
        inferred, post = res
        share = rev_share(belief, post[-1])
        if share < REVERSE_SHARE_MIN:
            self.reverse_status = "low_confidence"
            return None
        self.reverse_status = "used"
        return inferred, share, post

    def rev_commit(self, h, entries, shown, opps, deadline):
        """Learn how each opponent plays from one finished hand, once. With
        their cards shown we score their actions exactly; otherwise we average
        over the candidate hands."""
        board = [parse_card(c) for c in self.boards.get(h, [])]
        start_stacks = self.rev_stacks.pop(h, None)
        for opp in opps:
            events = rev_events(entries, opp, start_stacks)
            if SHOVE_READ and start_stacks is not None:
                self.shove_reads.commit(opp, events)
            if not REVERSE:
                continue
            events = [e for e in events if len(board) >= REV_BOARD_LEN[e.street]]
            if not events:
                continue
            belief = self.beliefs.setdefault(opp, PolicyBelief())
            cards = shown.get(opp) or []
            if len(cards) == 2:
                combo = tuple(parse_card(c) for c in cards)
                eqs = {st: self.rev_boards.equity(combo, board[:REV_BOARD_LEN[st]], deadline)
                       for st in {e.street for e in events}}
                if None not in eqs.values() and time.perf_counter() < deadline:
                    belief.commit(rev_loglik_known(eqs, events), showdown=True)
                continue
            rh = self.rev_hands.get(h)
            if rh is None:
                hole = self.holes.get(h, ())
                rh = RevHand(self.bot.ev, hole, seed=h * 1_000_003 + sum(hole))
            ll = rh.marginal_loglik(self.rev_boards, events, board, deadline)
            if ll is not None:
                belief.commit(ll)
        self.rev_hands.pop(h, None)

    @staticmethod
    def replay(entries, me, start_stacks=None):
        """One hand's log entries, replayed in order. Returns:
          events     ActionEvents with bet/raise/call worked out and
                     `facing_bet` (did they face a real bet?) filled in;
          shown      cards shown at showdown, by player;
          responses  (player, size, folded?, raised?) for every bet/raise WE made that
                     someone answered; size = fraction of the pot after calling,
                     counting only the chips the opponent could MATCH (a shove
                     at a short stack is a small bet to them; the engine hands
                     the rest back). Needs `start_stacks` (player -> chips at
                     the start of the hand); without it, an all-in call for
                     less still shows what they could match;
          preflop    player -> flags: acted voluntarily preflop; first action
                     unopened (open_chance) and was it a shove (opened); faced
                     a raise (reshove_chance) and shoved over it (reshoved)."""
        events, shown, responses, preflop = [], {}, [], {}
        bets, current, street, bb_amount, pot = {}, 0, "preflop", 0, 0
        put, folded = {}, set()                    # chips in this hand; who folded
        # Our last bet awaiting an answer: (matched size, our total, bet before it, pot base)
        pending = None
        for e in entries:
            pid, act, amt, st = e.get("bot_id"), e.get("action"), e.get("amount") or 0, e.get("street")
            if act == "show":
                shown[pid] = e.get("cards") or []
                continue
            if st != street:                       # a new street: nobody has bet yet
                street, bets, current = st, {}, 0
            if act in ("small_blind", "big_blind", "ante"):
                pot += amt
                put[pid] = put.get(pid, 0) + amt
                if act != "ante":
                    bets[pid] = bets.get(pid, 0) + amt
                    current = max(current, bets[pid])
                if act == "big_blind":
                    bb_amount = amt
                continue
            if st not in STREETS:
                continue
            mine = bets.get(pid, 0)
            facing = current > mine and (st != "preflop" or current > bb_amount)
            if pending is not None and pid != me and act in ("fold", "call", "raise", "all_in"):
                size, our_total, before, base = pending
                raised = act in ("raise", "all_in") and amt > current
                if act == "all_in" and before < amt < our_total:
                    size = (amt - before) / base   # all-in call for less: all they could match
                responses.append((pid, size, act == "fold", raised))
                pending = None
            if act == "call":
                kind = "call"
                bets[pid] = mine + amt
                pot += amt
                put[pid] = put.get(pid, 0) + amt
            elif act in ("raise", "all_in"):
                total = amt                        # their total on this street
                pot_before, to_call = pot, max(0, current - mine)
                pot += max(0, total - mine)
                put[pid] = put.get(pid, 0) + max(0, total - mine)
                if total > current:
                    kind = "bet" if current == 0 else "raise"
                    if pid == me:
                        base = max(1, pot_before + to_call)
                        matched = total
                        if start_stacks:
                            reach = [bets.get(o, 0) + start_stacks[o] - put.get(o, 0)
                                     for o in start_stacks
                                     if o != me and o not in folded and start_stacks[o] > 0]
                            if reach and max(reach) > current:
                                matched = min(total, max(reach))
                        pending = ((matched - current) / base, total, current, base)
                    current = total
                else:
                    kind = "call"                  # all-in for less than the bet
                bets[pid] = total
            elif act in ("fold", "check"):
                kind = act
                if act == "fold":
                    folded.add(pid)
            else:
                continue
            if st == "preflop" and pid != me:
                # Only an all-in that RAISES is a shove (an all-in call is a call).
                # Open shove: first raise of the hand; reshove: all-in over a raise.
                f = preflop.setdefault(pid, {"acted": False, "open_chance": False, "opened": False,
                                             "reshove_chance": False, "reshoved": False})
                if not f["acted"]:
                    f["acted"] = True
                    if not facing:
                        f["open_chance"] = True
                        f["opened"] = act == "all_in" and kind in ("bet", "raise")
                if facing:
                    f["reshove_chance"] = True
                    if act == "all_in" and kind in ("bet", "raise"):
                        f["reshoved"] = True
            events.append(ActionEvent(pid, e.get("seat", 0), STREETS[st], kind, amt,
                                      bets.get(pid, 0), facing))
        return events, shown, responses, preflop

    @staticmethod
    def postflop_jams(entries):
        """player -> the last postflop street (1-3) where they moved all-in
        with a bet or raise (an all-in call doesn't count)."""
        jams, street, bets, current = {}, None, {}, 0
        for e in entries:
            st, act = e.get("street"), e.get("action")
            if st != street:
                street, bets, current = st, {}, 0
            if st not in STREETS or STREETS[st] == 0:
                continue
            pid, amt = e.get("bot_id"), e.get("amount") or 0
            if act == "call":
                bets[pid] = bets.get(pid, 0) + amt
            elif act in ("raise", "all_in"):
                if act == "all_in" and amt > current:
                    jams[pid] = STREETS[st]
                bets[pid] = amt
                current = max(current, amt)
        return jams

    def sync(self, state):
        """Feed every finished hand in match_action_log to the opponent model, once."""
        self.me = state["your_bot_id"]
        self.bot.me = self.me
        now = state.get("hand_num", 0)
        by_hand = {}
        for e in state.get("match_action_log") or []:
            by_hand.setdefault(e.get("hand_num"), []).append(e)
        model = self.bot.model
        rev_deadline = time.perf_counter() + 0.1     # learning from old hands: at most 0.1 s
        for h in sorted(k for k in by_hand if isinstance(k, int) and k < now):
            if h in self.done_hands:
                continue
            entries = by_hand[h]
            self.done_hands.add(h)
            if not any(e.get("action") == "big_blind" for e in entries):
                continue                           # start of the hand fell off the log
            events, shown, responses, preflop = self.replay(entries, self.me,
                                                            self.rev_stacks.get(h))
            opps = sorted({e.get("bot_id") for e in entries} - {self.me, None})
            model.start_hand(opps)
            for ev in events:
                if ev.player_id != self.me:
                    model.observe(ev.player_id, ev.street, ev.kind, ev.facing_bet)
            for pid, size, folded, raised in responses:
                model.observe_response(pid, size, folded, raised)
            for pid, f in preflop.items():
                model.observe_preflop(pid, f["open_chance"], f["opened"],
                                      f["reshove_chance"], f["reshoved"])
            for pid in opps:
                # Turn barrels: they bet the flop, then acted first unopposed on the turn.
                mine = [ev for ev in events if ev.player_id == pid]
                if any(ev.street == 1 and ev.kind in ("bet", "raise") for ev in mine):
                    turn = [ev for ev in mine if ev.street == 2]
                    if turn and not turn[0].facing_bet:
                        model.observe_turn_barrel(pid, turn[0].kind in ("bet", "raise"))
            # Showdown: was their LAST postflop bet made with a weak hand, judged
            # on the board as it was when they bet (flop = 3 cards, turn = 4, ...)?
            board = self.boards.get(h, [])
            jams = self.postflop_jams(entries)
            for pid, cards in shown.items():
                if pid == self.me or len(cards) != 2:
                    continue
                if pid in jams and len(board) >= 2 + jams[pid]:
                    model.observe_jam_showdown(pid, showed_weak(
                        self.bot.ev, [parse_card(c) for c in cards],
                        [parse_card(c) for c in board[:2 + jams[pid]]]))
                bet_streets = [ev.street for ev in events
                               if ev.player_id == pid and ev.street > 0 and ev.kind in ("bet", "raise")]
                if bet_streets and len(board) >= 2 + max(bet_streets):
                    board_then = board[:2 + max(bet_streets)]
                    model.observe_showdown(pid, True, showed_weak(
                        self.bot.ev, [parse_card(c) for c in cards], [parse_card(c) for c in board_then]))
            model.end_hand()
            if (REVERSE or SHOVE_READ) and len(opps) == 1:
                self.rev_commit(h, entries, shown, opps, rev_deadline)
        # keep memory small
        for h in [k for k in self.boards if k < now - 50]:
            del self.boards[h]
        for d in (self.rev_hands, self.holes, self.rev_stacks):
            for h in [k for k in d if k < now - 5]:
                del d[h]


MEMORY = MatchMemory()


# --------------------------------------------------------------------------
# decide()
# --------------------------------------------------------------------------
def _safe(state, reason="fallback"):
    return {"action": "check" if state.get("can_check") else "fold", "debug": {"why": reason}}


def _warmup():
    """Fill the evaluator cache before the clock matters."""
    for hole, board in (("As Kd", ""), ("7c 7d", "2h 9s Kd"), ("Qh Jh", "Th 9c 2d 4s")):
        estimate_equity(hole, board, MEMORY.bot.ev, 1, time_budget_ms=300, max_iterations=4000)


def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        try:
            _warmup()
        except Exception as e:                       # harmless, but say so in the log
            print("warmup error:", repr(e))
        return {"action": "check"}
    try:
        return _decide(game_state)
    except Exception as e:                           # never crash: a crash is a fold
        print("decide error:", repr(e))
        return _safe(game_state, "error: " + repr(e)[:120])


def _decide(s: dict) -> dict:
    t0 = time.perf_counter()
    mem = MEMORY
    mem.sync(s)
    me = s["your_bot_id"]
    hand = s.get("hand_num", 0)
    board_str = list(s.get("community_cards") or [])
    mem.boards[hand] = board_str

    hole = [parse_card(c) for c in s["your_cards"]]
    board = [parse_card(c) for c in board_str]
    street = STREETS.get(s["street"], len(board) and len(board) - 2)
    bb = s["blinds"]["bb"]
    sb = s["blinds"]["sb"]
    owed = s["amount_owed"]
    my_bet = s["your_bet_this_street"]
    stack = s["your_stack"]
    current = s["current_bet"]

    opps = [p for p in s["players"] if p["bot_id"] != me and not p["is_folded"]
            and p.get("state") != "busted"]
    opp_ids = [p["bot_id"] for p in opps]
    heads_up = len(s["players"]) == 2 and len(opp_ids) == 1
    use_reverse = REVERSE and heads_up
    use_shove_read = SHOVE_READ and heads_up
    if use_reverse or use_shove_read:
        mem.remember_reverse_state(s, hole)
    elif heads_up:
        mem.remember_stacks(s)
    opp_reach = max((p["stack"] + p["bet_this_street"] for p in opps), default=0)
    opp_can_act = any(p["stack"] > 0 for p in opps)
    all_in_to = my_bet + stack
    eff_bb = min(all_in_to, opp_reach) / bb if bb else 99   # effective stack, big blinds

    can_raise = stack > owed and opp_can_act and all_in_to > current
    min_to = min(s["min_raise_to"], all_in_to) if can_raise else 0
    max_to = min(all_in_to, max(opp_reach, s["min_raise_to"])) if can_raise else 0

    # --- 1. Short stack: push or fold from the small blind -----------------
    unopened_sb = street == 0 and current <= bb and my_bet < bb and owed > 0 and can_raise
    cls = hand_class(s["your_cards"])
    opp_stats = mem.bot.model.get(opp_ids[0]) if opp_ids else None
    if unopened_sb and eff_bb <= PUSH_FOLD_BB:
        share = nash_push_share(eff_bb)
        info = {"eff_bb": round(eff_bb, 1), "hand": cls, "nash": round(share, 2)}
        if PERCENTILE[cls] <= share:
            return {"action": "all_in", "debug": {"why": "push (chart)", **info}}
        # Outside the chart: once we have seen how this opponent answers big
        # bets, shove anyway if the maths says so (exploit a tight caller).
        if opp_stats is not None and opp_stats.size_faced[3] >= 3:
            f = opp_stats.fold_at(3.0)                 # their fold rate to shoves
            calling = TopFraction(1.0, preflop_fraction=max(0.05, 1.0 - f))
            eq = estimate_equity(hole, [], mem.bot.ev, 1, opp_ranges=[calling],
                                 time_budget_ms=150, max_iterations=3000, rng=random).equity
            # Unmatched chips come back if the shorter opponent calls.
            matched_to = min(all_in_to, opp_reach)
            add = matched_to - my_bet
            ev = ev_bet(f, s["pot"], add, matched_to - current, eq)
            info.update(fold=round(f, 2), eq=round(eq, 2), ev=round(ev))
            if ev > 0:
                return {"action": "all_in", "debug": {"why": "push (exploit: they fold)", **info}}
        return {"action": "fold", "debug": {"why": "push/fold: fold", **info}}

    # --- 1b. Opponent shoves almost every hand: shove-or-fold back ----------
    # Against an all-in bot the only question is "is my hand good enough to
    # gamble?". Fold the rest for half a blind. Good enough = equity vs a
    # random hand beats 50% plus the deep-stack edge margin.
    m_ratio = (stack + my_bet) / max(1, sb + bb + s["blinds"].get("ante", 0))
    archetype = opp_stats.archetype() if opp_stats is not None else "unknown"
    if unopened_sb and archetype == "shover":
        need = 0.50 + edge_margin(m_ratio)
        if HAND_EQUITY[cls] >= need:
            return {"action": "all_in", "debug": {"why": "vs shover: shove", "hand": cls,
                                                  "need": round(need, 2)}}
        return {"action": "fold", "debug": {"why": "vs shover: fold", "hand": cls,
                                            "need": round(need, 2)}}

    # --- 2. 7-2 from the button: a small steal for the 7-2 prize ----------
    if SEVEN_TWO_STEAL and unopened_sb and cls in ("72o", "72s") and eff_bb > PUSH_FOLD_BB \
            and (opp_stats is None or opp_stats.fold_to_bet >= 0.3):
        to = max(min_to, int(2.5 * bb))
        return {"action": "raise", "amount": to, "debug": {"why": "7-2 steal"}}

    # --- 3. Everything else: the equity bot ---------------------------------
    events = mem.replay(s.get("action_log") or [], me)[0]
    bot = mem.bot
    limit = float(s.get("time_limit_ms") or 2000)
    used = (time.perf_counter() - t0) * 1000
    bot.time_budget_ms = max(50.0, min(THINK_CAP_MS, THINK_SHARE * limit) - used)
    bot.big_call_margin = edge_margin(m_ratio)
    bot.shove_read = None
    bot.jam_read = None
    if (JAM_BLUFF and heads_up and street > 0 and owed > 0 and not opp_can_act
            and opp_stats is not None and opp_stats.jam_showdowns >= JAM_MIN_SHOWDOWNS):
        bot.jam_read = opp_stats.jam_bluff_share      # they are all-in: how often is it air?
    if use_shove_read and street == 0 and owed > 0 and not can_raise:
        reads = rev_events(s.get("action_log") or [], opp_ids[0], mem.rev_stacks[hand])
        if reads:
            bot.shove_read = mem.shove_reads.read(opp_ids[0], reads[-1])

    # --- 3a. Reverse Monte Carlo: which holdings explain their betting?
    bot.reverse = None
    rev_debug = None
    if use_reverse:
        t1 = time.perf_counter()
        res = mem.reverse_range(s, hole, board, opp_ids[0],
                                min(REVERSE_BUDGET_MS, 0.35 * bot.time_budget_ms))
        elapsed_ms = (time.perf_counter() - t1) * 1000
        rev_debug = {"status": mem.reverse_status, "ms": round(elapsed_ms, 1)}
        if res is not None:
            inferred, share, post = res
            bot.reverse = (inferred, share)
            rev_debug.update(share=round(share, 2), unknown=round(post[-1], 2),
                             hands=len(inferred))
        bot.time_budget_ms = max(50.0, bot.time_budget_ms - elapsed_ms)

    obs = Observation(
        player_id=me, seat=s.get("seat_to_act", 0), hole=tuple(hole), board=tuple(board),
        street=street, pot=s["pot"], to_call=owed, my_bet=my_bet, current_bet=current,
        min_raise_to=min_to, max_raise_to=max_to, stack=stack, big_blind=bb,
        active_opponents=opp_ids, history=events,
        seat_ids=[p["bot_id"] for p in s["players"]], button=s.get("button_seat", 0),
        opponent_max_to=opp_reach if len(opp_ids) == 1 else None)
    a = bot.act(obs)
    debug = {"why": a.reason[:160], "opp": archetype,
             "think_ms": round((time.perf_counter() - t0) * 1000)}
    if bot.jam_read is not None:
        debug["jam_bluff"] = round(bot.jam_read, 2)
    if bot.shove_read is not None:
        debug["shove_read"] = [round(f, 3) for f in bot.shove_read]
    if rev_debug is not None:
        debug["rev"] = rev_debug

    # --- 4. Translate, with the engine's traps avoided ----------------------
    if a.kind == "raise" and can_raise:
        to = max(min_to, min(max_to, int(a.raise_to)))
        if to >= all_in_to:
            return {"action": "all_in", "debug": debug}
        if to > current:
            return {"action": "raise", "amount": to, "debug": debug}
    if a.kind in ("call", "raise") and owed > 0:
        return {"action": "call", "debug": debug}
    if owed == 0:
        return {"action": "check", "debug": debug}   # never fold for free
    return {"action": "fold", "debug": debug}
