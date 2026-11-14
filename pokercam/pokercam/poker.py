"""Texas Hold'em hand evaluation and win odds."""
import itertools
import random

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = [r + s for s in SUITS for r in RANKS]

CATEGORY = ["High card", "Pair", "Two pair", "Three of a kind", "Straight",
            "Flush", "Full house", "Four of a kind", "Straight flush"]
RANK_NAME = ["Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine",
             "Ten", "Jack", "Queen", "King", "Ace"]
RANK_PLURAL = [n + ("es" if n == "Six" else "s") for n in RANK_NAME]


def _straight_high(rank_set):
    """Highest straight's top rank in a set of ranks (0..12), or -1."""
    for hi in range(12, 3, -1):
        if all((hi - k) in rank_set for k in range(5)):
            return hi
    if {12, 0, 1, 2, 3} <= rank_set:
        return 3  # wheel: five-high
    return -1


def evaluate(cards):
    """Best 5-card hand from 5-7 cards. Returns a tuple; bigger is better."""
    ranks = [RANKS.index(c[0]) for c in cards]
    suits = [c[1] for c in cards]
    for s in SUITS:
        if suits.count(s) >= 5:
            fl = sorted((r for r, su in zip(ranks, suits) if su == s), reverse=True)
            sh = _straight_high(set(fl))
            if sh >= 0:
                return (8, sh)
            flush = (5,) + tuple(fl[:5])
            break
    else:
        flush = None

    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # ranks ordered by (count, rank) descending
    groups = sorted(counts, key=lambda r: (counts[r], r), reverse=True)
    top = counts[groups[0]]
    if top == 4:
        kicker = max(r for r in counts if r != groups[0])
        return (7, groups[0], kicker)
    trips = sorted((r for r in counts if counts[r] == 3), reverse=True)
    pairs = sorted((r for r in counts if counts[r] == 2), reverse=True)
    if trips and (len(trips) > 1 or pairs):
        second = max(trips[1:] + pairs)
        return (6, trips[0], second)
    if flush:
        return flush
    sh = _straight_high(set(ranks))
    if sh >= 0:
        return (4, sh)
    singles = sorted(counts, reverse=True)
    if trips:
        k = [r for r in singles if r != trips[0]][:2]
        return (3, trips[0]) + tuple(k)
    if len(pairs) >= 2:
        k = max(r for r in singles if r not in pairs[:2])
        return (2, pairs[0], pairs[1], k)
    if pairs:
        k = [r for r in singles if r != pairs[0]][:3]
        return (1, pairs[0]) + tuple(k)
    return (0,) + tuple(singles[:5])


def describe(score):
    cat = score[0]
    if cat == 8:
        return "Royal flush" if score[1] == 12 else "Straight flush, %s high" % RANK_NAME[score[1]]
    if cat == 7:
        return "Four %s" % RANK_PLURAL[score[1]]
    if cat == 6:
        return "Full house, %s full of %s" % (RANK_PLURAL[score[1]], RANK_PLURAL[score[2]])
    if cat == 5:
        return "Flush, %s high" % RANK_NAME[score[1]]
    if cat == 4:
        return "Straight, %s high" % RANK_NAME[score[1]]
    if cat == 3:
        return "Three %s" % RANK_PLURAL[score[1]]
    if cat == 2:
        return "Two pair, %s and %s" % (RANK_PLURAL[score[1]], RANK_PLURAL[score[2]])
    if cat == 1:
        return "Pair of %s" % RANK_PLURAL[score[1]]
    return "%s high" % RANK_NAME[score[1]]


def equity(hole_by_player, board, dead=(), iterations=1500, rng=None):
    """Win probability for each player whose two hole cards are known.

    hole_by_player: {key: [card, card]}.  Ties are split.
    Exact when two or fewer board cards are still to come, Monte Carlo otherwise.
    """
    keys = list(hole_by_player)
    if not keys:
        return {}
    used = set(board) | set(dead)
    for h in hole_by_player.values():
        used |= set(h)
    stub = [c for c in DECK if c not in used]
    need = 5 - len(board)
    wins = dict.fromkeys(keys, 0.0)

    if need <= 2:
        runouts = itertools.combinations(stub, need)
    else:
        rng = rng or random.Random()
        runouts = (rng.sample(stub, need) for _ in range(iterations))

    total = 0
    for extra in runouts:
        full = list(board) + list(extra)
        scores = {k: evaluate(hole_by_player[k] + full) for k in keys}
        best = max(scores.values())
        winners = [k for k in keys if scores[k] == best]
        for k in winners:
            wins[k] += 1.0 / len(winners)
        total += 1
    return {k: wins[k] / total for k in keys}
