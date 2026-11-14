"""Turns a stream of table observations into hands, results and player stats.

How a hand is followed (Texas Hold'em):
  * Face-down cards (the shared "back" marker) in front of a seat  -> that player is in.
  * Their cards disappear for a while                             -> they folded.
  * Face-up cards near the BOARD marker                           -> flop / turn / river.
  * Face-up cards in front of a seat                              -> that player's hole cards (showdown).
  * Table empty for a few seconds                                 -> hand over, result saved.
"""
import json
import os
import time

from . import poker
from .markers import pretty

STREETS = ["preflop", "flop", "turn", "river"]
STABLE_FRAMES = 3        # a card must sit still this many frames before it counts
FOLD_TIMEOUT = 8.0       # seconds without seeing a player's cards -> folded
CLEAR_TIMEOUT = 4.0      # seconds of empty table -> hand is over


def street_of(board):
    n = len(board)
    return "preflop" if n < 3 else "flop" if n == 3 else "turn" if n == 4 else "river"


class Game:
    def __init__(self, path):
        self.path = path
        self.names = {}          # seat(str) -> name
        self.history = []        # finished hand records
        self.hand = None
        self.wait_clear = False  # after a manual "end hand", wait for an empty table
        self._pending = {}       # card -> [zone, frames]
        self._start_frames = 0
        self._eq_key, self._eq = None, {}
        self.active_seats = []
        self.dealer_seat = None
        self.pot_now = None
        self._hidden_seen = {}   # (seat, card) -> frames a registered back was seen
        self._load()

    # ------------------------------------------------------- persistence
    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                data = json.load(f)
            self.names = data.get("names", {})
            self.history = data.get("history", [])

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"names": self.names, "history": self.history}, f)
        os.replace(tmp, self.path)

    def name(self, seat):
        return self.names.get(str(seat)) or "Seat %d" % seat

    # ------------------------------------------------------------ update
    def update(self, obs, now=None):
        now = time.time() if now is None else now
        self.active_seats = sorted(obs["seats"])
        if obs["dealer_seat"]:
            self.dealer_seat = obs["dealer_seat"]
        self.pot_now = obs["pot"]

        # only trust cards that have been in the same place for a few frames
        stable = {}
        for card, zone in obs["faces"].items():
            p = self._pending.get(card)
            if p and p[0] == zone:
                p[1] += 1
            else:
                self._pending[card] = p = [zone, 1]
            if p[1] >= STABLE_FRAMES:
                stable[card] = zone
        for card in list(self._pending):
            if card not in obs["faces"]:
                del self._pending[card]

        table_has_cards = bool(obs["faces"]) or bool(obs["backs"]) or bool(obs.get("unknown_faces"))

        if self.hand is None:
            if self.wait_clear:
                if not table_has_cards:
                    self.wait_clear = False
                return
            self._start_frames = self._start_frames + 1 if (stable or obs["backs"]) else 0
            if self._start_frames >= STABLE_FRAMES:
                self._start_hand(now)
            else:
                return

        h = self.hand
        known = set(h["board"]) | {c for cs in h["hole"].values() for c in cs}
        for card, zone in stable.items():
            if card in known:
                continue
            if zone[0] == "board" and len(h["board"]) < 5:
                h["board"].append(card)
            elif zone[0] == "seat":
                cards = h["hole"].setdefault(zone[1], [])
                if len(cards) < 2:
                    cards.append(card)
                    h["players"].add(zone[1])
                    if len(cards) == 2:
                        h["folded"].pop(zone[1], None)   # they showed: they did not fold
        street = street_of(h["board"])

        for seat, n in obs["backs"].items():
            h["uses_backs"] = True
            if street == "preflop":
                h["players"].add(seat)
            if seat in h["players"]:
                if seat in h["folded"] and seat not in h["manual_fold"]:
                    del h["folded"][seat]    # cards were only covered, not folded
                h["last_seen"][seat] = now
                h["seen_street"][seat] = street
        for seat in h["hole"]:
            h["last_seen"][seat] = now
            h["seen_street"][seat] = street

        if h["uses_backs"]:
            for seat in h["players"]:
                if seat in h["folded"] or len(h["hole"].get(seat, [])) == 2:
                    continue
                if now - h["last_seen"].get(seat, h["started"]) > FOLD_TIMEOUT:
                    h["folded"][seat] = h["seen_street"].get(seat, "preflop")

        for card, zone in obs.get("back_cards", []):
            if zone and zone[0] == "seat":
                k = (zone[1], card)
                self._hidden_seen[k] = self._hidden_seen.get(k, 0) + 1
                cards = h["hidden"].setdefault(zone[1], [])
                if self._hidden_seen[k] >= 3 and card not in cards and len(cards) < 2:
                    cards.append(card)

        if obs["dealer_seat"] and street == "preflop":
            h["dealer"] = obs["dealer_seat"]     # the button is often moved just after the deal starts

        if obs["pot"] is not None:
            h["pot"] = obs["pot"]
            # the biggest pot counts, but only a reading that held still for a moment
            run = h["pot_run"]
            run[0], run[1] = obs["pot"], (run[1] + 1 if run[0] == obs["pot"] else 1)
            if run[1] >= 10:
                h["pot_max"] = max(h["pot_max"], obs["pot"])

        if table_has_cards:
            h["last_cards"] = now
        elif now - h["last_cards"] > CLEAR_TIMEOUT:
            self._finish_hand(now)

    def _start_hand(self, now):
        self._start_frames = 0
        self._hidden_seen = {}
        self.hand = {
            "started": now, "board": [], "hole": {}, "players": set(), "folded": {},
            "uses_backs": False, "hidden": {}, "last_seen": {}, "seen_street": {}, "manual_fold": set(),
            "pot": None, "pot_max": 0, "pot_run": [None, 0], "last_cards": now, "dealer": self.dealer_seat,
        }

    @staticmethod
    def _hole(h):
        """Each seat's two cards: face-up ones if shown, else the ones read from their registered backs."""
        out = {s: list(c) for s, c in h["hidden"].items() if len(c) == 2}
        out.update({s: c for s, c in h["hole"].items() if len(c) == 2})
        out.update({s: c for s, c in h["hole"].items() if len(c) < 2 and s not in out})
        return out

    def _players(self, h):
        """Seats dealt into this hand."""
        if h["uses_backs"]:
            return sorted(h["players"])
        # no card backs in use: assume everyone sitting at the table was dealt in
        return sorted(set(self.active_seats) | h["players"])

    def _finish_hand(self, now):
        h, self.hand = self.hand, None
        players = self._players(h)
        if len(players) < 2 and not h["board"]:
            return   # stray card, not a real hand
        contenders = [s for s in players if s not in h["folded"]]
        holes = self._hole(h)
        shown = {s: holes[s] for s in contenders if len(holes.get(s, [])) == 2}
        winners, showdown, hand_names = [], False, {}
        for s, cards in holes.items():
            if len(cards) == 2 and len(h["board"]) >= 3:
                hand_names[str(s)] = poker.describe(poker.evaluate(cards + h["board"]))
        if len(h["board"]) == 5 and shown:
            scores = {s: poker.evaluate(c + h["board"]) for s, c in shown.items()}
            best = max(scores.values())
            winners = [s for s in shown if scores[s] == best]
            showdown = len(contenders) > 1
            for s in contenders:           # stayed to the end but mucked: lost at showdown
                if s not in shown:
                    h["folded"].setdefault(s, "showdown")
        elif len(contenders) == 1:
            winners = contenders
        elif h["uses_backs"] and contenders and not shown:
            # nobody showed: the winner is whoever still had cards after everyone else
            order = sorted(contenders, key=lambda s: h["last_seen"].get(s, 0), reverse=True)
            if len(order) == 1 or h["last_seen"].get(order[0], 0) - h["last_seen"].get(order[1], 0) > 1.5:
                winners = [order[0]]
                for s in order[1:]:
                    h["folded"][s] = h["seen_street"].get(s, "preflop")
        rec = {
            "id": (self.history[-1]["id"] + 1) if self.history else 1,
            "time": now,
            "names": {str(s): self.name(s) for s in players},
            "players": players,
            "folded": {str(s): st for s, st in h["folded"].items() if s in players},
            "board": h["board"],
            "hole": {str(s): c for s, c in holes.items()},
            "hand_names": hand_names,
            "winners": winners,
            "showdown": showdown,
            "pot": h["pot_max"] or None,
            "dealer": h["dealer"],
        }
        self.history.append(rec)
        self.save()

    # ----------------------------------------------------- phone actions
    def set_name(self, seat, name):
        self.names[str(seat)] = name.strip()[:20]
        self.save()

    def toggle_fold(self, seat):
        h = self.hand
        if not h:
            return
        if seat in h["folded"]:
            del h["folded"][seat]
            h["manual_fold"].discard(seat)
            h["last_seen"][seat] = time.time() + 3600   # manual override: don't auto-fold again
        else:
            h["players"].add(seat)
            h["manual_fold"].add(seat)
            h["folded"][seat] = street_of(h["board"])

    def end_hand(self):
        if self.hand:
            self._finish_hand(time.time())
            self.wait_clear = True

    def set_winner(self, hand_id, seat):
        for rec in self.history:
            if rec["id"] == hand_id and seat in rec["players"]:
                rec["winners"] = [seat]
                rec["folded"].pop(str(seat), None)
                self.save()

    def delete_hand(self, hand_id):
        self.history = [r for r in self.history if r["id"] != hand_id]
        self.save()

    def reset(self):
        if self.history:
            os.replace(self.path, self.path + ".%d.bak" % int(time.time()))
        self.history = []
        self.save()

    # ----------------------------------------------------------- outputs
    def stats(self):
        """Per-player numbers, computed from every saved hand."""
        out = {}
        for rec in self.history:
            n_board = len(rec["board"])
            for s in rec["players"]:
                name = rec["names"][str(s)]
                p = out.setdefault(name, dict(name=name, hands=0, saw_flop=0, saw_turn=0, saw_river=0,
                                              showdowns=0, won=0, won_showdown=0, won_no_showdown=0,
                                              fold_preflop=0, fold_postflop=0, chips_won=0, best=None, _best=None))
                p["hands"] += 1
                f = rec["folded"].get(str(s))
                reached = 4 if f in (None, "showdown") else STREETS.index(f)
                if n_board >= 3 and reached >= 1:
                    p["saw_flop"] += 1
                if n_board >= 4 and reached >= 2:
                    p["saw_turn"] += 1
                if n_board >= 5 and reached >= 3:
                    p["saw_river"] += 1
                if f == "preflop":
                    p["fold_preflop"] += 1
                elif f in STREETS:
                    p["fold_postflop"] += 1
                at_showdown = rec["showdown"] and f in (None, "showdown")
                if at_showdown:
                    p["showdowns"] += 1
                if s in rec["winners"]:
                    p["won"] += 1
                    p["won_showdown" if rec["showdown"] else "won_no_showdown"] += 1
                    if rec["pot"]:
                        p["chips_won"] += rec["pot"] / len(rec["winners"])
                cards = rec["hole"].get(str(s), [])
                if len(cards) == 2 and n_board == 5:
                    score = poker.evaluate(cards + rec["board"])
                    if p["_best"] is None or score > p["_best"]:
                        p["_best"], p["best"] = score, poker.describe(score)
        rows = []
        for p in out.values():
            del p["_best"]
            n = p["hands"]
            pct = lambda k, d=n: round(100.0 * p[k] / d) if d else 0
            p["flop_pct"] = pct("saw_flop")
            p["win_pct"] = pct("won")
            p["wtsd_pct"] = pct("showdowns", p["saw_flop"])          # went to showdown, of flops seen
            p["wsd_pct"] = pct("won_showdown", p["showdowns"])       # won at showdown
            p["chips_won"] = round(p["chips_won"])
            if n < 10:
                p["style"] = "Not enough hands yet"
            else:
                loose = "Loose" if p["flop_pct"] >= 45 else "Tight" if p["flop_pct"] <= 25 else "Balanced"
                if p["saw_flop"] >= 5 and p["wtsd_pct"] >= 45:
                    tail = "calls down a lot"
                elif p["saw_flop"] >= 5 and p["wtsd_pct"] <= 20:
                    tail = "gives up after the flop"
                elif p["won"] and p["won_no_showdown"] > p["won_showdown"]:
                    tail = "wins most pots without showing"
                else:
                    tail = "shows down good hands" if p["wsd_pct"] >= 55 and p["showdowns"] >= 3 else "steady"
                p["style"] = "%s, %s" % (loose, tail)
            rows.append(p)
        rows.sort(key=lambda p: (-p["won"], -p["hands"]))
        return rows

    def snapshot(self):
        h = self.hand
        live = None
        if h:
            board = h["board"]
            players = self._players(h)
            contenders = [s for s in players if s not in h["folded"]]
            holes = self._hole(h)
            shown = {s: holes[s] for s in contenders if len(holes.get(s, [])) == 2}
            key = (tuple(sorted((s, tuple(c)) for s, c in shown.items())), tuple(board))
            if key != self._eq_key:
                self._eq_key = key
                dead = [c for s, cs in holes.items() if s not in shown for c in cs]
                self._eq = poker.equity(shown, board, dead) if len(shown) >= 2 else {}
            rows = []
            for s in players:
                cards = holes.get(s, [])
                row = {"seat": s, "name": self.name(s), "cards": [pretty(c) for c in cards],
                       "folded": h["folded"].get(s), "hand": None, "equity": None,
                       "dealer": s == h["dealer"]}
                if len(cards) == 2 and len(board) >= 3:
                    row["hand"] = poker.describe(poker.evaluate(cards + board))
                if s in self._eq:
                    row["equity"] = round(100 * self._eq[s], 1)
                rows.append(row)
            leader = None
            if self._eq:
                top = max(self._eq.values())
                leader = [self.name(s) for s, e in self._eq.items() if e == top]
            live = {"street": street_of(board), "board": [pretty(c) for c in board], "players": rows,
                    "pot": h["pot"], "leader": leader, "uses_backs": h["uses_backs"],
                    "final": len(board) == 5 and len(shown) >= 2}
        last = None
        if self.history:
            r = self.history[-1]
            last = {"id": r["id"], "board": [pretty(c) for c in r["board"]], "pot": r["pot"],
                    "winners": [r["names"][str(s)] for s in r["winners"]],
                    "winning_hand": r["hand_names"].get(str(r["winners"][0])) if r["winners"] else None,
                    "showdown": r["showdown"],
                    "players": [{"seat": s, "name": r["names"][str(s)],
                                 "cards": [pretty(c) for c in r["hole"].get(str(s), [])],
                                 "hand": r["hand_names"].get(str(s)),
                                 "folded": r["folded"].get(str(s))} for s in r["players"]]}
        return {"live": live, "last": last, "stats": self.stats(), "hands_played": len(self.history),
                "seats": [{"seat": s, "name": self.name(s)} for s in self.active_seats],
                "pot_now": self.pot_now, "waiting_for_clear": self.wait_clear}
