"""Camera frame -> what is on the table.

Finds the printed markers (card backs, seats), the real dealer button and the
ALL IN button (which marks the middle of the table), works out which seat or
zone each card belongs to, and counts the chips lying in the pot area.
"""
import math
import time

import cv2
import numpy as np

from . import buttons, chips as chipfinder
from . import markers as M
from .faces import FaceReader

CHIP_SET = 2    # bump when DEFAULT_CHIPS changes, so saved settings pick up the new set
DEFAULT_CHIPS = [
    # name = colour of the wedges round the edge of the (black) chip, value,
    # that colour as R,G,B measured from photos  (re-learn from the phone under your own light)
    {"name": "white", "value": 1, "rgb": [251, 239, 234]},
    {"name": "beige", "value": 5, "rgb": [252, 238, 214]},
    {"name": "pink", "value": 25, "rgb": [252, 224, 225]},
    {"name": "blue", "value": 100, "rgb": [137, 177, 245]},
    {"name": "magenta", "value": 500, "rgb": [210, 159, 237]},
    {"name": "green", "value": 1000, "rgb": [139, 226, 217]},
]

DEFAULT_SETTINGS = {
    "chip_set": CHIP_SET,
    "seat_radius_mm": 300,    # a card this close to a seat marker belongs to that seat
    "board_radius_mm": 300,   # a card this close to the ALL IN button is a community card
    "pot_radius_mm": 230,     # chips this close to the ALL IN button are "the pot"
    "chip_diameter_mm": 39,
    "card_marker_mm": 40,     # 40 for the printed cards, 28 for the stickers
    "chip_tracking": True,
    "chips": DEFAULT_CHIPS,
}


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _lab(rgb):
    px = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(float)


class Vision:
    def __init__(self, settings=None, deck_path=None):
        self.reader = FaceReader(deck_path) if deck_path else None
        self.settings = dict(DEFAULT_SETTINGS)
        if settings:
            if settings.get("chip_set") != CHIP_SET:    # saved before the chip set changed
                settings = {k: v for k, v in settings.items() if k not in ("chips", "chip_set")}
            self.settings.update(settings)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 35
        params.adaptiveThreshWinSizeStep = 6
        params.minMarkerPerimeterRate = 0.02
        self.detector = cv2.aruco.ArucoDetector(M.get_dictionary(), params)
        # the camera is fixed, so table markers are remembered while a hand or
        # an arm is covering them
        self.seats = {}
        self.board = None
        self.px_per_mm = None
        self.last_circles = []   # raw chip candidates, for colour calibration
        self.board_from = None   # "button" (the ALL IN button) or "marker" (printed BOARD marker)
        self.dealer_button = None    # (x, y, radius) of the real dealer button
        self.allin_button = None     # (x, y, side) of the ALL IN button
        self._seen = {}          # button name -> [xy, times seen there in a row]
        self._buttons_at = -1.0
        self._chips_at = -1.0
        self._chip_tracks = []   # chips followed over time, so one odd reading does not change the pot
        self._chips = []

    # ------------------------------------------------------------ markers
    def process(self, frame, now=None):
        now = time.time() if now is None else now
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        found = []
        if ids is not None:
            for quad, mid in zip(corners, ids.flatten()):
                q = quad.reshape(4, 2)
                side = float(np.mean([_dist(q[i], q[(i + 1) % 4]) for i in range(4)]))
                found.append({"id": int(mid), "xy": tuple(q.mean(axis=0)), "side": side, "quad": q})

        table_sides = []
        dealer = None
        faces, backs, back_cards = {}, [], []
        known_backs = self.reader.back_to_card if self.reader else {}
        for m in found:
            seat = M.seat_from_id(m["id"])
            card = M.card_from_id(m["id"])
            if seat:
                self.seats[seat] = self._smooth(self.seats.get(seat), m["xy"])
                table_sides.append(m["side"])
            elif m["id"] == M.BOARD_ID:
                self.board, self.board_from = self._smooth(self.board, m["xy"]), "marker"
                table_sides.append(m["side"])
            elif m["id"] == M.DEALER_ID:
                dealer = m["xy"]
            elif m["id"] in known_backs and self.reader.learn is None:
                backs.append(m["xy"])        # a sticker registered to one particular card
                back_cards.append((known_backs[m["id"]], m["xy"]))
            elif card:
                faces[card] = m["xy"]
            elif m["id"] == M.CARD_BACK_ID:
                backs.append(m["xy"])

        if table_sides:
            ppm = float(np.median(table_sides)) / M.TABLE_MARKER_MM
            self.px_per_mm = ppm if self.px_per_mm is None else 0.9 * self.px_per_mm + 0.1 * ppm
        elif self.px_per_mm is None:
            card_sides = [m["side"] for m in found if m["id"] <= M.CARD_BACK_ID]
            if card_sides:
                self.px_per_mm = float(np.median(card_sides)) / self.settings["card_marker_mm"]

        if self.px_per_mm and now - self._buttons_at >= 0.5:      # the buttons rarely move
            self._buttons_at = now
            self._find_buttons(frame, found)
        if dealer is None and self.dealer_button:
            dealer = self.dealer_button[:2]

        obs = {
            "dealer_button": self.dealer_button,
            "allin_button": self.allin_button,
            "seats": dict(self.seats),
            "board": self.board,
            "dealer_seat": self._nearest_seat(dealer)[0] if dealer else None,
            "faces": {},          # card -> ("board",) | ("seat", n)
            "backs": {},          # seat -> number of face-down cards
            "muck_backs": 0,
            "unknown_faces": 0,   # face-up cards on the table that could not be read (yet)
            "face_rects": [],
            "face_xy": [],
            "chips": [],
            "pot": None,
            "px_per_mm": self.px_per_mm,
            "markers": found,
        }
        for card, xy in faces.items():
            zone = self.zone(xy)
            if zone:
                obs["faces"][card] = zone
        for xy in backs:
            zone = self.zone(xy)
            if zone and zone[0] == "seat":
                obs["backs"][zone[1]] = obs["backs"].get(zone[1], 0) + 1
            else:
                obs["muck_backs"] += 1

        obs["back_cards"] = [(c, self.zone(xy)) for c, xy in back_cards]
        if self.reader and (self.px_per_mm or self.reader.learn is not None):
            self._read_faces(frame, found, obs, now)

        if self.settings["chip_tracking"] and self.board is not None and self.px_per_mm:
            if now - self._chips_at >= 0.25:        # a few looks a second is plenty
                self._chips_at = now
                self._chips = self._track_chips(self._find_chips(frame, found, obs["face_xy"]), now)
            obs["chips"] = self._chips
            obs["pot"] = sum(c["value"] for c in obs["chips"])
        return obs

    def _find_buttons(self, frame, found):
        """The real dealer button and ALL IN button (no printed marker needed)."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        ppm = self.px_per_mm
        self.dealer_button = self._steady("dealer", buttons.find_dealer(lab, ppm))
        self.allin_button = self._steady("allin", buttons.find_allin(lab, ppm))
        printed = any(m["id"] == M.BOARD_ID for m in found)
        if self.allin_button and not printed:
            self.board, self.board_from = self._smooth(self.board, self.allin_button[:2]), "button"

    def _steady(self, name, hit):
        """Only believe a button that is seen in the same place a few times running."""
        if hit is None:
            self._seen.pop(name, None)
            return None
        old = self._seen.get(name)
        if old and _dist(old[0], hit) < 12 * self.px_per_mm:
            old[0], old[1] = hit, old[1] + 1
        else:
            self._seen[name] = old = [hit, 1]
        return hit if old[1] >= 3 else None

    def _read_faces(self, frame, found, obs, now):
        """Real card faces (no marker): find them and compare with the learned deck."""
        rd = self.reader
        exclude = [m["xy"] for m in found] + list(self.seats.values())
        if self.board is not None:
            exclude.append(self.board)
        rects = rd.find(frame, self.px_per_mm, exclude)
        if not self.px_per_mm and rd.scale_guess:       # no table markers: take the scale from the card itself
            self.px_per_mm = rd.scale_guess
            rects = rd.find(frame, self.px_per_mm, exclude)
        obs["face_xy"] = [f["xy"] for f in rects]
        if rd.learn is not None:
            near_board = list(rects)
            rd.learn_step(near_board, [m["id"] for m in found if m["id"] <= M.CARD_BACK_ID])
            obs["face_rects"] = [(f["box"], "learning") for f in near_board]
            return
        for f in rects:
            zone = self.zone(f["xy"])
            if not zone:
                continue
            card, best = rd.identify(f, self.px_per_mm, now)
            if card:
                obs["faces"].setdefault(card, zone)
            elif best >= 0.40:
                obs["unknown_faces"] += 1
            obs["face_rects"].append((f["box"], card or "?"))
        rd.forget_old(now)

    @staticmethod
    def _smooth(old, new):
        if old is None or _dist(old, new) > 25:
            return new
        return (0.8 * old[0] + 0.2 * new[0], 0.8 * old[1] + 0.2 * new[1])

    def _nearest_seat(self, xy):
        best, best_d = None, 1e9
        for seat, sxy in self.seats.items():
            d = _dist(xy, sxy)
            if d < best_d:
                best, best_d = seat, d
        return best, best_d

    def zone(self, xy):
        """Which part of the table a point is in."""
        if not self.px_per_mm:
            return None
        s = self.settings
        seat, ds = self._nearest_seat(xy)
        ds_mm = ds / self.px_per_mm
        if self.board is not None:
            db_mm = _dist(xy, self.board) / self.px_per_mm
            if db_mm <= s["board_radius_mm"] and db_mm <= ds_mm:
                return ("board",)
            if seat and ds_mm <= s["seat_radius_mm"]:
                return ("seat", seat)
            return None
        # no BOARD marker printed: anything not near a seat is the board
        if seat and ds_mm <= s["seat_radius_mm"]:
            return ("seat", seat)
        return ("board",)

    # -------------------------------------------------------------- chips
    def _find_chips(self, frame, found, face_xy=()):
        s = self.settings
        ppm = self.px_per_mm
        r_chip = s["chip_diameter_mm"] / 2.0 * ppm
        if r_chip < 5:
            return []   # camera too far / resolution too low to see chips
        R = int(s["pot_radius_mm"] * ppm)
        bx, by = int(self.board[0]), int(self.board[1])
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = max(0, bx - R), max(0, by - R), min(w, bx + R), min(h, by + R)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return []
        self.last_circles = []
        refs = [(c, _lab(c["rgb"])) for c in s["chips"]]
        # keep clear of cards, table markers and the two buttons
        blocked = []
        for m in found:
            is_card = m["id"] <= M.CARD_BACK_ID
            blocked.append((m["xy"], m["side"] * (1.45 if is_card else 0.9)))
        blocked += [(xy, 58 * ppm) for xy in face_xy]   # pips on real card faces are not chips
        if self.dealer_button:
            blocked.append((self.dealer_button[:2], self.dealer_button[2] * 1.1))
        if self.allin_button:
            blocked.append((self.allin_button[:2], self.allin_button[2] * 0.55))
        chips = []
        for c in chipfinder.find(roi, r_chip):
            fx, fy = c["xy"][0] + x0, c["xy"][1] + y0
            if _dist((fx, fy), (bx, by)) > R:
                continue
            if any(_dist((fx, fy), xy) < rad for xy, rad in blocked):
                continue
            colour = np.array(c["lab"])
            self.last_circles.append({"xy": (fx, fy), "lab": c["lab"]})
            # brightness depends on the lighting, so it counts for less than the tint
            gap = lambda ref: float(np.linalg.norm((colour - ref) * (0.2, 1.0, 1.0)))
            best = min(refs, key=lambda cr: gap(cr[1]))
            chips.append({"name": best[0]["name"] if gap(best[1]) <= 30 else None,
                          "xy": (float(fx), float(fy)), "r": c["r"]})
        return chips

    def _track_chips(self, seen, now):
        """Follow each chip over a couple of seconds and report the colour it usually shows."""
        reach = self.settings["chip_diameter_mm"] * 0.3 * self.px_per_mm
        for c in seen:
            track = next((t for t in self._chip_tracks if _dist(t["xy"], c["xy"]) < reach), None)
            if track is None:
                track = {"votes": [], "n": 0}
                self._chip_tracks.append(track)
            track.update(xy=c["xy"], r=c["r"], last=now, n=track["n"] + 1)
            track["votes"] = (track["votes"] + [c["name"]])[-9:]
        self._chip_tracks = [t for t in self._chip_tracks if now - t["last"] < 1.2]
        values = {c["name"]: c["value"] for c in self.settings["chips"]}
        out = []
        for t in self._chip_tracks:
            names = [v for v in t["votes"] if v in values]
            if t["n"] < 2 or not names:
                continue
            name = max(set(names), key=names.count)
            out.append({"name": name, "value": values[name], "xy": t["xy"], "r": t["r"]})
        return out

    def calibrate_chip(self, name):
        """Learn a chip colour from the single chip currently lying in the pot area."""
        if len(self.last_circles) != 1:
            return False, "Put exactly one chip (flat, alone) next to the ALL IN button. I can see %d." % len(self.last_circles)
        lab = np.uint8([[self.last_circles[0]["lab"]]])
        b, g, r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
        for c in self.settings["chips"]:
            if c["name"] == name:
                c["rgb"] = [int(r), int(g), int(b)]
                return True, "Saved colour for %s chips." % name
        return False, "No chip called %s." % name

    # ------------------------------------------------------------ preview
    def annotate(self, frame, obs):
        out = frame.copy()
        ppm = self.px_per_mm or 1.0
        if self.board is not None:
            c = (int(self.board[0]), int(self.board[1]))
            cv2.circle(out, c, int(self.settings["board_radius_mm"] * ppm), (255, 200, 0), 1)
            cv2.circle(out, c, int(self.settings["pot_radius_mm"] * ppm), (0, 215, 255), 1)
        for btn, label in ((obs.get("dealer_button"), "DEALER"), (obs.get("allin_button"), "ALL IN = POT")):
            if btn:
                c = (int(btn[0]), int(btn[1]))
                cv2.circle(out, c, int(btn[2] * (1.0 if label == "DEALER" else 0.62)), (0, 255, 0), 2)
                cv2.putText(out, label, (c[0] - 40, c[1] - int(btn[2] * 0.7) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
                cv2.putText(out, label, (c[0] - 40, c[1] - int(btn[2] * 0.7) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        for seat, xy in self.seats.items():
            c = (int(xy[0]), int(xy[1]))
            cv2.circle(out, c, int(self.settings["seat_radius_mm"] * ppm), (90, 90, 255), 1)
            cv2.putText(out, "SEAT %d" % seat, (c[0] - 34, c[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 255), 2)
        for m in obs["markers"]:
            card = M.card_from_id(m["id"])
            label = card or ("back" if m["id"] == M.CARD_BACK_ID else None)
            cv2.polylines(out, [m["quad"].astype(np.int32)], True, (0, 255, 0), 2)
            if label:
                p = (int(m["xy"][0]) - 14, int(m["xy"][1]) + 6)
                cv2.putText(out, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(out, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for box, label in obs["face_rects"]:
            cv2.polylines(out, [box.astype(np.int32)], True, (255, 0, 255), 2)
            p = (int(box[:, 0].mean()) - 14, int(box[:, 1].mean()) + 6)
            cv2.putText(out, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(out, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        for ch in obs["chips"]:
            cv2.circle(out, (int(ch["xy"][0]), int(ch["xy"][1])), int(ch["r"]), (0, 215, 255), 2)
            cv2.putText(out, str(ch["value"]), (int(ch["xy"][0]) - 8, int(ch["xy"][1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)
        return out
