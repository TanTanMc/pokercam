"""Read ordinary playing-card faces (no marker on the face).

The camera is shown every card of the deck once ("teaching"); a small picture
of each face is stored. During play, every white card-sized rectangle on the
table is straightened out and compared with the 52 stored pictures.
"""
import os

import cv2
import numpy as np

from .poker import DECK

CARD_W_MM, CARD_H_MM = 63.0, 88.0
PW, PH = 64, 90            # size of the straightened card picture
MARGIN = 5                 # slack for small misalignment when comparing
LEARN_ORDER = [r + s for s in "shdc" for r in "A23456789TJQK"]
LEARN_FRAMES = 6           # frames a card must lie still while being learned
LOCK_FRAMES = 4            # frames of agreement before a card is announced
MIN_SCORE = 0.55
RIVAL_GAP = 0.10          # cards scoring this close to the best are compared in detail
LOOKS_LIKE_A_CARD = 0.40


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class FaceReader:
    def __init__(self, path):
        self.path = path
        self.templates = {}     # card -> PH x PW x 3 picture
        self._inner = {}        # card -> cropped picture used for matching
        self.tracks = []        # cards currently lying on the table
        self.learn = None
        self.scale_guess = None
        self.back_ids = {}       # card -> id of the sticker on its back (when every sticker is different)
        self.hint = ""          # what the camera sees right now, for the phone page
        if os.path.exists(path):
            with np.load(path) as data:
                for card in data.files:
                    if card == "backs":
                        self.back_ids = {DECK[i]: int(v) for i, v in enumerate(data[card]) if v >= 0}
                    else:
                        self._set(card, data[card])

    def _set(self, card, img):
        self.templates[card] = img
        self._inner[card] = np.ascontiguousarray(img[MARGIN:PH - MARGIN, MARGIN:PW - MARGIN])

    def save(self):
        backs = np.array([self.back_ids.get(c, -1) for c in DECK], dtype=np.int16)
        np.savez_compressed(self.path, backs=backs, **self.templates)

    @property
    def back_to_card(self):
        return {v: k for k, v in self.back_ids.items()}

    @property
    def count(self):
        return len(self.templates)

    # ------------------------------------------------------ find + compare
    def find(self, frame, ppm, exclude):
        """White, card-sized rectangles that do not contain a printed marker.

        ppm may be None (scale not known yet): then any rectangle with a card's
        proportions is accepted and its size gives the scale (see self.scale_guess).
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0]
        chroma = np.hypot(lab[:, :, 1].astype(np.float32) - 128, lab[:, :, 2].astype(np.float32) - 128)
        felt = float(np.median(L[::8, ::8]))
        # a card is light AND colourless: pale mint or pink areas of a patterned mat are not cards
        mask = ((L > max(felt + 40, 120)) & (chroma < 18)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 0.0012 * frame.shape[0] * frame.shape[1]
        out, shorts = [], []
        for c in contours:
            area = cv2.contourArea(c)
            rect = cv2.minAreaRect(c)
            short, long_ = sorted(rect[1])
            if short < 5 or area < 0.7 * short * long_:
                continue
            if ppm:
                want_w, want_h = CARD_W_MM * ppm, CARD_H_MM * ppm
                if not 0.6 * want_w * want_h < area < 1.3 * want_w * want_h:
                    continue
                if abs(short / want_w - 1) > 0.18 or abs(long_ / want_h - 1) > 0.18:
                    continue
            else:
                if area < min_area or abs(short / long_ - CARD_W_MM / CARD_H_MM) > 0.06:
                    continue
            if area / (short * long_) < 0.85:
                continue        # not a clean rectangle (two cards touching, a hand on top...)
            box = cv2.boxPoints(rect).astype(np.float32)
            if any(cv2.pointPolygonTest(box, (float(p[0]), float(p[1])), False) >= 0 for p in exclude):
                continue        # has a marker on it: a card back or a printed-marker card
            if _dist(box[0], box[1]) > _dist(box[1], box[2]):
                box = np.roll(box, -1, axis=0)      # make the first edge a short one
            dst = np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]])
            patch = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(box, dst), (PW, PH),
                                        flags=cv2.INTER_AREA)
            out.append({"xy": rect[0], "box": box, "patch": cv2.GaussianBlur(patch, (3, 3), 0)})
            shorts.append(short)
        self.scale_guess = float(np.median(shorts)) / CARD_W_MM if shorts else None
        return out

    def scores(self, patch):
        """How well the picture matches each learned card (best of both ways up)."""
        flipped = cv2.rotate(patch, cv2.ROTATE_180)
        res = np.full(52, -1.0)
        for i, card in enumerate(DECK):
            t = self._inner.get(card)
            if t is not None:
                res[i] = max(cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED).max(),
                             cv2.matchTemplate(flipped, t, cv2.TM_CCOEFF_NORMED).max())
        return res

    def _aligned(self, patch, card):
        """The part of the picture that lines up with a learned card, contrast-normalised."""
        t = self._inner[card]
        best = None
        for img in (patch, cv2.rotate(patch, cv2.ROTATE_180)):
            res = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
            _, score, _, (x, y) = cv2.minMaxLoc(res)
            if best is None or score > best[0]:
                best = (score, img[y:y + t.shape[0], x:x + t.shape[1]])
        return self._norm(best[1]), self._norm(t)

    @staticmethod
    def _norm(img):
        f = img.astype(np.float32)
        return (f - f.mean()) / (f.std() + 1e-6)

    def _beats(self, patch, a, b):
        """Look only at the spots where cards a and b differ: is the picture closer to a?"""
        pa, ta = self._aligned(patch, a)
        pb, tb = self._aligned(patch, b)
        diff = cv2.GaussianBlur(np.abs(ta - tb).max(axis=2), (3, 3), 0)
        differ = diff > max(0.3, 0.5 * float(diff.max()))
        if differ.sum() < 8:
            return False
        ea, eb = np.abs(pa - ta)[differ].mean(), np.abs(pb - tb)[differ].mean()
        return ea < 0.75 * eb

    def identify(self, face, ppm, now):
        """Follow one card over several frames. Returns (card or None, best score)."""
        track = None
        for t in self.tracks:
            if _dist(t["xy"], face["xy"]) < 15 * ppm:
                track = t
                break
        if track is None:
            track = {"xy": face["xy"], "votes": {}, "card": None, "best": 0.0}
            self.tracks.append(track)
        track["xy"], track["last"] = face["xy"], now
        if track["card"] is None and self.templates:
            s = self.scores(face["patch"])
            order = np.argsort(s)[::-1]
            top = int(order[0])
            track["best"] = max(track["best"], float(s[top]))
            if s[top] >= MIN_SCORE:
                # lookalikes (same rank other suit, one pip more or less) get a closer look
                rivals = [int(i) for i in order[1:6] if s[i] >= s[top] - RIVAL_GAP]
                if all(self._beats(face["patch"], DECK[top], DECK[r]) for r in rivals):
                    v = track["votes"]
                    v[DECK[top]] = v.get(DECK[top], 0) + 1
                    others = sum(v.values()) - v[DECK[top]]
                    if v[DECK[top]] >= LOCK_FRAMES and v[DECK[top]] >= 2 * others:
                        track["card"] = DECK[top]
        return track["card"], track["best"]

    def forget_old(self, now):
        self.tracks = [t for t in self.tracks if now - t["last"] < 1.0]

    # ------------------------------------------------------------ teaching
    def start_learning(self, with_backs=False):
        self.learn = {"i": 0, "buf": [], "state": "card", "gone": 0, "xy": None,
                      "with_backs": with_backs, "bid": None, "bn": 0}
        if with_backs:
            self.back_ids = {}

    def stop_learning(self):
        self.learn = None

    def redo(self):
        if self.learn:
            self.learn.update(i=max(0, self.learn["i"] - 1), buf=[], state="remove", gone=0)

    def skip_back(self):
        ln = self.learn
        if ln and ln["state"] == "flip":
            self._next_card(ln)

    def _next_card(self, ln):
        ln.update(i=ln["i"] + 1, buf=[], state="remove", gone=0, bid=None, bn=0)
        if ln["i"] >= len(LEARN_ORDER):
            self.save()
            self.learn = None

    def learn_step(self, faces, codes=()):
        """Call once per frame with the card faces in view and the ids of any card-range codes in view."""
        ln = self.learn
        self.hint = ("I see no card. Lay the card flat, face up, in view of the camera, away from other cards." if not faces
                     else "I see %d cards: lay down just one." % len(faces) if len(faces) > 1 else "Hold still...")
        if ln["state"] == "flip":
            ids = sorted(set(codes))
            self.hint = ("Now turn this card over and show its back (code facing the camera)." if not ids
                         else "I see %d different codes: show just one card." % len(ids) if len(ids) > 1 else "Hold still...")
            if faces or len(ids) != 1:
                ln.update(bid=None, bn=0)
                return
            ln["bn"] = ln["bn"] + 1 if ln["bid"] == ids[0] else 1
            ln["bid"] = ids[0]
            if ln["bn"] >= LEARN_FRAMES:
                taken = self.back_to_card.get(ids[0])
                if ids[0] != 52 and taken is None:
                    self.back_ids[LEARN_ORDER[ln["i"]]] = ids[0]
                self._next_card(ln)     # (the same code on every card means there is nothing to register)
            return
        if ln["state"] == "remove":
            ln["gone"] = ln["gone"] + 1 if not faces else 0
            if ln["gone"] >= 3:
                ln.update(state="card", gone=0, buf=[])
            return
        if len(faces) != 1:
            ln["buf"] = []
            return
        f = faces[0]
        if ln["buf"] and _dist(ln["xy"], f["xy"]) > 3:
            ln["buf"] = []      # still being moved
        patch = f["patch"]
        if ln["buf"]:           # keep every sample the same way up as the first
            first = ln["buf"][0]
            flipped = cv2.rotate(patch, cv2.ROTATE_180)
            same = cv2.matchTemplate(patch, first, cv2.TM_CCOEFF_NORMED)[0, 0]
            if cv2.matchTemplate(flipped, first, cv2.TM_CCOEFF_NORMED)[0, 0] > same:
                patch = flipped
        ln["buf"].append(patch)
        ln["xy"] = f["xy"]
        if len(ln["buf"]) >= LEARN_FRAMES:
            self._set(LEARN_ORDER[ln["i"]], np.median(np.stack(ln["buf"]), axis=0).astype(np.uint8))
            if ln["with_backs"]:
                ln.update(state="flip", buf=[], bid=None, bn=0)
            else:
                self._next_card(ln)

    def status(self):
        ln = self.learn
        return {"count": self.count,
                "hint": self.hint,
                "learning": None if ln is None else {
                    "index": ln["i"], "next": LEARN_ORDER[ln["i"]], "waiting_remove": ln["state"] == "remove",
                    "flip": ln["state"] == "flip", "with_backs": ln["with_backs"]},
                "backs_registered": len(self.back_ids)}
