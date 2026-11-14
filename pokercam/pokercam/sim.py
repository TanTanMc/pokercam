"""A fake overhead camera for demo mode and testing.

Draws a poker table and plays scripted hands, so the whole pipeline
(detector -> hand tracking -> phone page) can be tried without a camera.

  real_deck=False  home-printed cards with a marker on the face and the back,
                   printed BOARD and DEALER markers
  real_deck=True   ordinary cards: normal faces, a marker sticker on the back only;
                   the real dealer button, and the ALL IN button marking the pot

The chips and buttons are cut-outs of photos of the real ones (assets/).
"""
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import markers as M
from .faces import LEARN_ORDER
from .poker import DECK

TABLE_W, TABLE_H = 1280, 720          # table area in mm
FELT = (50, 110, 30)                  # BGR
SEATS = {1: (300, 650), 2: (980, 650), 3: (980, 70), 4: (300, 70)}
BOARD = (640, 250)
CHIPS = ["white", "beige", "pink", "blue", "magenta", "green"]
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
HAND_SECONDS = 28.0
FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"

PIPS = {
    "A": [(.5, .5)], "2": [(.5, .2), (.5, .8)], "3": [(.5, .2), (.5, .5), (.5, .8)],
    "4": [(.3, .2), (.7, .2), (.3, .8), (.7, .8)],
    "5": [(.3, .2), (.7, .2), (.3, .8), (.7, .8), (.5, .5)],
    "6": [(.3, .2), (.7, .2), (.3, .5), (.7, .5), (.3, .8), (.7, .8)],
    "7": [(.3, .2), (.7, .2), (.3, .5), (.7, .5), (.3, .8), (.7, .8), (.5, .35)],
    "8": [(.3, .2), (.7, .2), (.3, .5), (.7, .5), (.3, .8), (.7, .8), (.5, .35), (.5, .65)],
    "9": [(.3, .2), (.7, .2), (.3, .4), (.7, .4), (.3, .6), (.7, .6), (.3, .8), (.7, .8), (.5, .5)],
    "T": [(.3, .2), (.7, .2), (.3, .4), (.7, .4), (.3, .6), (.7, .6), (.3, .8), (.7, .8), (.5, .3), (.5, .7)],
}


def _font(px):
    return ImageFont.truetype(FONT, px) if os.path.exists(FONT) else ImageFont.load_default()


def _text(d, xy, s, f, fill):
    b = d.textbbox((0, 0), s, font=f)
    d.text((xy[0] - (b[2] + b[0]) / 2, xy[1] - (b[3] + b[1]) / 2), s, font=f, fill=fill)


def real_face(card):
    """An ordinary playing-card face, 63x88 mm at 4 px/mm, BGR."""
    k = 4
    w, h = 63 * k, 88 * k
    img = Image.new("RGB", (w, h), (252, 252, 250))
    d = ImageDraw.Draw(img)
    ink = (200, 25, 30) if card[1] in "dh" else (15, 15, 15)
    suit, rank = M.SUIT_GLYPH[card[1]], "10" if card[0] == "T" else card[0]
    if card[0] in PIPS:
        f = _font(30 * k if card[0] == "A" else 15 * k)
        for x, y in PIPS[card[0]]:
            _text(d, (w * x, h * (0.12 + 0.76 * y)), suit, f, ink)
    else:   # court cards: a framed picture area
        d.rectangle([13 * k, 12 * k, w - 13 * k, h - 12 * k], outline=ink, width=k)
        tint = {"J": (250, 215, 120), "Q": (245, 170, 180), "K": (150, 180, 240)}[card[0]]
        d.rectangle([15 * k, 14 * k, w - 15 * k, h - 14 * k], fill=tint)
        _text(d, (w / 2, h * 0.38), rank, _font(30 * k), ink)
        _text(d, (w / 2, h * 0.66), suit, _font(22 * k), ink)
    idx = Image.new("RGB", (11 * k, 20 * k), (252, 252, 250))
    di = ImageDraw.Draw(idx)
    _text(di, (5.5 * k, 5.5 * k), rank, _font(9 * k), ink)
    _text(di, (5.5 * k, 14.5 * k), suit, _font(8 * k), ink)
    img.paste(idx, (int(1.5 * k), int(1.5 * k)))
    img.paste(idx.rotate(180), (w - idx.width - int(1.5 * k), h - idx.height - int(1.5 * k)))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def real_back(mid=None):
    """A patterned card back with the 28 mm marker sticker in the middle."""
    k = 4
    img = np.full((88 * k, 63 * k, 3), (150, 60, 40), np.uint8)
    for i in range(-88 * k, 63 * k, 6 * k):
        cv2.line(img, (i, 0), (i + 88 * k, 88 * k), (190, 110, 90), k // 2)
    side = 28 * k
    x, y = (63 * k - side) // 2, (88 * k - side) // 2
    cv2.rectangle(img, (x - 4 * k, y - 4 * k), (x + side + 4 * k, y + side + 4 * k), (255, 255, 255), -1)
    img[y:y + side, x:x + side] = cv2.cvtColor(M.marker_image(M.CARD_BACK_ID if mid is None else mid, side), cv2.COLOR_GRAY2BGR)
    return img


def printed_card(mid):
    k = 4
    img = np.full((88 * k, 63 * k), 255, np.uint8)
    img[24 * k:64 * k, 11 * k:51 * k] = M.marker_image(mid, 40 * k)
    if mid == M.CARD_BACK_ID:
        cv2.rectangle(img, (3 * k, 3 * k), (60 * k, 85 * k), 0, k)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class Simulator:
    def __init__(self, seed=7, real_deck=False, unique_backs=False):
        self.seed, self.real, self.unique = seed, real_deck, unique_backs
        self.S = 1.5 if real_deck else 1.0            # camera pixels per mm
        self.size = (int(TABLE_W * self.S), int(TABLE_H * self.S))
        self._cache = {}

    def _card(self, card, face_up):
        key = (card, face_up) if (face_up or self.unique) else "back"
        if key not in self._cache:
            if self.real:
                img = real_face(card) if face_up else real_back(M.card_id(card) if self.unique else None)
            else:
                img = printed_card(M.card_id(card) if face_up else M.CARD_BACK_ID)
            self._cache[key] = cv2.resize(img, (int(63 * self.S), int(88 * self.S)), interpolation=cv2.INTER_AREA)
        return self._cache[key]

    def _asset(self, name, size_mm):
        """A photo cut-out (BGRA) scaled so it is size_mm across."""
        if name not in self._cache:
            img = cv2.imread(os.path.join(ASSETS, name + ".png"), cv2.IMREAD_UNCHANGED)
            px = int(round(size_mm * self.S))
            self._cache[name] = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)
        return self._cache[name]

    def _paste(self, canvas, img, centre_mm, angle):
        alpha = None
        if img.shape[2] == 4:
            img, alpha = img[:, :, :3], img[:, :, 3]
        h, w = img.shape[:2]
        cx, cy = centre_mm[0] * self.S, centre_mm[1] * self.S
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        mat[0, 2] += cx - w / 2.0
        mat[1, 2] += cy - h / 2.0
        r = int(max(w, h))          # only touch the area around the card
        x0, y0 = max(0, int(cx) - r), max(0, int(cy) - r)
        x1, y1 = min(canvas.shape[1], int(cx) + r), min(canvas.shape[0], int(cy) + r)
        mat[0, 2] -= x0
        mat[1, 2] -= y0
        size = (x1 - x0, y1 - y0)
        warped = cv2.warpAffine(img, mat, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(np.full((h, w), 255, np.uint8) if alpha is None else alpha, mat, size) > 127
        canvas[y0:y1, x0:x1][mask] = warped[mask]

    def _table(self, n=0):
        img = np.full((self.size[1], self.size[0], 3), FELT, np.uint8)
        side = int(50 * self.S)
        pad = int(8 * self.S)

        def tile(mid, xy):
            t = np.full((side + 2 * pad, side + 2 * pad, 3), 255, np.uint8)
            t[pad:pad + side, pad:pad + side] = cv2.cvtColor(M.marker_image(mid, side), cv2.COLOR_GRAY2BGR)
            self._paste(img, t, xy, 0)
        for seat, xy in SEATS.items():
            tile(M.SEAT_ID_BASE + seat - 1, xy)
        dealer = SEATS[(n % 4) + 1]
        if self.real:       # the real buttons instead of printed markers
            self._paste(img, self._asset("allin", 77), BOARD, 0)
            self._paste(img, self._asset("dealer", 60), (dealer[0] + 95, dealer[1]), 20.0 * n)
        else:
            tile(M.BOARD_ID, BOARD)
            tile(M.DEALER_ID, (dealer[0] + 90, dealer[1]))
        return img

    def _finish(self, img, t):
        img = cv2.GaussianBlur(img, (3, 3), 0)
        noise = np.random.default_rng(int(t * 30)).normal(0, 4, img.shape)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def script(self, n):
        """The n-th demo hand: cards, who folds when."""
        rng = random.Random(self.seed * 1000 + n)
        deck = DECK[:]
        rng.shuffle(deck)
        seats = sorted(SEATS)
        hole = {s: [deck.pop(), deck.pop()] for s in seats}
        order = seats[:]
        rng.shuffle(order)
        spread = 5 if self.real else 14
        return {"hole": hole, "board": [deck.pop() for _ in range(5)],
                "fold_pre": order[0], "fold_flop": order[1], "show": order[2:],
                "angles": [rng.uniform(-spread, spread) for _ in range(20)],
                "flips": [rng.choice([0, 180]) for _ in range(20)],
                "chips": [rng.choice(CHIPS) for _ in range(9)]}

    def frame(self, t):
        n, tt = int(t // HAND_SECONDS), t % HAND_SECONDS
        sc = self.script(n)
        img = self._table(n)
        ang = iter(a + f for a, f in zip(sc["angles"], sc["flips"]))
        if 1 <= tt < 23:
            for seat, (sx, sy) in SEATS.items():
                cy = sy - 120 if sy > TABLE_H / 2 else sy + 120
                face_up = tt >= 17 and seat in sc["show"]
                gone = (seat == sc["fold_pre"] and tt >= 4) or (seat == sc["fold_flop"] and tt >= 10)
                for k, card in enumerate(sc["hole"][seat]):
                    a = next(ang)
                    if not gone:
                        self._paste(img, self._card(card, face_up), (sx - 36 + 72 * k, cy), a)
            shown = 0 if tt < 7 else 3 if tt < 11 else 4 if tt < 14 else 5
            for i, card in enumerate(sc["board"][:shown]):
                self._paste(img, self._card(card, True), (BOARD[0] + (i - 2) * 74, BOARD[1] + 110), next(ang))
            n_chips = 0 if tt < 5 else 3 if tt < 9 else 6 if tt < 12 else 9
            for i, colour in enumerate(sc["chips"][:n_chips]):
                self._paste(img, self._asset("chip_" + colour, 39),
                            (BOARD[0] - 180 + 45 * i, BOARD[1] - 95 - (i % 2) * 44), 37.0 * i)
        return self._finish(img, t)

    def learn_frame(self, t):
        """Teaching the deck: one card per second shown next to the BOARD marker."""
        per = 2.0 if self.unique else 1.0
        i, tt = int(t // per), t % per
        img = self._table()
        if i < len(LEARN_ORDER):
            if tt < 0.7:
                self._paste(img, self._card(LEARN_ORDER[i], True), (BOARD[0] + 20, BOARD[1] + 115), 3.0)
            elif self.unique and 1.1 <= tt < 1.9:
                self._paste(img, self._card(LEARN_ORDER[i], False), (BOARD[0] + 20, BOARD[1] + 115), 3.0)
        return self._finish(img, t)
