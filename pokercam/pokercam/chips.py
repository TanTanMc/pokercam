"""Find poker chips that have a dark body with coloured wedges round the edge.

A chip is recognised by its shape (a dark disc of the right size with eight
evenly spaced lighter wedges round the rim); its value comes from the colour
of those wedges.
"""
import cv2
import numpy as np

N_ANGLES = 96
WEDGES = 8              # coloured wedges round the rim of a chip
MIN_REGULAR = 0.40      # how much of the rim pattern must be "eight evenly spaced wedges"
_ANG = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
_COS, _SIN = np.cos(_ANG), np.sin(_ANG)
_JITTER = sorted(((k, dx, dy) for k in (1.0, 0.93, 1.07) for dx in (-1, 0, 1) for dy in (-1, 0, 1)),
                 key=lambda j: (abs(j[1]) + abs(j[2]), j[0] != 1.0))     # centre first


def _ring(lab, cx, cy, rad):
    """Lab values on a circle of the given radius (N_ANGLES x 3)."""
    h, w = lab.shape[:2]
    xs = np.clip(np.rint(cx + rad * _COS).astype(int), 0, w - 1)
    ys = np.clip(np.rint(cy + rad * _SIN).astype(int), 0, h - 1)
    return lab[ys, xs]


def check(lab, cx, cy, r):
    """Is there a chip at this circle? Returns (how sure 0..1, wedge colour in Lab) or None."""
    body_px = np.concatenate([_ring(lab, cx, cy, r * f) for f in (0.64, 0.69, 0.74)])
    body = np.median(body_px, axis=0)
    if body[0] > 135 or abs(body[1] - 128) + abs(body[2] - 128) > 26:
        return None             # the body of the chip must be dark and colourless
    if body_px[:, 0].std() > 26 or np.median(_ring(lab, cx, cy, r * 0.35)[:, 0]) > 150:
        return None             # ... and the same all the way round
    rings = np.stack([_ring(lab, cx, cy, r * f) for f in (0.84, 0.89, 0.94)])   # 3 x N x 3
    rim = rings.mean(axis=0)
    profile = np.linalg.norm(rim - body, axis=1)
    spec = np.abs(np.fft.rfft(profile - profile.mean()))
    if spec[WEDGES] * 2 / N_ANGLES < 10:
        return None             # no wedges to speak of
    regular = float((spec[WEDGES] ** 2 + spec[2 * WEDGES] ** 2) / (spec[1:] ** 2).sum())
    if regular < MIN_REGULAR:
        return None             # the wedges must be evenly spaced round the rim
    peaks = profile >= np.percentile(profile, 70)       # the middles of the wedges
    wide = np.stack([_ring(lab, cx, cy, r * f) for f in (0.80, 0.85, 0.90, 0.95)])[:, peaks]
    dist = np.linalg.norm(wide - body, axis=2)
    dist[wide[:, :, 0] < body[0] + 10] = 0
    strongest = wide[dist.argmax(axis=0), np.arange(wide.shape[1])]
    top = dist.max(axis=0)
    colour = strongest[top >= np.median(top)].mean(axis=0)   # the purest wedge pixels
    around = np.median(np.concatenate([_ring(lab, cx, cy, r * f) for f in (1.25, 1.4)]), axis=0)
    if colour[0] < body[0] + 15 or np.linalg.norm(colour - around) < 20:
        return None             # wedges are lighter than the body and not just the table showing
    return regular, colour


def _recentre(lab, cx, cy, r):
    """A better centre for a chip: the middle of its dark body."""
    h, w = lab.shape[:2]
    x0, y0 = max(0, int(cx - 1.4 * r)), max(0, int(cy - 1.4 * r))
    x1, y1 = min(w, int(cx + 1.4 * r) + 1), min(h, int(cy + 1.4 * r) + 1)
    win = lab[y0:y1, x0:x1]
    body = np.median(_ring(lab, cx, cy, r * 0.69), axis=0)
    dark = (np.linalg.norm(win - body, axis=2) < 30).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[i, cv2.CC_STAT_AREA]
    if not 0.35 * np.pi * r * r < area < 1.3 * np.pi * r * r:
        return None
    return cents[i][0] + x0, cents[i][1] + y0


def find(bgr, r_chip):
    """All chips in the picture: [{"xy", "r", "lab"}]. r_chip = expected radius in pixels."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.medianBlur(lab[:, :, 0], 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=r_chip * 1.5,
                               param1=80, param2=16,
                               minRadius=int(r_chip * 0.8), maxRadius=int(r_chip * 1.25) + 1)
    out = []
    if circles is None:
        return out
    lab = lab.astype(np.float32)
    for cx, cy, r in circles[0][:80]:
        if np.median(_ring(lab, cx, cy, r * 0.69)[:, 0]) > 150:
            continue
        centres = [(cx, cy)]
        better = _recentre(lab, cx, cy, r)
        if better is not None:
            centres.append(better)
        step = max(1.0, 0.07 * r)       # try a little either side as well
        best = None
        for bx, by in reversed(centres):
            for k, dx, dy in _JITTER:
                res = check(lab, bx + dx * step, by + dy * step, r * k)
                if res is not None and (best is None or res[0] > best[0]):
                    best = res + ((bx, by),)
            if best is not None and best[0] > 0.6:
                break               # clearly a chip, well centred: no need to look further
        if best is not None:
            out.append({"xy": (float(best[2][0]), float(best[2][1])), "r": float(r),
                        "lab": [float(v) for v in best[1]]})
    return out
