"""Find the real table buttons, so no printed marker is needed for them.

  dealer button   a round disc: pale mint centre with dark lettering, ringed by
                  twelve dark-red segments alternating with mint ones
  ALL IN button   a black triangle with a red spade in it; it marks the middle
                  of the table (community cards and pot)
"""
import cv2
import numpy as np

DEALER_MM = (40, 95)            # allowed diameter of the dealer button
DEALER_SEGMENTS = 12            # red segments round its edge
ALLIN_SIDE_MM = (35, 130)       # allowed side of the black triangle
N = 144
_ANG = np.linspace(0, 2 * np.pi, N, endpoint=False)
_COS, _SIN = np.cos(_ANG), np.sin(_ANG)


def _ring(lab, cx, cy, rad):
    h, w = lab.shape[:2]
    xs = np.clip(np.rint(cx + rad * _COS).astype(int), 0, w - 1)
    ys = np.clip(np.rint(cy + rad * _SIN).astype(int), 0, h - 1)
    return lab[ys, xs].astype(np.float32)


def dealer_score(lab, cx, cy, r):
    """How clearly the red / mint segments show on a circle round (cx, cy): 0..1."""
    centre = np.median(np.concatenate([_ring(lab, cx, cy, r * f) for f in (0.2, 0.4)]), axis=0)
    if centre[0] < 135 or centre[1] > 127:
        return 0.0              # the middle must be pale and not reddish
    best = 0.0
    for f in (0.66, 0.72, 0.78, 0.84):
        a = _ring(lab, cx, cy, r * f)[:, 1]
        spec = np.abs(np.fft.rfft(a - a.mean()))
        k = DEALER_SEGMENTS
        if spec[k] * 2 / N < 8:
            continue            # too faint to be red against mint
        best = max(best, float(spec[k] ** 2 / (spec[1:] ** 2).sum()))
    return best


def find_dealer(lab, ppm):
    """Returns (x, y, radius in px) or None. lab = the frame in OpenCV Lab."""
    lo, hi = DEALER_MM[0] / 2.0 * ppm, DEALER_MM[1] / 2.0 * ppm
    cands = []
    # look for round things in a half-size picture (much quicker), check them at full size
    gray = cv2.medianBlur(cv2.resize(lab[:, :, 0], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), 3)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=lo / 2, param1=80, param2=18,
                               minRadius=int(lo / 2), maxRadius=int(hi / 2) + 1)
    if circles is not None:
        cands += [(2 * c[0], 2 * c[1], 2 * c[2]) for c in circles[0][:40]]
    # the pale middle of the button, with or without the mint segments attached
    pale = ((lab[:, :, 0] > 150) & (lab[:, :, 1] < 124)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(pale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if 0.55 * lo <= r <= hi:
            cands += [(cx, cy, r / 0.66), (cx, cy, r / 0.93)]
    best = None
    near = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    for cx, cy, r in cands[:120]:
        if not lo * 0.9 <= r <= hi * 1.1:
            continue
        mid = np.median(_ring(lab, cx, cy, r * 0.3), axis=0)
        if mid[0] < 135 or mid[1] > 127:
            continue
        # roughly first, then carefully around the best place
        rough = max((dealer_score(lab, cx + dx * 0.1 * r, cy + dy * 0.1 * r, r), dx, dy) for dx, dy in near)
        if rough[0] < 0.10:
            continue
        cx, cy = cx + rough[1] * 0.1 * r, cy + rough[2] * 0.1 * r
        score = max(dealer_score(lab, cx + dx * 0.05 * r, cy + dy * 0.05 * r, r * k)
                    for k in (1.0, 0.92, 1.08) for dx, dy in near)
        if score >= 0.35 and (best is None or score > best[0]):
            best = (score, (float(cx), float(cy), float(r)))
    return best[1] if best else None


def find_allin(lab, ppm):
    """Returns (x, y, side in px) of the black triangle or None."""
    L, a = lab[:, :, 0], lab[:, :, 1]
    dark = (L < 60).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lo, hi = ALLIN_SIDE_MM[0] * ppm, ALLIN_SIDE_MM[1] * ppm
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.3 * lo * lo:
            continue
        hull = cv2.convexHull(c)
        tri = None
        for eps in (0.04, 0.06, 0.09):                  # corners are a little rounded
            poly = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
            if len(poly) == 3:
                tri = poly.reshape(3, 2).astype(np.float32)
                break
        if tri is None:
            continue
        sides = [float(np.linalg.norm(tri[i] - tri[(i + 1) % 3])) for i in range(3)]
        if min(sides) < lo or max(sides) > hi or max(sides) > 1.35 * min(sides):
            continue
        hull_area = cv2.contourArea(hull)
        if not 0.80 <= hull_area / cv2.contourArea(tri) <= 1.25 or area < 0.75 * hull_area:
            continue                                    # not a filled triangle
        mask = np.zeros(L.shape, np.uint8)
        cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        inside = mask > 0
        red = float(((a > 160) & (L > 70) & inside).sum()) / max(1, int(inside.sum()))
        if not 0.015 <= red <= 0.35:
            continue                                    # no red spade inside
        if best is None or area > best[0]:
            cx, cy = tri.mean(axis=0)
            best = (area, (float(cx), float(cy), float(np.mean(sides))))
    return best[1] if best else None
