"""Marker ID scheme shared by the printables and the vision system.

All markers come from OpenCV's ArUco DICT_4X4_100 dictionary: big, chunky
4x4 black/white squares that print fine on any black-and-white printer and
are far easier for an overhead camera to read than QR codes or barcodes.
"""
import cv2

DICT_ID = cv2.aruco.DICT_4X4_100

RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades
SUIT_GLYPH = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}

CARD_BACK_ID = 52          # same marker on the back of every card
SEAT_ID_BASE = 60          # seats 1..10 -> ids 60..69
MAX_SEATS = 10
BOARD_ID = 70              # marks the middle of the table (board + pot)
DEALER_ID = 71             # dealer button

TABLE_MARKER_MM = 50.0     # printed size of seat / board markers


def card_id(card):
    """'As' -> marker id."""
    return SUITS.index(card[1]) * 13 + RANKS.index(card[0])


def card_from_id(mid):
    """marker id -> 'As' (or None if the id is not a card face)."""
    if 0 <= mid < 52:
        return RANKS[mid % 13] + SUITS[mid // 13]
    return None


def seat_from_id(mid):
    if SEAT_ID_BASE <= mid < SEAT_ID_BASE + MAX_SEATS:
        return mid - SEAT_ID_BASE + 1
    return None


def pretty(card):
    r = "10" if card[0] == "T" else card[0]
    return r + SUIT_GLYPH[card[1]]


def get_dictionary():
    return cv2.aruco.getPredefinedDictionary(DICT_ID)


def marker_image(mid, side_px):
    return cv2.aruco.generateImageMarker(get_dictionary(), mid, side_px)
