"""Generate the black-and-white PDFs to print.

    python make_printables.py            # A4
    python make_printables.py --letter   # US Letter

Output (in ./print):
    1_cards_duplex.pdf   52 playing cards, fronts and backs interleaved for
                         double-sided printing (flip on LONG edge).
    2_table_markers.pdf  Seat markers 1-10, BOARD marker, DEALER button.
    3_back_stickers.pdf  For a real deck: one identical sticker for the BACK of
                         every card. The camera reads the faces by itself.
    4_unique_back_stickers.pdf
                         Optional: 52 DIFFERENT back stickers (numbered 1-52). Stick any
                         one on any card; while teaching the deck you show each card's face
                         and then its back, and the camera pairs them. A face-down card is
                         then known (live win odds before anyone shows).
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

from pokercam import markers as M

DPI = 300
FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
]


def mm(v):
    return int(round(v / 25.4 * DPI))


def font(size_mm):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, mm(size_mm))
    return ImageFont.load_default()


def marker(mid, side_mm):
    return Image.fromarray(M.marker_image(mid, mm(side_mm)))


def new_page(size):
    return Image.new("L", size, 255)


def centered_text(draw, xy, text, fnt, **kw):
    box = draw.textbbox((0, 0), text, font=fnt)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.text((xy[0] - w // 2 - box[0], xy[1] - h // 2 - box[1]), text, font=fnt, **kw)


# ---------------------------------------------------------------- cards
CARD_W, CARD_H = 63.0, 88.0
CARD_MARKER = 40.0


def corner_index(card):
    """Small image with the rank over the suit, for a card corner."""
    img = Image.new("L", (mm(13), mm(21)), 255)
    d = ImageDraw.Draw(img)
    rank = "10" if card[0] == "T" else card[0]
    centered_text(d, (img.width // 2, mm(5.5)), rank, font(8), fill=0)
    red = card[1] in "dh"  # "red" suits are drawn as outlines
    centered_text(d, (img.width // 2, mm(15)), M.SUIT_GLYPH[card[1]], font(8),
                  fill=255 if red else 0, stroke_width=mm(0.35) if red else 0, stroke_fill=0)
    return img


def draw_card_front(card):
    img = Image.new("L", (mm(CARD_W), mm(CARD_H)), 255)
    d = ImageDraw.Draw(img)
    m = marker(M.card_id(card), CARD_MARKER)
    img.paste(m, ((img.width - m.width) // 2, (img.height - m.height) // 2))
    idx = corner_index(card)
    img.paste(idx, (mm(2), mm(2)))
    img.paste(idx.rotate(180), (img.width - idx.width - mm(2), img.height - idx.height - mm(2)))
    d.rectangle([0, 0, img.width - 1, img.height - 1], outline=160, width=2)
    return img


def draw_card_back():
    img = Image.new("L", (mm(CARD_W), mm(CARD_H)), 255)
    d = ImageDraw.Draw(img)
    # decorative frame (kept well away from the marker's white quiet zone)
    d.rectangle([mm(3), mm(3), img.width - mm(3), img.height - mm(3)], outline=0, width=mm(1.2))
    d.rectangle([mm(5.5), mm(5.5), img.width - mm(5.5), img.height - mm(5.5)], outline=0, width=mm(0.4))
    m = marker(M.CARD_BACK_ID, CARD_MARKER)
    img.paste(m, ((img.width - m.width) // 2, (img.height - m.height) // 2))
    d.rectangle([0, 0, img.width - 1, img.height - 1], outline=160, width=2)
    return img


def card_pages(page_size):
    cards = [r + s for s in M.SUITS for r in M.RANKS]
    cw, ch = mm(CARD_W), mm(CARD_H)
    x0 = (page_size[0] - 3 * cw) // 2   # centred grid -> backs line up when duplexed
    y0 = (page_size[1] - 3 * ch) // 2
    back = draw_card_back()
    pages = []
    for i in range(0, 52, 9):
        front_pg, back_pg = new_page(page_size), new_page(page_size)
        for n, card in enumerate(cards[i:i + 9]):
            col, row = n % 3, n // 3
            front_pg.paste(draw_card_front(card), (x0 + col * cw, y0 + row * ch))
            # mirror the column so the back lands behind its front
            back_pg.paste(back, (x0 + (2 - col) * cw, y0 + row * ch))
        pages += [front_pg, back_pg]
    return pages


# -------------------------------------------------------- table markers
def table_tile(mid, title, subtitle):
    w, h = mm(80), mm(84)
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    m = marker(mid, M.TABLE_MARKER_MM)
    img.paste(m, ((w - m.width) // 2, mm(8)))
    centered_text(d, (w // 2, mm(66)), title, font(7), fill=0)
    centered_text(d, (w // 2, mm(75)), subtitle, font(3), fill=90)
    d.rectangle([0, 0, w - 1, h - 1], outline=170, width=2)
    return img


def table_pages(page_size):
    tiles = [table_tile(M.SEAT_ID_BASE + i, "SEAT %d" % (i + 1), "place in front of the player, near the rail")
             for i in range(M.MAX_SEATS)]
    tiles.append(table_tile(M.BOARD_ID, "BOARD / POT", "centre of the table, next to the community cards"))
    tiles.append(table_tile(M.DEALER_ID, "DEALER", "dealer button - move it each hand"))
    tw, th = tiles[0].size
    x0 = (page_size[0] - 2 * tw) // 2
    y0 = (page_size[1] - 3 * th) // 2
    pages = []
    for i in range(0, len(tiles), 6):
        pg = new_page(page_size)
        for n, t in enumerate(tiles[i:i + 6]):
            pg.paste(t, (x0 + (n % 2) * tw, y0 + (n // 2) * th))
        pages.append(pg)
    return pages


# -------------------------------------------------------------- stickers
STICKER = 28.0


def sticker_pages(page_size, unique=False):
    cell_w, cell_h = mm(38), mm(40)
    cols = page_size[0] // cell_w - 0
    cols = min(cols, 5)
    rows = (page_size[1] - mm(16)) // cell_h
    per = cols * rows
    if unique:
        items = [(i, "%d" % (i + 1)) for i in range(52)]
    else:
        items = [(M.CARD_BACK_ID, "back")] * (2 * cols * rows)   # 52 + spares
    x0 = (page_size[0] - cols * cell_w) // 2
    y0 = (page_size[1] - rows * cell_h) // 2
    pages = []
    f = font(3.2)
    for i in range(0, len(items), per):
        pg = new_page(page_size)
        d = ImageDraw.Draw(pg)
        for n, (mid, label) in enumerate(items[i:i + per]):
            x, y = x0 + (n % cols) * cell_w, y0 + (n // cols) * cell_h
            m = marker(mid, STICKER)
            pg.paste(m, (x + (cell_w - m.width) // 2, y + mm(4)))
            centered_text(d, (x + cell_w // 2, y + mm(4) + m.height + mm(2.6)), label, f, fill=0)
            d.rectangle([x, y, x + cell_w, y + cell_h], outline=200, width=1)
        pages.append(pg)
    return pages


def save_pdf(pages, path):
    pages[0].save(path, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
    print("wrote %s (%d pages)" % (path, len(pages)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letter", action="store_true", help="US Letter instead of A4")
    args = ap.parse_args()
    page = (mm(215.9), mm(279.4)) if args.letter else (mm(210), mm(297))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "print")
    os.makedirs(out, exist_ok=True)
    save_pdf(card_pages(page), os.path.join(out, "1_cards_duplex.pdf"))
    save_pdf(table_pages(page), os.path.join(out, "2_table_markers.pdf"))
    save_pdf(sticker_pages(page), os.path.join(out, "3_back_stickers.pdf"))
    save_pdf(sticker_pages(page, unique=True), os.path.join(out, "4_unique_back_stickers.pdf"))


if __name__ == "__main__":
    main()
