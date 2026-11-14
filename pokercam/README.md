# PokerCam

A camera above the poker table reads printed markers on the cards and shows
live results and player statistics on your phone.

## 1. Print (black and white is fine)

Files are in `print/` (re-create with `python make_printables.py`, add `--letter` for US paper).
Print at **100% / actual size**, not "fit to page".

| File | What it is |
|---|---|
| `1_cards_duplex.pdf` | A full 52-card deck. Print **double-sided, flip on long edge** on thick paper (200 gsm+), then cut along the grey lines. Every back carries the same marker, so backs give nothing away. |
| `2_table_markers.pdf` | SEAT 1-10 markers: cut out and tape to the table. (The BOARD/POT and DEALER markers on it are only a fallback: the camera recognises the real ALL IN button and dealer button instead.) |
| `3_back_stickers.pdf` | For an ordinary deck: one identical sticker for the **back** of every card (70 on two pages, 18 spare). Nothing goes on the faces; the camera reads them itself after you teach it the deck once (below). |

## 2. Set up the table

- Mount the camera straight above the table, looking down, whole table in view. 1080p or better.
  A USB webcam or an iPhone (Continuity Camera) both work.
- Tape one SEAT marker in front of each player near the edge of the table.
- Lay the triangular **ALL IN button** flat in the middle of the table: it marks the pot. Deal the community cards right next to it, and push bets next to it too (within about 23 cm).
- Put the round **dealer button** in front of whoever deals, as usual. The camera finds it by its ring of red and mint segments.
- Even lighting, no strong glare on the cards.

## 3. Run

Double-click `start.command` (or `./start.command --camera 1` to pick another camera,
`--list-cameras` to see them, `--demo` to try it with a simulated table and an ordinary deck, `--demo printed` for the marker cards).
macOS will ask to allow camera access the first time.
Open the address it prints on your phone, on the same Wi-Fi. Use the **Setup** tab to check
the camera sees the markers and to type player names.

## Using an ordinary deck (back stickers only)

1. Stick one back sticker in the middle of the back of every card.
2. On the phone open **Setup > Your deck > Teach the deck**. It shows a card; lay that card face up, on its own,
   next to the ALL IN button. When the phone says "take the card away", remove it and lay down the next one.
   52 cards, about 2 minutes, once per deck (and again if you move the camera or change the lighting a lot).
3. Play. Face-up cards are recognised from their real faces; face-down cards from the back sticker.

For this to work the camera needs to be sharp enough: Setup shows "Sharpness", which should be 1.3 pixels per mm
or more (a 1080p camera covering a table up to about 1.4 m wide; use 4K or mount it lower for bigger tables).
Face-up cards must not touch or overlap each other, or the camera cannot see where one ends.

## How to play so the camera can follow

- Keep your two face-down cards in front of your seat marker. Cards gone for 8 seconds = fold.
  (If you cover your cards with a chip or hands for long, tap **Un-fold** on the phone, or just keep playing: showing your cards at the end cancels the fold.)
- At showdown, turn your cards face up in front of your own seat.
- Clear all cards off the table between hands; that ends the hand and saves it.

## What it can and cannot know

- It sees only face-up cards, so hole cards are known at showdown (or all-in when cards are turned over). Live win odds appear from that moment.
- Stats come from who was dealt in, who folded on which street, who reached showdown and who won. It does not see bet sizes, so there is no raise/aggression stat.
- The pot number is an estimate: it counts chips lying flat near the ALL IN button. The chips are the black ones with coloured wedges round the edge (white 1, beige 5, pink 25, blue 100, magenta 500, green 1000); the wedge colour gives the value. Stacked chips count as one. White, beige and pink are close, so teach it your chip colours under your own light in Setup.
- The dealer button and ALL IN button are recognised by their look, so they need to lie flat, uncovered, without strong glare.

Stats are stored in `data/stats.json`.
