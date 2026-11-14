"""PokerCam: overhead camera -> live poker stats on your phone.

    python app.py                 # use the default camera
    python app.py --camera 1      # another camera (e.g. iPhone Continuity Camera, USB webcam)
    python app.py --demo          # no camera needed: simulated table, ordinary deck with back stickers
    python app.py --demo printed  # simulated table, home-printed marker cards
    python app.py --list-cameras

Then open the address it prints on your phone (same Wi-Fi).
"""
import argparse
import json
import os
import socket
import threading
import time

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

from pokercam.game import Game
from pokercam.markers import pretty
from pokercam.vision import Vision

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")


class Tracker:
    """Owns the camera thread and everything it feeds."""

    def __init__(self, args):
        os.makedirs(DATA, exist_ok=True)
        self.args = args
        self.settings_path = os.path.join(DATA, "settings.json")
        settings = None
        if os.path.exists(self.settings_path):
            with open(self.settings_path) as f:
                settings = json.load(f)
        self.vision = Vision(settings, os.path.join(DATA, "demo_deck.npz" if args.demo else "deck.npz"))
        self.game = Game(os.path.join(DATA, "demo_stats.json" if args.demo else "stats.json"))
        if args.demo and not self.game.names:
            self.game.names = {"1": "David", "2": "Ana", "3": "Mihai", "4": "Elena"}
        self.lock = threading.Lock()
        self.frame = None
        self.obs = None
        self.status = {"camera_ok": False, "message": "Starting camera...", "fps": 0, "demo": bool(args.demo)}

    def save_settings(self):
        with open(self.settings_path, "w") as f:
            json.dump(self.vision.settings, f, indent=1)

    def run(self):
        if self.args.demo:
            from pokercam.sim import Simulator
            sim = Simulator(real_deck=self.args.demo == "real")
            rd = self.vision.reader
            if sim.real and rd.count < 52:      # teach the simulated deck (fast-forward)
                self.status["message"] = "Demo: learning the deck..."
                rd.start_learning()
                t = 0.0
                while rd.learn is not None and t < 130:
                    self.vision.process(sim.learn_frame(t), t)
                    t += 1 / 12.0
                rd.tracks = []
            t0, learn_t0 = time.time(), [None]

            def read():
                if rd.learn is not None:        # "Teach the deck" pressed in the demo: act it out
                    learn_t0[0] = learn_t0[0] or time.time()
                    return True, sim.learn_frame(time.time() - learn_t0[0])
                learn_t0[0] = None
                return True, sim.frame(time.time() - t0)
            cap = None
        else:
            cap = self._open()
            read = lambda: cap.read()
        last, n, tick = 0.0, 0, time.time()
        while True:
            ok, frame = read()
            if not ok or frame is None:
                self.status.update(camera_ok=False, message="Camera not delivering pictures. Check the camera "
                                   "permission for your terminal in System Settings > Privacy > Camera.")
                time.sleep(1.0)
                if cap is not None:
                    cap.release()
                    cap = self._open()
                    read = lambda: cap.read()
                continue
            now = time.time()
            if now - last < 0.08:      # ~12 looks per second is plenty
                time.sleep(0.01)
                continue
            last = now
            with self.lock:
                obs = self.vision.process(frame, now)
                self.game.update(obs, now)
                self.frame, self.obs = frame, obs
            n += 1
            if now - tick >= 2.0:
                self.status.update(camera_ok=True, message="", fps=round(n / (now - tick), 1),
                                   resolution="%dx%d" % (frame.shape[1], frame.shape[0]))
                n, tick = 0, now

    def _open(self):
        cap = cv2.VideoCapture(self.args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        return cap

    def state(self):
        with self.lock:
            snap = self.game.snapshot()
            obs = self.obs or {}
            st = dict(self.status)
            st.update(seats_seen=sorted(obs.get("seats", {})), board_seen=obs.get("board") is not None,
                      markers_now=len(obs.get("markers", [])), board_from=self.vision.board_from,
                      dealer_seen=bool(obs.get("dealer_seat")),
                      px_per_mm=round(self.vision.px_per_mm, 2) if self.vision.px_per_mm else None)
            s = self.vision.settings
            snap["camera"] = st
            snap["deck"] = self.vision.reader.status()
            if snap["deck"]["learning"]:
                snap["deck"]["learning"]["next"] = pretty(snap["deck"]["learning"]["next"])
            snap["settings"] = {"chip_tracking": s["chip_tracking"], "card_marker_mm": s["card_marker_mm"],
                                "chips": [{"name": c["name"], "value": c["value"], "rgb": c["rgb"]} for c in s["chips"]]}
        return snap

    def action(self, a):
        kind, msg = a.get("type"), "ok"
        with self.lock:
            g, s = self.game, self.vision.settings
            if kind == "rename":
                g.set_name(int(a["seat"]), str(a["name"]))
            elif kind == "fold":
                g.toggle_fold(int(a["seat"]))
            elif kind == "end_hand":
                g.end_hand()
            elif kind == "set_winner":
                g.set_winner(int(a["hand_id"]), int(a["seat"]))
            elif kind == "delete_hand":
                g.delete_hand(int(a["hand_id"]))
            elif kind == "reset_stats":
                g.reset()
            elif kind == "learn_start":
                self.vision.reader.start_learning(bool(a.get("backs")))
            elif kind == "learn_skip":
                self.vision.reader.skip_back()
            elif kind == "learn_stop":
                self.vision.reader.stop_learning()
            elif kind == "learn_redo":
                self.vision.reader.redo()
            elif kind == "chip_value":
                for c in s["chips"]:
                    if c["name"] == a["name"]:
                        c["value"] = max(0, int(a["value"]))
                self.save_settings()
            elif kind == "calibrate_chip":
                ok, msg = self.vision.calibrate_chip(str(a["name"]))
                if ok:
                    self.save_settings()
            elif kind == "setting" and a.get("key") in ("chip_tracking", "card_marker_mm"):
                s[a["key"]] = bool(a["value"]) if a["key"] == "chip_tracking" else float(a["value"])
                self.save_settings()
            else:
                msg = "unknown action"
        return msg

    def jpeg(self):
        with self.lock:
            if self.frame is None:
                return None
            img = self.vision.annotate(self.frame, self.obs)
        if img.shape[1] > 960:
            img = cv2.resize(img, (960, int(img.shape[0] * 960.0 / img.shape[1])))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else None


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no traffic is sent; just picks the Wi-Fi interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def list_cameras():
    for i in range(6):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read()
        if ok:
            print("camera %d: %dx%d" % (i, frame.shape[1], frame.shape[0]))
        cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--demo", nargs="?", const="real", default=None, choices=["real", "printed"],
                    help="simulated table, no camera (real = ordinary deck, printed = marker cards)")
    ap.add_argument("--list-cameras", action="store_true")
    args = ap.parse_args()
    if args.list_cameras:
        return list_cameras()

    tracker = Tracker(args)
    threading.Thread(target=tracker.run, daemon=True).start()

    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index():
        return send_from_directory(os.path.join(ROOT, "web"), "index.html")

    @app.get("/api/state")
    def state():
        return jsonify(tracker.state())

    @app.post("/api/action")
    def action():
        return jsonify({"message": tracker.action(request.get_json(force=True) or {})})

    @app.get("/snapshot.jpg")
    def snapshot():
        data = tracker.jpeg()
        if data is None:
            return Response(status=503)
        return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    print("\n  PokerCam is running%s." % (" in DEMO mode" if args.demo else ""))
    print("  On your phone (same Wi-Fi) open:   http://%s:%d" % (lan_ip(), args.port))
    print("  On this computer:                  http://localhost:%d\n" % args.port)
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
