"""Virtual Ping-Pong with a LEGO Education Double Motor paddle.

Inputs:
    Webcam pose (MediaPipe)  -> is your wrist where the ball arrives?
    Double Motor IMU         -> are you actually swinging (and when)?
    AprilTags (tag36h11)     -> start the game + pick the level:
                                  tag 1 = EASY, tag 2 = MEDIUM, tag 3 = HARD
Output:
    MQTT  ME193/Rogers/<name>  <- best streak as a float, e.g. "7.0"
    Double Motor buzzes on hit / miss (haptic feedback).

Paddles (--paddle):
    lego      LEGO Education Double Motor over Bluetooth (default in this file).
    unoq      Arduino UNO Q + MPU6050 + light sensor over MQTT -- use pingpong_unoq.py,
              which is this same game with the UNO Q as its default paddle.
    keyboard  no hardware (same as --no-paddle).

Hit policy: when the ball reaches the target ring, it's a HIT if the paddle
IMU detected a swing within the timing window around arrival AND one of your
wrists was inside the ring at the moment of that swing. Otherwise MISS.

Run (from the lego_robot_project folder):
    my_env\\Scripts\\python.exe ME193\\pingpong\\pingpong.py
    my_env\\Scripts\\python.exe ME193\\pingpong\\pingpong.py --card-serial 7572 --card-color orange
    my_env\\Scripts\\python.exe ME193\\pingpong\\pingpong.py --no-paddle     (SPACE = swing, no hardware)
    my_env\\Scripts\\python.exe ME193\\pingpong\\pingpong.py --no-mqtt

Fun stuff:
    SMASH        swing much harder than your normal swing -> the robot is
                 SMASH_BONUS more likely to miss (the only way to beat Champ Bot
                 besides golden balls).
    GOLDEN BALL  sometimes the robot serves a golden ball: return it = instant point.
    Sound effects (--mute to turn off) and screen shake.

Keys: 1/2/3 start a level (same as the AprilTags), M = back to menu,
      SPACE = swing, X = smash, C = light sensor (only with --no-paddle), Q / Esc = quit.
"""

import argparse
import math
import os
import random
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from cv2 import aruco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "poserace"))      # pose_camera.py (+ downloaded model)
sys.path.append(os.path.join(HERE, "..", "WhistleSoccer"))  # fallback; pingpong/mqttlib.py (a copy) is used first

import graphics as gfx  # noqa: E402
import pose_camera  # noqa: E402  (only for its model download/path)
from paddle import KeyboardPaddle, Paddle, UnoQPaddle  # noqa: E402
from sfx import Sounds  # noqa: E402
from tracking import CameraThread, HandTracker, PoseThread  # noqa: E402

# ── game tuning ──────────────────────────────────────────────────────────────

LEVELS = {  # AprilTag id -> (name, seconds for the ball to reach you)
    1: ("EASY", 2.2),
    2: ("MEDIUM", 1.6),
    3: ("HARD", 1.1),
}
SPEEDUP = 0.95          # each hit makes the next ball 5% faster...
MIN_TRAVEL_FRAC = 0.55  # ...down to 55% of the level's starting travel time
EARLY_S = 0.30          # a swing up to this long BEFORE the ball arrives counts
LATE_S = 0.25           # ...or up to this long AFTER
PERFECT_S = 0.08        # swing this close to arrival = PERFECT
GREAT_S = 0.16          # ...this close = GREAT, otherwise GOOD
WRIST_MATCH_S = 0.15    # wrist position used = closest frames within this of the swing
CAMERA_LAG_S = 0.03     # sensor/driver delay before a frame reaches us (capture time is stamped)
BOUNCE_AT = 0.6         # fraction of the flight where the ball bounces (after clearing the net)
RETURN_S = 0.7          # your hit flies to the opponent, who hits it straight back
PAUSE_AFTER_HIT_S = 0.0
PAUSE_AFTER_MISS_S = 1.3
COUNTDOWN_S = 3.0
TAG_HOLD_FRAMES = 10    # hold a level tag in view this many frames to start
FIRE_STREAK = 5         # streak at which the ball catches fire
WIN_POINTS = 5          # first to this many points wins the match
BOT_MISS_CHANCE = {1: 0.30, 2: 0.20, 3: 0.0}  # chance the robot misses each of your returns
PAUSE_AFTER_POINT_S = 1.3
GAMEOVER_LOCK_S = 2.0   # ignore tags on the results screen this long (so you don't restart by accident)
SMASH_POWER = 1.35      # a swing this much harder than your calibrated normal swing = SMASH
SMASH_BONUS = 0.25      # ...adds this to the robot's miss chance (even Champ Bot's)
SMASH_DECIDE_S = 0.15   # wait this long after the hit so the swing's peak has been measured
GOLDEN_CHANCE = 0.12    # chance a serve is a golden ball (return it = instant point)

TAG_FAMILY = aruco.DICT_APRILTAG_36h11
WINDOW = "Virtual Ping-Pong"

# Colors (BGR)
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
GREEN, RED, YELLOW, CYAN, ORANGE = (90, 225, 90), (70, 70, 240), (0, 220, 255), (255, 210, 60), (0, 150, 255)
GOLD, GREY = (40, 200, 255), (175, 175, 175)
LEVEL_COLORS = {1: (110, 215, 90), 2: (0, 185, 255), 3: (80, 80, 245)}
BOT_NAMES = {1: "FORREST GUMP", 2: "MARTY SUPREME", 3: "PROF ROGERS"}
BOT_COLORS = {1: (70, 170, 60), 2: (200, 110, 30), 3: (60, 60, 220)}  # scoreboard colour per opponent

# Photo opponents (transparent PNGs in ME193/pingpong/assets). If a photo is missing,
# that level falls back to the LEGO robot. paddle = paddle centre in the original
# image (px); the rest is sizing/placement, see graphics.PhotoOpponent.
OPPONENT_PHOTOS = {
    1: dict(image="forrest.png", paddle=(262, 432), height_frac=0.34, sink_frac=0.145,
            head_frac=0.09, face_box=(0.15, 0.0, 0.47, 0.22),
            lines=dict(taunts=["Mama always said practice!", "I just kept on hitting.", "Run, ball, run!",
                               "That's all I have to say about that."],
                       shocked=["Well, golly!", "Whoa, that's a good one!", "Gee!"],
                       grumpy=["Stupid is as stupid does...", "Aw, shucks.", "I'm not a smart man."],
                       gloat=["I sure do like ping-pong!", "Box of chocolates!", "And that's all I have to say."])),
    2: dict(image="marty.png", paddle=(212, 174), height_frac=0.30, sink_frac=0.094,
            head_frac=0.15, face_box=(0.36, 0.0, 0.82, 0.28),
            lines=dict(taunts=["Too slow for Marty!", "Supreme!", "Is that all you got?", "Watch and learn."],
                       shocked=["Hey, not bad!", "Okay, OKAY!", "Who taught you that?"],
                       grumpy=["That was luck.", "Ugh! Again!", "My grip slipped!", "Rematch. Now."],
                       gloat=["Marty Supreme, baby!", "The greatest!", "Remember the name."])),
    3: dict(image="prof.png", paddle=(385, 295)),
}


# ── MQTT score publisher ─────────────────────────────────────────────────────

class ScorePublisher:
    """Connects in the background so a slow/failed broker never freezes the game."""

    def __init__(self, topic, enabled=True):
        self.topic = topic
        self.client = None
        self.status = "connecting..." if enabled else "off"
        self._latest = None
        if enabled:
            threading.Thread(target=self._connect, daemon=True).start()

    def _connect(self):
        try:
            from mqttlib import MQTTClient
            client = MQTTClient()
            client.connect()
            self.client = client
            self.status = "connected"
            if self._latest is not None:
                self.publish(self._latest)
        except Exception as e:
            self.status = "FAILED"
            print(f"[mqtt] could not connect: {e}")

    def publish(self, best_streak):
        self._latest = best_streak
        if self.client is None:
            return
        payload = str(float(best_streak))
        try:
            self.client.publish(self.topic, payload)
            print(f"[mqtt] {self.topic} <- {payload}")
        except Exception as e:
            print(f"[mqtt] publish failed: {e}")

    def close(self):
        if self.client is not None:
            self.client.disconnect()


# ── game ─────────────────────────────────────────────────────────────────────

class Game:
    def __init__(self, paddle, publisher, width, height, sounds=None, hand_pref="auto"):
        self.paddle = paddle
        self.publisher = publisher
        self.sfx = sounds or Sounds(enabled=False)
        self.w, self.h = width, height

        self.state = "MENU"   # MENU -> COUNTDOWN -> INCOMING -> RETURN / MISSED -> INCOMING ...
        self.t_state = 0.0
        self.level = None
        self.travel = 2.0
        self.streak = 0
        self.best = 0
        self.hits = 0
        self.misses = 0
        self.player_pts = 0
        self.bot_pts = 0
        self.bot_will_miss = False
        self.bot_whiffed = False
        self.bot_decided = True
        self.winner = None
        self.golden = False
        self.smash = False
        self.return_ox = 0.0
        self.last_beep = None

        self.spawn = (0, 0)
        self.target = (0, 0)
        self.radius = 80
        self.serve_t = 0.0
        self.arrive_t = 0.0
        self.return_from = (0, 0)
        self.return_land = (0, 0)  # where your hit bounces on the opponent's side
        self.return_to = (0, 0)    # the opponent's paddle
        self.hit_swing_t = None
        self.bounced = False

        self.tag_candidate, self.tag_count = None, 0
        self.tag_corners = []            # [(id, 4x2 corners)] seen this frame (menu only)
        self.wrists = deque(maxlen=120)  # (t, [(x, y, vis) left, (x, y, vis) right])
        self.pose = None
        self.last_pose_id = None
        self.tracker = HandTracker()  # smoothed + predicted landmarks for drawing the racket
        self.perf = ""                # fps readout set by main()
        self.display_t = None         # capture time of the frame being drawn

        # Which hand holds the paddle (0 = "lw" landmark, 1 = "rw" landmark).
        # Picked during the countdown, then LOCKED for the whole match so only
        # that one hand is ever tracked, drawn, or counted for hits.
        self.hand = 1
        self.hand_locked = False
        self.hand_pref = hand_pref  # "auto" (raised hand) / "left" / "right" (your actual hand)
        self.hand_motion = [0.0, 0.0]
        self.prev_wrist = [None, None]

        # Graphics
        self.table = gfx.Table(width, height)
        self.ball = gfx.Ball()
        self.fx = gfx.Effects(width, height)
        self.robot = gfx.Opponent(self.table, width, height)
        self.photo_opps, self.portraits = {}, {}
        for lvl, cfg in OPPONENT_PHOTOS.items():
            cfg = dict(cfg)
            path = os.path.join(HERE, "assets", cfg.pop("image"))
            try:
                opp = gfx.PhotoOpponent(self.table, width, height, path, cfg.pop("paddle"), **cfg)
            except Exception as e:  # missing/bad photo: that level uses the robot
                print(f"[opponent] could not load {path}: {e}")
                continue
            self.photo_opps[lvl] = opp
            self.portraits[lvl] = opp.portrait(64)
        self.opp = self.robot
        self.ball_trail = deque(maxlen=9)
        self.paddle_trail = deque(maxlen=6)
        self.last_draw = None

    # ── state changes ──

    def go_menu(self, now):
        self.state, self.t_state = "MENU", now
        self.tag_candidate, self.tag_count = None, 0

    def start(self, level, now):
        self.level = level
        self.travel = LEVELS[level][1]
        self.streak = 0
        self.player_pts = self.bot_pts = 0
        self.winner = None
        self.state, self.t_state = "COUNTDOWN", now
        self.hand_locked = False  # re-pick the paddle hand during this countdown
        self.opp = self.photo_opps.get(level, self.robot)
        self.opp.reset(self.w / 2, BOT_COLORS[level], BOT_NAMES[level])
        self.opp.react("idle", now, 2.0, "Let's play!")
        self.paddle.buzz("select")
        self.sfx.play("select")
        self.last_beep = None
        print(f"Starting level {LEVELS[level][0]}")

    def serve(self, now):
        w, h = self.w, self.h
        pose = self.pose
        shoulders_ok = pose and pose["ls"][2] > 0.5 and pose["rs"][2] > 0.5
        if shoulders_ok:
            # Aim somewhere your arm can reach: around your shoulders, scaled by your size.
            cx = (pose["ls"][0] + pose["rs"][0]) / 2
            cy = (pose["ls"][1] + pose["rs"][1]) / 2
            sw = max(60.0, abs(pose["ls"][0] - pose["rs"][0]))
            tx = cx + random.uniform(-1.5, 1.5) * sw
            ty = cy + random.uniform(-0.3, 1.0) * sw
            self.radius = max(55, int(0.7 * sw))
        else:
            tx = random.uniform(0.25, 0.75) * w
            ty = random.uniform(0.40, 0.75) * h
            self.radius = 80
        tx = min(max(tx, 0.08 * w), 0.92 * w)
        ty = min(max(ty, 0.38 * h), 0.92 * h)
        self.target = (tx, ty)
        self.spawn = self.opp.paddle_pos(now)  # the ball comes off the opponent's paddle
        self.opp.swing(now)
        self.golden = random.random() < GOLDEN_CHANCE
        if self.golden:
            self.opp.react("happy", now, 1.0, "Golden ball!")
            self.sfx.play("golden")
        else:
            self.sfx.play("robot_hit")
        self.serve_t = now
        self.arrive_t = now + self.travel
        self.bounced = False
        self.ball_trail.clear()
        self.state, self.t_state = "INCOMING", now

    def on_hit(self, now):
        self.streak += 1
        self.hits += 1
        if self.streak > self.best:
            self.best = self.streak
            self.publisher.publish(self.best)
        self.sfx.play("hit")  # (the haptic buzz waits for the smash check, see decide_return)
        base = LEVELS[self.level][1]
        self.travel = max(base * MIN_TRAVEL_FRAC, self.travel * SPEEDUP)
        x, y, _ = self.ball_pos(now)
        self.return_from = (x, y)
        self.bounced = False
        self.state, self.t_state = "RETURN", now

        # The opponent starts running to where your shot will land. Whether it
        # actually gets there is decided a moment later (decide_return), once we
        # know how hard you swung.
        t = self.table
        ox = random.uniform(t.far_left, t.far_right)
        self.return_ox = ox
        self.return_land = (ox + random.uniform(-0.03, 0.03) * self.w,
                            t.far_y + 0.45 * (t.net_y - t.far_y))
        self.opp.move_to(ox, now, RETURN_S * 0.8)
        self.return_to = self.opp.paddle_pos(now + RETURN_S, ox)
        self.bot_will_miss = False
        self.bot_whiffed = False
        self.bot_decided = False
        self.smash = False

        # Effects: timing grade, sparks, shockwave, streak milestones.
        off = abs(self.hit_swing_t - self.arrive_t) if self.hit_swing_t is not None else 1.0
        grade, color = ("PERFECT!", GOLD) if off < PERFECT_S else ("GREAT!", GREEN) if off < GREAT_S \
            else ("GOOD", CYAN)
        fire = self.streak >= FIRE_STREAK
        self.fx.burst(x, y, n=45 if fire else 30,
                      colors=((0, 90, 255), (0, 200, 255), (60, 60, 255)) if fire else
                      ((0, 220, 255), (0, 150, 255), (255, 255, 255)))
        self.fx.ring(x, y, now, color, r0=self.radius * 0.5, r1=self.radius * 2.2)
        px = min(max(x, 160), self.w - 160)
        py = max(y - self.radius - 30, 150)
        self.fx.popup(grade, px, py, now, color, size=58)
        self.fx.popup(f"x{self.streak}", px, py - 60, now, WHITE, size=40, weight="bold")
        if grade == "PERFECT!":
            self.opp.shocked(now)
        if self.streak % 5 == 0:
            self.opp.shocked(now, "You're on fire!")
            words = {5: "ON FIRE!", 10: "UNSTOPPABLE!", 15: "LEGENDARY!"}
            self.fx.popup(words.get(self.streak, f"{self.streak} IN A ROW!"), self.w / 2, self.h * 0.45,
                          now, ORANGE, size=96, dur=1.3)

    def decide_return(self, now):
        """SMASH check + does the robot reach your shot? Runs SMASH_DECIDE_S after the hit."""
        self.bot_decided = True
        power = self.paddle.swing_power(self.hit_swing_t) if self.hit_swing_t is not None else 0.0
        self.smash = power >= SMASH_POWER
        x, y = self.return_from
        if self.smash:
            self.paddle.buzz("smash")
            self.sfx.play("smash")
            self.fx.shake(now, amp=16, dur=0.35)
            self.fx.burst(x, y, n=50, colors=((0, 60, 255), (0, 160, 255), (0, 230, 255)), speed=750, life=0.8)
            self.fx.popup("SMASH!", min(max(x, 200), self.w - 200), max(y - self.radius - 150, 120), now,
                          (0, 90, 255), size=80, dur=0.9)
        else:
            self.paddle.buzz("hit")

        chance = BOT_MISS_CHANCE[self.level] + (SMASH_BONUS if self.smash else 0.0)
        self.bot_will_miss = self.golden or random.random() < chance
        if self.bot_will_miss:
            # It runs the wrong way and the ball flies past it.
            t = self.table
            ox = self.return_ox
            side = random.choice((-1, 1))
            wrong = ox + side * 0.12 * self.w
            if not (t.far_left - 0.04 * self.w <= wrong <= t.far_right + 0.04 * self.w):
                wrong = ox - side * 0.12 * self.w
            self.opp.move_to(wrong, now, max(0.1, RETURN_S * 0.9 - (now - self.t_state)))
            lx, ly = self.return_land
            self.return_to = (lx + (lx - x) * 0.15, -0.08 * self.h)

    def on_miss(self, now, reason):
        x, y, _ = self.ball_pos(now)
        if self.streak >= 3:
            self.fx.popup(f"streak of {self.streak} lost", self.w / 2, self.h * 0.66, now, GREY,
                          size=34, dur=PAUSE_AFTER_MISS_S, weight="bold")
        self.streak = 0
        self.misses += 1
        self.paddle.buzz("miss")
        self.sfx.play("miss")
        self.fx.shake(now, amp=8, dur=0.25)
        self.bot_pts += 1
        self.opp.taunt(now)
        self.travel = LEVELS[self.level][1]
        self.fx.red_flash(now)
        self.fx.burst(x, y, n=14, colors=((80, 80, 220), (120, 120, 120)), speed=260, life=0.5)
        self.fx.popup("MISS", self.w / 2, self.h * 0.50, now, RED, size=90, dur=PAUSE_AFTER_MISS_S)
        self.fx.popup(reason.upper(), self.w / 2, self.h * 0.59, now, WHITE, size=34,
                      dur=PAUSE_AFTER_MISS_S, weight="bold")
        self.state, self.t_state = "MISSED", now
        self.announce_match_point(now)

    def on_bot_miss(self, now):
        """The robot missed your return: point to you."""
        self.player_pts += 1
        self.paddle.buzz("point")
        self.travel = LEVELS[self.level][1]
        if self.golden:
            self.sfx.play("golden_point")
            self.opp.grumpy(now, say="Not fair!")
            title, sub = "GOLDEN POINT!", "you returned the golden ball"
        else:
            self.sfx.play("point")
            self.opp.grumpy(now)
            title = "POINT!"
            sub = "SMASH winner!" if self.smash else f"{BOT_NAMES[self.level].title()} missed!"
        self.fx.popup(title, self.w / 2, self.h * 0.50, now, GOLD, size=90, dur=PAUSE_AFTER_POINT_S)
        self.fx.popup(sub, self.w / 2, self.h * 0.59, now, WHITE, size=34, dur=PAUSE_AFTER_POINT_S,
                      weight="bold")
        self.fx.burst(self.w / 2, self.h * 0.5, n=40,
                      colors=((40, 200, 255), (90, 225, 90), (255, 210, 60), (255, 255, 255)), speed=600)
        self.state, self.t_state = "POINT", now
        self.announce_match_point(now)

    def announce_match_point(self, now):
        lead, trail = max(self.player_pts, self.bot_pts), min(self.player_pts, self.bot_pts)
        if lead == WIN_POINTS - 1 and lead > trail:
            self.fx.popup("MATCH POINT", self.w / 2, self.h * 0.73, now, ORANGE, size=40,
                          dur=PAUSE_AFTER_POINT_S, weight="black")

    def after_point(self, now):
        """Next rally, or the end of the match."""
        if self.player_pts >= WIN_POINTS or self.bot_pts >= WIN_POINTS:
            self.game_over(now)
        else:
            self.serve(now)

    def game_over(self, now):
        self.winner = "player" if self.player_pts >= WIN_POINTS else "bot"
        self.state, self.t_state = "GAMEOVER", now
        self.tag_candidate, self.tag_count = None, 0
        self.sfx.play("win" if self.winner == "player" else "lose")
        if self.winner == "player":
            self.paddle.buzz("win")
            self.opp.grumpy(now, dur=99, say="You win... this time.")
        else:
            self.paddle.buzz("lose")
            self.opp.gloat(now)
        print(f"Match over: YOU {self.player_pts} - {self.bot_pts} {BOT_NAMES[self.level]}  "
              f"({'you win' if self.winner == 'player' else 'robot wins'})")

    # ── ball + hit policy ──

    def progress(self, now):
        return (now - self.serve_t) / self.travel

    def ball_ground(self, p):
        (sx, sy), (tx, ty) = self.spawn, self.target
        return sx + (tx - sx) * p, sy + (ty - sy) * p

    def ball_lift(self, p):
        if p < BOUNCE_AT:
            return math.sin(math.pi * p / BOUNCE_AT) * 0.10 * self.h
        return math.sin(math.pi * (p - BOUNCE_AT) / (1 - BOUNCE_AT)) * 0.12 * self.h

    def ball_pos(self, now):
        """(x, y, radius) of the incoming ball. Flies from the far end of the
        table, clears the net, bounces once, and lands on the target at progress 1."""
        p = self.progress(now)
        x, y = self.ball_ground(p)
        r = 6 + 28 * max(0.0, min(p, 1.3)) ** 1.5  # grows as it comes toward you
        return x, y - self.ball_lift(p), r

    def wrist_distance(self, t):
        """Closest the PADDLE wrist got to the target within WRIST_MATCH_S of time t
        (the other hand is ignored)."""
        best = float("inf")
        for wt, wrists in self.wrists:
            if abs(wt - t) <= WRIST_MATCH_S:
                x, y, vis = wrists[self.hand]
                if vis > 0.4:
                    best = min(best, math.dist((x, y), self.target))
        return best

    def judge(self, now):
        """The policy. Returns "hit", a miss reason string, or None (undecided)."""
        window_start = self.arrive_t - EARLY_S
        window_end = self.arrive_t + LATE_S
        swings = self.paddle.swings_between(window_start, min(now, window_end))
        for s in swings:
            if self.wrist_distance(s) <= self.radius:
                self.hit_swing_t = s
                return "hit"
        # Wait a bit past the window for the paddle's readings to arrive (Wi-Fi
        # paddles deliver late; swings are stamped with when they really happened).
        if now < window_end + getattr(self.paddle, "latency", 0.0):
            return None
        if swings:
            return "wrong spot"
        if self.wrist_distance(self.arrive_t) <= self.radius:
            return "no swing"
        return "missed"

    # ── per-frame update ──

    def update(self, now, frame, pose, detector, pose_t=None, pose_id=None, frame_t=None):
        """pose comes from the background PoseThread: pose_t = capture time of its
        frame, pose_id = frame number (so each pose is only used once).
        frame_t = capture time of `frame` (the picture about to be drawn)."""
        self.display_t = frame_t if frame_t is not None else now
        if pose_t is None:  # simple callers / tests: treat every call as a fresh pose
            pose_t, pose_id = now, (self.last_pose_id or 0) + 1
        if pose_id != self.last_pose_id:
            self.last_pose_id = pose_id
            self.pose = pose
            if pose:
                self.wrists.append((pose_t - CAMERA_LAG_S, [pose["lw"], pose["rw"]]))
                self.track_paddle_hand(pose)
                self.tracker.update(pose, pose_t, pose_id)

        # Paddle's light sensor = a button: start (menu) / rematch (results screen).
        if self.paddle.serve_pressed():
            if self.state == "MENU" or (self.state == "GAMEOVER" and now - self.t_state >= GAMEOVER_LOCK_S):
                self.start(self.level or 2, now)

        if self.state == "MENU":
            self.read_tags(frame, detector, now)
        elif self.state == "COUNTDOWN":
            n = math.ceil(COUNTDOWN_S - (now - self.t_state))
            if n != self.last_beep and n > 0:
                self.last_beep = n
                self.sfx.play("beep")
            if now - self.t_state >= COUNTDOWN_S:
                self.lock_hand(now)
                self.serve(now)
        elif self.state == "INCOMING":
            verdict = self.judge(now)
            if verdict == "hit":
                self.on_hit(now)
            elif verdict is not None:
                self.on_miss(now, verdict)
        elif self.state == "RETURN":
            if not self.bot_decided:
                if now - self.t_state >= SMASH_DECIDE_S:
                    self.decide_return(now)
            elif self.bot_will_miss:
                if not self.bot_whiffed and now - self.t_state >= RETURN_S * 0.75:
                    self.bot_whiffed = True
                    self.opp.swing(now)  # swings at thin air
                if now - self.t_state >= RETURN_S:
                    self.on_bot_miss(now)
            elif now - self.t_state >= RETURN_S + PAUSE_AFTER_HIT_S:
                self.serve(now)
        elif self.state == "MISSED":
            if now - self.t_state >= PAUSE_AFTER_MISS_S:
                self.after_point(now)
        elif self.state == "POINT":
            if now - self.t_state >= PAUSE_AFTER_POINT_S:
                self.after_point(now)
        elif self.state == "GAMEOVER":
            if now - self.t_state >= GAMEOVER_LOCK_S:
                self.read_tags(frame, detector, now)  # show a tag to play again
            if self.winner == "player" and random.random() < 0.15:  # confetti
                self.fx.burst(random.uniform(0.2, 0.8) * self.w, random.uniform(0.2, 0.5) * self.h, n=25,
                              colors=((40, 200, 255), (90, 225, 90), (255, 210, 60), (230, 120, 255)),
                              speed=500, life=1.0)

    def track_paddle_hand(self, pose):
        """Before a match the paddle follows a candidate hand; once locked it never switches."""
        for i, key in enumerate(("lw", "rw")):
            x, y, vis = pose[key]
            if vis > 0.4:
                if self.prev_wrist[i] is not None:
                    d = math.dist((x, y), self.prev_wrist[i])
                    self.hand_motion[i] = 0.9 * self.hand_motion[i] + 0.1 * d
                self.prev_wrist[i] = (x, y)
            else:
                self.prev_wrist[i] = None
                self.hand_motion[i] *= 0.9
        if self.hand_locked:
            return
        if self.state == "COUNTDOWN":
            choice = self.pick_hand(pose)
            if choice is not None:
                self.hand = choice
        else:  # menu: follow whichever hand is moving more, just for show
            other = 1 - self.hand
            if self.hand_motion[other] > 1.8 * self.hand_motion[self.hand] + 2:
                self.hand = other

    def pick_hand(self, pose):
        """auto: the RAISED hand (higher on screen). left/right: by screen side --
        the picture is mirrored, so your right hand is the wrist further right."""
        (lx, ly, lv), (rx, ry, rv) = pose["lw"], pose["rw"]
        if lv < 0.4 and rv < 0.4:
            return None
        if lv < 0.4 or rv < 0.4:
            return 0 if lv >= 0.4 else 1
        if self.hand_pref == "right":
            return 0 if lx > rx else 1
        if self.hand_pref == "left":
            return 0 if lx < rx else 1
        return 0 if ly < ry else 1

    def lock_hand(self, now):
        if self.pose:
            choice = self.pick_hand(self.pose)
            if choice is not None:
                self.hand = choice
        self.hand_locked = True
        self.paddle_trail.clear()
        print(f"Paddle hand locked: landmark {'lw' if self.hand == 0 else 'rw'}")

    def read_tags(self, frame, detector, now):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        seen = None
        self.tag_corners = []
        if ids is not None:
            self.tag_corners = [(int(i), c.reshape(4, 2)) for i, c in zip(ids.flatten(), corners)]
            levels = [i for i, _ in self.tag_corners if i in LEVELS]
            if len(levels) == 1:
                seen = levels[0]
        if seen is not None and seen == self.tag_candidate:
            self.tag_count += 1
        else:
            self.tag_candidate, self.tag_count = seen, (1 if seen is not None else 0)
        if self.tag_candidate is not None and self.tag_count >= TAG_HOLD_FRAMES:
            self.start(self.tag_candidate, now)

    # ── drawing ──

    def draw(self, img, now):
        dt = 0.033 if self.last_draw is None else min(0.1, now - self.last_draw)
        self.last_draw = now

        if self.state == "MENU":
            img[:] = cv2.convertScaleAbs(img, alpha=0.45)
            self.draw_menu(img, now)
            self.draw_paddle(img, now)
        else:
            img[:] = cv2.convertScaleAbs(img, alpha=0.62 if self.state == "COUNTDOWN" else 0.75)
            self.opp.draw(img, now)
            self.table.draw(img)
            if self.state == "INCOMING":
                self.draw_target(img, now)
            self.draw_paddle(img, now)
            if self.state == "INCOMING":
                self.draw_incoming_ball(img, now)
            elif self.state == "RETURN":
                self.draw_return_ball(img, now)
            elif self.state == "MISSED":
                self.draw_missed_ball(img, now)
            elif self.state == "COUNTDOWN":
                self.draw_countdown(img, now)
            if self.state == "GAMEOVER":
                self.draw_game_over(img, now)
            else:
                self.draw_hud(img, now)

        self.fx.draw(img, now, dt)
        self.fx.apply_shake(img, now)

    def ball_kind(self):
        if self.golden:
            return "gold"
        return "fire" if self.streak >= FIRE_STREAK or self.smash else "normal"

    def trail_color(self):
        if self.golden:
            return (60, 200, 255)
        return (0, 80, 255) if self.streak >= FIRE_STREAK or self.smash else (0, 110, 200)

    def draw_ball_with_trail(self, img, x, y, r, opacity=1.0):
        self.ball_trail.append((x, y, r))
        n = len(self.ball_trail)
        pts = [(tx, ty, tr * (0.4 + 0.5 * i / n), 0.55 * i / n) for i, (tx, ty, tr) in enumerate(self.ball_trail)]
        gfx.additive_glow(img, pts[:-1], self.trail_color(), blur=5)
        if self.golden:
            gfx.glow_circle(img, (x, y), r * 1.2, (60, 210, 255), thick=3, glow=16, core=False)
            if random.random() < 0.5:  # sparkles
                self.fx.burst(x, y, n=2, colors=((150, 240, 255), (255, 255, 255)), speed=120, life=0.35)
        elif self.streak >= FIRE_STREAK or self.smash:
            gfx.glow_circle(img, (x, y), r * 1.1, (0, 90, 255), thick=2, glow=12, core=False)
        self.ball.draw(img, x, y, r, self.ball_kind(), opacity)

    def draw_incoming_ball(self, img, now):
        p = self.progress(now)
        if p >= BOUNCE_AT and not self.bounced:
            self.bounced = True
            gx, gy = self.ball_ground(BOUNCE_AT)
            self.fx.ring(gx, gy, now, WHITE, r0=6, r1=40, dur=0.3)
            self.sfx.play("bounce")
        x, y, r = self.ball_pos(now)
        # Shadow on the table (smaller and fainter the higher the ball is).
        gx, gy = self.ball_ground(p)
        if p <= 1 and self.table.on_table(gx, gy):
            height = self.ball_lift(p) / (0.12 * self.h)
            gfx.soft_ellipse(img, (gx, gy), (r * (1.1 - 0.4 * height), r * (0.45 - 0.15 * height)),
                             alpha=0.55 * (1 - 0.6 * height))
        self.draw_ball_with_trail(img, x, y, r)

    def draw_return_ball(self, img, now):
        """Your shot: arcs over the net, bounces on the opponent's side, rises to its paddle."""
        p = (now - self.t_state) / RETURN_S
        if p >= 1:
            return
        (fx, fy), (lx, ly), (ox, oy) = self.return_from, self.return_land, self.return_to
        if p < 0.7:
            q = p / 0.7
            gx, gy = fx + (lx - fx) * q, fy + (ly - fy) * q
            lift = math.sin(math.pi * q) * 0.22 * self.h
        else:
            q = (p - 0.7) / 0.3
            gx, gy = lx + (ox - lx) * q, ly + (oy - ly) * q
            lift = math.sin(math.pi * q) * 0.03 * self.h
            if not self.bounced:
                self.bounced = True
                self.fx.ring(lx, ly, now, WHITE, r0=4, r1=25, dur=0.25)
                self.sfx.play("bounce")
        r = 34 - 27 * p
        if self.table.on_table(gx, gy):
            gfx.soft_ellipse(img, (gx, gy), (r * 0.9, r * 0.35), alpha=0.4)
        self.draw_ball_with_trail(img, gx, gy - lift, r)

    def draw_missed_ball(self, img, now):
        age = now - self.t_state
        if age < 0.6:
            x, y, r = self.ball_pos(now)
            self.ball.draw(img, x, y, r, self.ball_kind(), opacity=1 - age / 0.6)

    def draw_target(self, img, now):
        p = self.progress(now)
        tx, ty = self.target
        near = self.wrist_distance(now - CAMERA_LAG_S) <= self.radius
        color = GREEN if near else YELLOW
        pulse = 0.75 + 0.25 * math.sin(now * 10)
        gfx.glow_circle(img, (tx, ty), self.radius, color, thick=3, glow=14, strength=pulse)
        # crosshair ticks
        for ang in (0, 90, 180, 270):
            a = math.radians(ang)
            r0, r1 = self.radius - 10, self.radius + 10
            cv2.line(img, (int(tx + math.cos(a) * r0), int(ty + math.sin(a) * r0)),
                     (int(tx + math.cos(a) * r1), int(ty + math.sin(a) * r1)), color, 2, cv2.LINE_AA)
        if p < 1:  # shrinking timing ring: swing when it meets the target ring
            gfx.glow_circle(img, (tx, ty), self.radius * (1 + 2 * (1 - p)), color, thick=2, glow=8,
                            strength=0.4 + 0.6 * p)
        if p > 0.75:
            gfx.text(img, "SWING!", (tx, ty - self.radius - 28), 30, color, "black",
                     opacity=min(1.0, (p - 0.75) * 6))

    def draw_paddle(self, img, now):
        # One-Euro filtered + predicted to when the on-screen frame was captured.
        pose = self.tracker.get(self.display_t if self.display_t is not None else now)
        wkey, ekey = ("lw", "le") if self.hand == 0 else ("rw", "re")
        wx, wy, wvis = pose[wkey]
        if wvis < 0.4:
            return
        ex, ey, evis = pose[ekey]
        sw = abs(pose["ls"][0] - pose["rs"][0]) if pose["ls"][2] > 0.5 and pose["rs"][2] > 0.5 else 130
        size = min(75, max(24, 0.3 * sw))
        swinging = bool(self.paddle.swings) and now - self.paddle.swings[-1] < 0.25

        if swinging and len(self.paddle_trail) > 1:
            n = len(self.paddle_trail)
            pts = [(x, y, size * 0.8, 0.5 * (i + 1) / n) for i, (x, y) in enumerate(self.paddle_trail)]
            gfx.additive_glow(img, pts, (60, 60, 230), blur=11)
        bx, by = gfx.draw_paddle(img, (wx, wy), (ex, ey) if evis > 0.4 else None, size,
                                 glow_color=YELLOW if swinging else None)
        self.paddle_trail.append((bx, by))  # (the other hand is deliberately not drawn)

    def draw_countdown(self, img, now):
        w, h = self.w, self.h
        elapsed = now - self.t_state
        n = max(1, math.ceil(COUNTDOWN_S - elapsed))
        f = elapsed - math.floor(elapsed)  # 0..1 within the current second
        cx, cy = w / 2, h * 0.6
        gfx.panel(img, cx - 280, h * 0.33, cx + 280, h * 0.9, (25, 20, 16), 0.6, 28)
        gfx.text(img, "GET READY", (cx, h * 0.39), 44, WHITE, "black")
        lc = LEVEL_COLORS[self.level]
        gfx.text(img, f"{LEVELS[self.level][0]}   vs   {BOT_NAMES[self.level]}", (cx, h * 0.46), 26, lc, "bold")
        cv2.ellipse(img, (int(cx), int(cy)), (90, 90), -90, 0, 360 * (1 - f), lc, 6, cv2.LINE_AA)
        gfx.text(img, str(n), (cx, cy), 120 * (1 + 0.5 * (1 - f) ** 4), WHITE, "black")
        if self.hand_pref == "auto":
            gfx.text(img, "RAISE YOUR PADDLE HAND", (cx, h * 0.81), 30, YELLOW, "black")
            gfx.text(img, "only that hand counts for this match", (cx, h * 0.86), 20, GREY, "regular")
        else:
            gfx.text(img, f"paddle hand: {self.hand_pref.upper()}", (cx, h * 0.81), 28, YELLOW, "black")
            gfx.text(img, "stand where you can reach the whole table", (cx, h * 0.86), 20, GREY, "regular")

    def draw_menu(self, img, now):
        w, h = self.w, self.h

        self.draw_tags(img)

        # Title with a bouncing ball.
        gfx.text(img, "PING PONG", (w / 2, h * 0.15), int(h * 0.13), WHITE, "black")
        edition = {"lego": "LEGO  DOUBLE  MOTOR  EDITION", "unoq": "ARDUINO  UNO Q  EDITION"}.get(
            getattr(self.paddle, "kind", ""), "KEYBOARD  EDITION")
        gfx.text(img, edition, (w / 2, h * 0.25), 24, ORANGE, "bold")
        bx = w / 2 + math.sin(now * 1.6) * w * 0.3
        by = h * 0.33 - abs(math.sin(now * 4.0)) * h * 0.06
        self.ball.draw(img, bx, by, 13)

        gfx.text(img, f"First to {WIN_POINTS} points wins!  Hold up an AprilTag to pick your level",
                 (w / 2, h * 0.43), 28, WHITE, "bold")
        quick = LEVELS[self.level or 2][0]
        hint = "or press 1 / 2 / 3"
        if getattr(self.paddle, "has_button", False):
            hint += f"   ·   or cover the paddle's light sensor to play {quick}"
        gfx.text(img, hint,
                 (w / 2, h * 0.49), 22, GREY, "regular")
        self.draw_level_cards(img, int(h * 0.56))

        self.draw_status_line(img)

    def draw_tags(self, img):
        """Outline any AprilTags in view, labelled with their level."""
        for tag_id, c in self.tag_corners:
            color = LEVEL_COLORS.get(tag_id, GREY)
            pts = c.astype(np.int32)
            cv2.polylines(img, [pts], True, color, 4, cv2.LINE_AA)
            cx, cy = pts.mean(axis=0)
            label = LEVELS[tag_id][0] if tag_id in LEVELS else f"tag {tag_id}?"
            gfx.text(img, label, (cx, pts[:, 1].min() - 22), 28, color, "black")

    def draw_level_cards(self, img, y0):
        w, h = self.w, self.h
        cw, ch, gap = int(w * 0.24), int(h * 0.26), int(w * 0.03)
        x = (w - (3 * cw + 2 * gap)) // 2
        for lvl, (name, secs) in LEVELS.items():
            lc = LEVEL_COLORS[lvl]
            active = self.tag_candidate == lvl
            gfx.panel(img, x, y0, x + cw, y0 + ch, (30, 24, 20), 0.75, 18, lc, 4 if active else 2)
            gfx.text(img, f"TAG {lvl}", (x + cw / 2, y0 + ch * 0.17), 20, GREY, "bold")
            if lvl in self.portraits:
                pb, pa = self.portraits[lvl]
                gfx.blit(img, pb, pa, x + cw - pb.shape[1] - 12, y0 + 12)
            gfx.text(img, name, (x + cw / 2, y0 + ch * 0.42), 46, lc, "black")
            miss = int(BOT_MISS_CHANCE[lvl] * 100)
            gfx.text(img, f"ball {secs:.1f} s", (x + cw / 2, y0 + ch * 0.62), 19, WHITE, "regular")
            gfx.text(img, f"{BOT_NAMES[lvl].title()} misses {miss}%" if miss else
                     f"{BOT_NAMES[lvl].title()} never misses", (x + cw / 2, y0 + ch * 0.76), 17, GREY, "regular")
            if active:
                frac = min(1.0, self.tag_count / TAG_HOLD_FRAMES)
                bx0, by0, bw = x + 20, y0 + ch - 30, cw - 40
                gfx.panel(img, bx0, by0, bx0 + bw, by0 + 12, (60, 60, 60), 0.9, 6)
                gfx.panel(img, bx0, by0, bx0 + max(12, int(bw * frac)), by0 + 12, lc, 1.0, 6)
            x += cw + gap

    def draw_status_line(self, img):
        w, h = self.w, self.h
        p = self.paddle
        if isinstance(p, KeyboardPaddle):
            paddle_ok, paddle_txt = False, "keyboard"
        else:
            paddle_ok = p.connected
            paddle_txt = f"{p.name} " + ("connected" if paddle_ok else "waiting for data...")
            if paddle_ok and not p.calibrated:
                paddle_txt += " (not calibrated)"
        status = [("PADDLE", GREEN if paddle_ok else ORANGE, paddle_txt),
                  ("MQTT", GREEN if self.publisher.status == "connected" else ORANGE, self.publisher.status)]
        sx = w / 2 - 260
        for label, color, value in status:
            cv2.circle(img, (int(sx), int(h * 0.9)), 7, color, -1, cv2.LINE_AA)
            gfx.text(img, f"{label}  {value}", (sx + 16, h * 0.9), 20, WHITE, "bold", anchor="lm")
            sx += 260
        if self.best:
            gfx.text(img, f"session best: {self.best}", (w / 2, h * 0.96), 22, GOLD, "bold")

    def draw_game_over(self, img, now):
        w, h = self.w, self.h
        win = self.winner == "player"
        age = now - self.t_state
        pop = 1 + 0.6 * max(0.0, 1 - age / 0.25) ** 2
        gfx.panel(img, w * 0.2, h * 0.3, w * 0.8, h * 0.93, (25, 20, 16), 0.78, 28, GOLD if win else RED, 3)
        loser_text = f"{BOT_NAMES[self.level].split()[0]} WINS"
        gfx.text(img, "YOU WIN!" if win else loser_text, (w / 2, h * 0.39), 84 * pop, GOLD if win else RED,
                 "black")
        gfx.text(img, f"{self.player_pts}  :  {self.bot_pts}", (w / 2, h * 0.52), 64, WHITE, "black")
        gfx.text(img, f"YOU   vs   {BOT_NAMES[self.level]}  ({LEVELS[self.level][0]})", (w / 2, h * 0.6), 22,
                 GREY, "bold")
        gfx.text(img, f"best streak (this session): {self.best}", (w / 2, h * 0.67), 24, GOLD, "bold")
        if age >= GAMEOVER_LOCK_S:
            gfx.text(img, "Show an AprilTag (or press 1 / 2 / 3) to play again   ·   M  menu",
                     (w / 2, h * 0.75), 20, WHITE, "regular")
            if getattr(self.paddle, "has_button", False):
                gfx.text(img, f"cover the paddle's light sensor = rematch on {LEVELS[self.level][0]}",
                         (w / 2, h * 0.785), 18, GREY, "regular")
            for lvl in LEVELS:
                lc = LEVEL_COLORS[lvl]
                active = self.tag_candidate == lvl
                bx = w / 2 + (lvl - 2) * 170
                gfx.panel(img, bx - 75, h * 0.81, bx + 75, h * 0.88, (30, 24, 20), 0.8, 14, lc, 3 if active else 1)
                gfx.text(img, f"{lvl}  {LEVELS[lvl][0]}", (bx, h * 0.845), 20, lc, "black")
        self.draw_tags(img)

    def draw_hud(self, img, now):
        w, h = self.w, self.h
        fire = self.streak >= FIRE_STREAK

        # Scoreboard (top-left): YOU : BOT points; dots show progress to WIN_POINTS.
        bc = BOT_COLORS[self.level]
        gfx.panel(img, 16, 16, 300, 128, (30, 24, 20), 0.68, 18)
        gfx.text(img, "YOU", (78, 32), 17, GREY, "bold")
        gfx.text(img, BOT_NAMES[self.level].split()[0], (238, 32), 17, GREY, "bold")
        gfx.text(img, str(self.player_pts), (78, 68), 52, WHITE, "black")
        gfx.text(img, ":", (158, 64), 40, GREY, "black")
        gfx.text(img, str(self.bot_pts), (238, 68), 52, bc, "black")
        for i in range(WIN_POINTS):
            for cx, pts, col in ((78, self.player_pts, WHITE), (238, self.bot_pts, bc)):
                x = int(cx + (i - (WIN_POINTS - 1) / 2) * 15)
                if i < pts:
                    cv2.circle(img, (x, 108), 5, col, -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (x, 108), 5, (120, 120, 120), 1, cv2.LINE_AA)

        # Level + speed pills.
        lc = LEVEL_COLORS[self.level]
        gfx.panel(img, 16, 138, 128, 172, (30, 24, 20), 0.68, 17, lc, 2)
        gfx.text(img, LEVELS[self.level][0], (72, 155), 18, lc, "black")
        speed = LEVELS[self.level][1] / self.travel
        gfx.panel(img, 136, 138, 262, 172, (30, 24, 20), 0.68, 17)
        gfx.text(img, f"SPEED  x{speed:.2f}", (199, 155), 16, WHITE, "bold")

        # Streak = continuous hits (your best streak is what goes to MQTT).
        gfx.panel(img, 16, 180, 262, 214, (30, 24, 20), 0.68, 17)
        gfx.text(img, f"STREAK {self.streak}", (82, 197), 17, ORANGE if fire else WHITE, "black")
        gfx.text(img, f"BEST {self.best}", (200, 197), 17, GOLD, "black")

        # Status pills (top-right).
        ok = self.publisher.status == "connected"
        gfx.panel(img, w - 330, 16, w - 16, 52, (30, 24, 20), 0.68, 18)
        cv2.circle(img, (w - 310, 34), 7, GREEN if ok else ORANGE, -1, cv2.LINE_AA)
        gfx.text(img, self.publisher.topic if ok else f"MQTT {self.publisher.status}", (w - 296, 34), 17,
                 WHITE, "bold", anchor="lm")

        # Swing meter (right edge): gyro magnitude. Red tick = swing threshold,
        # orange tick = SMASH level (SMASH_POWER x your normal swing).
        g, thr = self.paddle.gyro_mag(), self.paddle.threshold
        smash_level = self.paddle.typical_peak * SMASH_POWER
        top_val = smash_level * 1.2
        frac = min(1.0, g / top_val) if top_val else 0
        swinging = bool(self.paddle.swings) and now - self.paddle.swings[-1] < 0.25
        x0, y0, bw, bh = w - 52, 80, 26, int(h * 0.42)
        gfx.panel(img, x0 - 8, y0 - 8, x0 + bw + 8, y0 + bh + 40, (30, 24, 20), 0.68, 14)
        gfx.panel(img, x0, y0, x0 + bw, y0 + bh, (60, 60, 60), 0.9, 10)
        fill = int(bh * frac)
        if fill > 4:
            color = (0, 90, 255) if g >= smash_level else YELLOW if swinging else CYAN
            gfx.panel(img, x0, y0 + bh - fill, x0 + bw, y0 + bh, color, 1.0, 10)
        ty = int(y0 + bh * (1 - thr / top_val))
        sy = int(y0 + bh * (1 - smash_level / top_val))
        cv2.line(img, (x0 - 6, ty), (x0 + bw + 6, ty), RED, 3, cv2.LINE_AA)
        cv2.line(img, (x0 - 6, sy), (x0 + bw + 6, sy), (0, 90, 255), 3, cv2.LINE_AA)
        gfx.text(img, "SMASH", (x0 - 10, sy), 12, (0, 140, 255), "black", anchor="rm")
        gfx.text(img, "SWING", (x0 + bw / 2, y0 + bh + 18), 14, WHITE, "bold")

        gfx.text(img, "M  menu     Q  quit", (20, h - 20), 16, GREY, "regular", anchor="lb", opacity=0.8)
        if self.perf:
            gfx.text(img, self.perf, (w - 20, h - 20), 14, GREY, "regular", anchor="rb", opacity=0.7)


# ── main ─────────────────────────────────────────────────────────────────────

def main(default_paddle="lego"):
    """default_paddle: "lego" here; pingpong_unoq.py calls main("unoq")."""
    parser = argparse.ArgumentParser(description="Virtual ping-pong with a LEGO Double Motor paddle")
    parser.add_argument("--name", default="Yash", help="MQTT topic becomes ME193/Rogers/<name>")
    parser.add_argument("--card-serial", default="7572", help="Double Motor connection card serial")
    parser.add_argument("--card-color", default="orange", help="Double Motor connection card color")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--paddle", choices=("lego", "unoq", "keyboard"), default=default_paddle,
                        help=f"lego = Double Motor, unoq = Arduino UNO Q + MPU6050, keyboard = none "
                             f"(default: {default_paddle})")
    parser.add_argument("--broker", default="localhost",
                        help="MQTT broker the UNO Q sends to (default: start/use Mosquitto on this laptop)")
    parser.add_argument("--no-paddle", action="store_true", help="same as --paddle keyboard: SPACE = swing")
    parser.add_argument("--no-mqtt", action="store_true")
    parser.add_argument("--mute", action="store_true", help="no sound effects")
    parser.add_argument("--hand", choices=("auto", "left", "right"), default="auto",
                        help="paddle hand (auto = the hand you raise during the countdown)")
    parser.add_argument("--smooth-pose", action="store_true",
                        help="MediaPipe VIDEO mode (smoother but laggier racket; the old behaviour)")
    args = parser.parse_args()

    kind = "keyboard" if args.no_paddle else args.paddle
    if kind == "keyboard":
        paddle = KeyboardPaddle()
    elif kind == "lego":
        paddle = Paddle(card_serial=args.card_serial, card_color=args.card_color)
        print("Connecting to Double Motor (turn it on so the light blinks)...")
        paddle.connect()
        print(f"Connected! swing threshold = {paddle.threshold:.0f}")
    else:
        paddle = UnoQPaddle(broker=args.broker)
        print("Connecting to the UNO Q paddle over MQTT (start the Ping Pong Paddle app on the board)...")
        paddle.connect(wait_s=5)
        if paddle.connected:
            print(f"UNO Q paddle connected: {paddle.rate:.0f} samples/s, swing threshold {paddle.threshold:.1f} rad/s")
        else:
            print("  no data from the UNO Q yet -- the game will start anyway and pick it up when it arrives")
    if kind != "keyboard" and not paddle.calibrated:
        print(f"  (not calibrated yet -- run: my_env\\Scripts\\python.exe ME193\\pingpong\\test_paddle.py --paddle {kind})")

    topic = f"ME193/Rogers/{args.name}"
    publisher = ScorePublisher(topic, enabled=not args.no_mqtt)
    print(f"Score topic: {topic}")

    try:
        camera = CameraThread(args.camera)  # grabs frames non-stop in the background
    except RuntimeError:
        paddle.close()
        raise
    h, w = camera.shape[:2]
    pose_camera._ensure_downloaded(pose_camera.MODEL_PATH, pose_camera.MODEL_URL)
    poser = PoseThread(camera, pose_camera.MODEL_PATH, smooth=args.smooth_pose)  # pose in the background
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(TAG_FAMILY), aruco.DetectorParameters())
    game = Game(paddle, publisher, w, h, Sounds(enabled=not args.mute), hand_pref=args.hand)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, w, h)
    last_fid = -1
    frames, t_fps, draw_fps = 0, time.monotonic(), 0.0
    try:
        while True:
            frame, t_cap, fid = camera.wait_newer(last_fid)
            last_fid = fid
            frame = frame.copy()  # the camera thread keeps reusing its buffer
            now = time.monotonic()

            pose, pose_t, pose_id = poser.latest()
            game.update(now, frame, pose, detector, pose_t, pose_id, t_cap)
            game.draw(frame, now)
            cv2.imshow(WINDOW, frame)

            frames += 1
            if now - t_fps >= 1.0:
                draw_fps, frames, t_fps = frames / (now - t_fps), 0, now
                game.perf = (f"screen {draw_fps:.0f} fps  ·  camera {camera.fps:.0f} fps  ·  "
                             f"pose {poser.fps:.0f} fps ({poser.ms:.0f} ms)")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            can_start = game.state == "MENU" or (game.state == "GAMEOVER"
                                                 and now - game.t_state >= GAMEOVER_LOCK_S)
            if key in (ord("1"), ord("2"), ord("3")) and can_start:
                game.start(key - ord("0"), now)
            elif key == ord("m"):
                game.go_menu(now)
            elif key == ord(" ") and isinstance(paddle, KeyboardPaddle):
                paddle.fake_swing()
            elif key == ord("x") and isinstance(paddle, KeyboardPaddle):
                paddle.fake_swing(power=1.6)
            elif key == ord("c") and isinstance(paddle, KeyboardPaddle):
                paddle.press_serve()
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nBest streak: {game.best}   hits: {game.hits}   misses: {game.misses}")
        poser.close()
        camera.close()
        cv2.destroyAllWindows()
        paddle.close()
        publisher.close()


if __name__ == "__main__":
    main()
