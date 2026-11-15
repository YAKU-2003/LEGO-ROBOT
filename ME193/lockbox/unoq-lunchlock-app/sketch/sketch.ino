// LunchLock latch -- runs on the UNO Q's microcontroller.
//
// TT motor on the Cytron Maker Drive, motor 1. A TT motor has NO position sensor, so the
// position is ESTIMATED from how long the motor has run ("dead reckoning"):
//
//     position 0       = CLOSED  = where the latch is when the board starts / when "home" is set
//     position OPEN_MS = OPEN    = milliseconds of clockwise running from closed
//
// SAFETY: the latch must never turn ANTICLOCKWISE past closed (it would hit the electronics).
//   - nothing can ever drive it below position 0
//   - a normal close stops CLOSE_SHORT_MS before 0 (timing drift can only leave it slightly
//     open, never past closed)
//   - if the board resets while the lid is open it THINKS it is closed, so "close" does nothing
//     (safe). Use jog / the Maker Drive's test buttons to bring it back, then "home".
//
// Wiring (Cytron Maker Drive):
//   M1A -> pin 6 (~PWM)    M1B -> pin 5 (~PWM)
//   VB+ / VB- -> AA battery pack (+ / -)
//   VB- (GND) -> UNO Q GND   <-- shared ground, REQUIRED for the board's signals to work
//   TT motor -> Motor 1 screw terminal
//
// Python (python/main.py) calls these over the Bridge:
//   latch_open()        go to OPEN_MS
//   latch_close()       go to CLOSE_SHORT_MS (just short of closed)
//   latch_jog(ms)       move by ms: + = clockwise (opening), - = anticlockwise (never below 0)
//   latch_home()        "the latch is closed right now" -> position = 0
//   latch_set_open(ms)  change how far "open" goes
//   latch_stop()        stop now
// and the sketch reports  Bridge.notify("latch_state", position_ms, moving)

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

// ---------------- LED matrix = the safe's "LEDs" (built into the UNO Q) ----------------
// python/main.py calls show_lock(state, entered, progress) whenever something changes:
//   state 0 LOCKED (small lock icon)   1 ENTER (3 boxes, filled = digits entered,
//   bottom bar = hold progress)   2 OPEN (tick)   3 WRONG (cross)   4 LOCKED OUT (cross + bar)
// Optional real button: BUTTON_PIN to GND (INPUT_PULLUP). Set BUTTON_PIN to -1 if none.
const int ROWS = 8, COLS = 13;
const int BUTTON_PIN = -1;   // -1 = no button yet (use the laptop:  latch_cli.py -> b ). Set to 2 once wired.
Arduino_LED_Matrix matrix;

const char* ICON_LOCKED[ROWS] = {
  ".....###.....", "....#...#....", "....#...#....", "...#######...",
  "...###.###...", "...###.###...", "...#######...", "............."};
const char* ICON_OPEN[ROWS] = {
  "............#", "...........#.", "..........#..", "#........#...",
  ".#......#....", "..#....#.....", "...#..#......", "....##......."};
const char* ICON_WRONG[ROWS] = {
  "...#.....#...", "....#...#....", ".....#.#.....", "......#......",
  ".....#.#.....", "....#...#....", "...#.....#...", "............."};

void drawIcon(const char* icon[], uint8_t* frame, uint8_t level) {
  for (int r = 0; r < ROWS; r++)
    for (int c = 0; c < COLS; c++)
      if (icon[r][c] == '#') frame[r * COLS + c] = level;
}

void show_lock(int state, int entered, int progress) {
  uint8_t frame[ROWS * COLS] = {0};
  if (state == 0) drawIcon(ICON_LOCKED, frame, 2);                 // dim lock = idle
  else if (state == 2) drawIcon(ICON_OPEN, frame, 7);
  else if (state == 3 || state == 4) drawIcon(ICON_WRONG, frame, 7);
  if (state == 1) {
    for (int d = 0; d < 3; d++) {                                   // three 3x3 boxes = 3 digits
      int c0 = 1 + d * 4;
      for (int r = 1; r <= 3; r++)
        for (int c = c0; c < c0 + 3; c++) {
          bool edge = (r == 1 || r == 3 || c == c0 || c == c0 + 2);
          frame[r * COLS + c] = d < entered ? 7 : (edge ? 3 : 0);
        }
    }
    int n = progress * COLS / 100;                                   // hold progress bar
    for (int c = 0; c < n && c < COLS; c++) frame[6 * COLS + c] = 5;
  }
  if (state == 4) for (int c = 0; c < COLS; c++) frame[7 * COLS + c] = 2;
  matrix.draw(frame);
}

int lastButton = HIGH;
unsigned long buttonChanged = 0;
bool buttonReported = false;

// ---------------- settings: change these to match your build ----------------
const int  M1A_PIN = 6;               // Maker Drive M1A  (wired to UNO Q pin ~6)
const int  M1B_PIN = 5;               // Maker Drive M1B  (wired to UNO Q pin ~5)
// ALSO NEEDED: a wire from UNO Q GND to the Maker Drive's GND / VB- (shared ground),
// otherwise the driver can't see the pins' signals (the test buttons still work without it).
const bool CLOCKWISE_IS_M1A = false;  // flipped: the lid now opens the other way (M1B direction).
                                      // If it ever opens the wrong way, flip this back
const bool USE_PWM = true;            // false = pins fully HIGH/LOW (full speed, always works);
                                      // true  = analogWrite(SPEED) for a gentler, slower lid
const int  SPEED = 170;               // PWM 0..255, only used when USE_PWM = true
long       OPEN_MS = 150;             // run time for ~90 degrees open (TT motor ~2 rev/s on 4xAA at
                                      // SPEED 170 -> quarter turn ~125-150 ms). Calibrate with latch_cli!
const long CLOSE_SHORT_MS = 8;        // close stops this much before 0, never past it
const long MAX_MS = 400;              // hard upper limit (~well past 90 deg), in case of a bad setting
const long KICK_MS = 40;              // full power for the first 40 ms of every move, then SPEED:
                                      // a slow TT motor often can't START against the lid's friction
const bool BRAKE_HIGH = true;         // stopped = both inputs HIGH = motor terminals shorted = BRAKE,
                                      // holds the lid where it stopped. If the lid still drifts back
                                      // after stopping, try false (both LOW).
// ---------------------------------------------------------------------------

long posMs = 0;          // estimated position (0 = closed)
long targetMs = 0;
int  dir = 0;            // +1 clockwise, -1 anticlockwise, 0 stopped
unsigned long lastT = 0;
unsigned long lastReport = 0;

unsigned long moveStart = 0;
bool kicking = false;

void out(int pin, int value) {
  // one write style per mode (mixing digitalWrite and analogWrite on a pin is unreliable on some cores)
  if (USE_PWM) analogWrite(pin, value);          // 0 = LOW, 255 = fully HIGH
  else         digitalWrite(pin, value > 0 ? HIGH : LOW);
}

void setPins(int d, int power) {
  // d = +1 clockwise, -1 anticlockwise, 0 stopped; power 0..255
  int cw = CLOCKWISE_IS_M1A ? M1A_PIN : M1B_PIN;
  int ccw = CLOCKWISE_IS_M1A ? M1B_PIN : M1A_PIN;
  if (d == 0) {
    int level = BRAKE_HIGH ? 255 : 0;            // brake: hold the lid where it is
    out(cw, level);
    out(ccw, level);
    return;
  }
  out(d > 0 ? ccw : cw, 0);
  out(d > 0 ? cw : ccw, power);
}

void driveRaw(int d) {
  if (d != 0 && d != dir) {                   // starting (or reversing) a move -> kick-start
    moveStart = millis();
    kicking = USE_PWM && SPEED < 255 && KICK_MS > 0;
    setPins(d, kicking ? 255 : SPEED);
  } else if (d == 0) {
    kicking = false;
    setPins(0, 0);
  }
  dir = d;
}

void report() {
  Bridge.notify("latch_state", (int)posMs, dir != 0 ? 1 : 0);
}

void goTo(long t) {
  if (t < 0) t = 0;                     // never anticlockwise past closed
  if (t > MAX_MS) t = MAX_MS;
  targetMs = t;
}

void latch_open()            { goTo(OPEN_MS); }
void latch_close()           { if (posMs > CLOSE_SHORT_MS) goTo(CLOSE_SHORT_MS); }   // close only ever turns anticlockwise
void latch_jog(int ms)       { goTo(targetMs + ms); }
void latch_stop()            { targetMs = posMs; driveRaw(0); report(); }
void latch_home()            { driveRaw(0); posMs = 0; targetMs = 0; report(); }
void latch_set_open(int ms)  { if (ms > 0 && ms <= MAX_MS) OPEN_MS = ms; }

void setup() {
  pinMode(M1A_PIN, OUTPUT);
  pinMode(M1B_PIN, OUTPUT);
  driveRaw(0);
  if (BUTTON_PIN >= 0) pinMode(BUTTON_PIN, INPUT_PULLUP);
  matrix.begin();
  matrix.setGrayscaleBits(3);   // frame values 0..7
  show_lock(0, 0, 0);
  Bridge.begin();
  Bridge.provide("show_lock", show_lock);
  Bridge.provide("latch_open", latch_open);
  Bridge.provide("latch_close", latch_close);
  Bridge.provide("latch_jog", latch_jog);
  Bridge.provide("latch_home", latch_home);
  Bridge.provide("latch_set_open", latch_set_open);
  Bridge.provide("latch_stop", latch_stop);
  lastT = millis();
  Bridge.notify("status", "LunchLock latch ready - assuming the latch is CLOSED now (position 0)");
}

void loop() {
  unsigned long now = millis();
  long dt = (long)(now - lastT);
  lastT = now;

  // advance the position estimate by however long the motor ran since the last loop
  if (dir != 0) posMs += dir * dt;
  if (posMs < 0) posMs = 0;

  // move toward the target; stop as soon as we reach or pass it
  if (targetMs > posMs + 2)       { if (dir != 1) driveRaw(1); }
  else if (targetMs < posMs - 2)  { if (dir != -1) driveRaw(-1); }
  else if (dir != 0)              { driveRaw(0); report(); }

  // kick-start finished -> drop to the gentle SPEED for the rest of the move
  if (kicking && dir != 0 && now - moveStart >= (unsigned long)KICK_MS) {
    kicking = false;
    setPins(dir, SPEED);
  }

  if (now - lastReport > 250) { lastReport = now; report(); }

  // real button (debounced 30 ms) -> tell Python, which starts code entry
  if (BUTTON_PIN >= 0) {
    int b = digitalRead(BUTTON_PIN);
    if (b != lastButton) { lastButton = b; buttonChanged = now; }
    if (b == LOW && now - buttonChanged > 30 && !buttonReported) {
      buttonReported = true;
      Bridge.notify("button");
    }
    if (b == HIGH) buttonReported = false;
  }
}
