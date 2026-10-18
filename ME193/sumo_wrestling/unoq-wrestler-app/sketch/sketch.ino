// Sumo wrestler -- runs on the UNO Q's microcontroller.
//
// Two jobs, nothing clever (all the deciding happens in python/main.py):
//   1. Stream the sensors to the Linux side 100 times a second:
//        Bridge "sense" -> ax, ay, az (m/s^2), gz (rad/s), lightL, lightR (raw ADC)
//   2. Turn drive(left, right) commands (-100..100 %) into PWM on the Maker Drive
//      that powers the two LEGO motors. Stops on its own if commands go quiet.
//
// Wiring (3.3 V logic -- nothing above 3.3 V on any UNO Q pin):
//   MPU6050 (GY-521): VCC->3V3, GND->GND, SDA->SDA, SCL->SCL
//   LEGO light sensors: signal -> A1 (front-left) and A2 (front-right)
//   Maker Drive: M1A/M1B -> pins 3/5 (left), M2A/M2B -> pins 6/9 (right), GND shared
//
// Libraries (add in App Lab): Adafruit MPU6050, Adafruit Unified Sensor, Adafruit BusIO.

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>

const int LIGHT_L = A1;
const int LIGHT_R = A2;

const int M1A = 3;   // left motor
const int M1B = 5;
const int M2A = 6;   // right motor
const int M2B = 9;

// If a wheel spins the wrong way when told to go forward, flip its flag
// instead of rewiring.
const bool INVERT_LEFT = false;
const bool INVERT_RIGHT = false;

const unsigned long SAMPLE_MS = 10;     // 100 Hz sensor stream
const unsigned long FAILSAFE_MS = 300;  // stop if python/main.py goes quiet this long

const int ROWS = 8;
const int COLS = 13;

Adafruit_MPU6050 mpu;
Arduino_LED_Matrix matrix;
bool found = false;
unsigned long lastSample = 0;
volatile unsigned long last_cmd_ms = 0;
volatile bool moving = false;

// 8 x 13 icons: '#' = LED on, '.' = off.
const char* ICON_READY[ROWS] = {
  ".............",
  ".............",
  ".....###.....",
  "....#####....",
  "....#####....",
  ".....###.....",
  ".............",
  "............."};
const char* ICON_ERROR[ROWS] = {
  "...#.....#...",
  "....#...#....",
  ".....#.#.....",
  "......#......",
  ".....#.#.....",
  "....#...#....",
  "...#.....#...",
  "............."};
const char* ICON_FIGHT[ROWS] = {
  "......#......",
  ".....###.....",
  "....#####....",
  "...#######...",
  "......#......",
  "......#......",
  "......#......",
  "............."};
const char* ICON_EDGE[ROWS] = {
  "......#......",
  "......#......",
  "......#......",
  "...#######...",
  "....#####....",
  ".....###.....",
  "......#......",
  "............."};
const char* ICON_HIT[ROWS] = {
  ".......##....",
  "......##.....",
  ".....##......",
  "....#######..",
  "......##.....",
  ".....##......",
  "....##.......",
  "...#........."};
const char* ICON_MANUAL[ROWS] = {
  ".............",
  "..##.....##..",
  "..###...###..",
  "..##.#.#.##..",
  "..##..#..##..",
  "..##.....##..",
  "..##.....##..",
  "............."};

void drawIcon(const char* icon[]) {
  uint8_t frame[ROWS * COLS];
  for (int r = 0; r < ROWS; r++) {
    for (int c = 0; c < COLS; c++) {
      frame[r * COLS + c] = (icon[r][c] == '#') ? 7 : 0;
    }
  }
  matrix.draw(frame);
}

// Called from python/main.py: 0 ready, 1 fight, 2 edge, 3 hit, 4 manual, other = clear
void show_icon(int id) {
  switch (id) {
    case 0: drawIcon(ICON_READY); break;
    case 1: drawIcon(ICON_FIGHT); break;
    case 2: drawIcon(ICON_EDGE); break;
    case 3: drawIcon(ICON_HIT); break;
    case 4: drawIcon(ICON_MANUAL); break;
    default: matrix.clear(); break;
  }
}

// NOTE: no pinMode() on the motor pins. On the UNO Q, calling pinMode() before
// analogWrite() breaks PWM (analogWrite configures the pin itself).
void set_motor(int pinA, int pinB, int speed, bool invert) {
  if (invert) speed = -speed;
  speed = constrain(speed, -100, 100);
  int duty = abs(speed) * 255 / 100;
  if (speed > 0) {
    analogWrite(pinB, 0);
    analogWrite(pinA, duty);
  } else if (speed < 0) {
    analogWrite(pinA, 0);
    analogWrite(pinB, duty);
  } else {
    analogWrite(pinA, 0);  // low/low = brake
    analogWrite(pinB, 0);
  }
}

void drive(int left, int right) {
  set_motor(M1A, M1B, left, INVERT_LEFT);
  set_motor(M2A, M2B, right, INVERT_RIGHT);
  moving = (left != 0 || right != 0);
  last_cmd_ms = millis();
}

void stop_motors() {
  drive(0, 0);
}

bool start_mpu() {
  if (!mpu.begin()) return false;
  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);  // collisions are sharp
  mpu.setGyroRange(MPU6050_RANGE_1000_DEG);       // spins in place ~ 300-600 deg/s
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);     // smooth out motor vibration
  return true;
}

void setup() {
  stop_motors();  // also configures the four pins as PWM outputs, motors off

  matrix.begin();
  matrix.setGrayscaleBits(3);  // frame values 0..7
  matrix.clear();

  Bridge.begin();
  Bridge.provide("drive", drive);
  Bridge.provide("stop_motors", stop_motors);
  Bridge.provide("show_icon", show_icon);

  Wire.begin();
  Wire.setClock(400000);
  delay(500);
  found = start_mpu();
  drawIcon(found ? ICON_READY : ICON_ERROR);
}

void loop() {
  unsigned long now = millis();
  if (moving && now - last_cmd_ms > FAILSAFE_MS) {
    stop_motors();
  }

  if (!found) {
    // Keep streaming the light sensors (IMU = 0) so the edge guard still
    // works, and keep retrying the MPU6050 once a second.
    static unsigned long lastRetry = 0;
    if (now - lastRetry > 1000) {
      lastRetry = now;
      Bridge.notify("status", "MPU6050 NOT FOUND - check SDA/SCL wiring and that VCC is on 3V3");
      found = start_mpu();
      if (found) {
        drawIcon(ICON_READY);
        Bridge.notify("status", "MPU6050 found");
      }
    }
  }

  if (now - lastSample < SAMPLE_MS) {
    return;
  }
  lastSample = now;

  float ax = 0, ay = 0, az = 0, gz = 0;
  if (found) {
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    ax = a.acceleration.x;
    ay = a.acceleration.y;
    az = a.acceleration.z;
    gz = g.gyro.z;
  }
  Bridge.notify("sense", ax, ay, az, gz, analogRead(LIGHT_L), analogRead(LIGHT_R));
}
