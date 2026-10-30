// Gait wearable -- runs on the UNO Q's microcontroller.
//
// Reads the MPU6050 (accel + gyro) 50 times a second and sends each reading to
// the Linux side (python/main.py) with Bridge.notify. Python forwards it to the
// laptop over MQTT. The laptop classifies the activity and sends back the two
// models' answers, which are drawn on the LED matrix as two letters:
//
//   left letter  = model 1 (peak acceleration on one axis only)
//   right letter = model 2 (all IMU outputs)
//   S = standing still, W = walking, R = running, B = biking, ? = not sure yet
//
// Wiring (3.3 V logic -- power the sensor from 3V3, NOT 5V):
//   MPU6050 (GY-521): VCC->3V3, GND->GND, SDA->SDA, SCL->SCL (pins next to AREF)
//
// Mounting: strap the board + sensor to the OUTSIDE of the ankle, chip flat
// against the leg, same orientation every time (mark "UP" on the case). The
// models learn the orientation, so changing it between training and testing
// will hurt accuracy.
//
// Libraries (add in App Lab): Adafruit MPU6050, Adafruit Unified Sensor, Adafruit BusIO.

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>

const unsigned long SAMPLE_MS = 20;  // 50 Hz -- plenty for human motion (steps are 1-3 Hz)
const int ROWS = 8;
const int COLS = 13;
const int GLYPH_W = 5;
const int GLYPH_H = 7;

Adafruit_MPU6050 mpu;
Arduino_LED_Matrix matrix;
bool found = false;
unsigned long lastSample = 0;
int zeroReads = 0;  // readings in a row with |a| ~ 0 (impossible -- gravity is always there)

// 5 x 7 letters: '#' = LED on, '.' = off. Index = activity id from the laptop.
//   0 = ?, 1 = S (standing), 2 = W (walking), 3 = R (running), 4 = B (biking)
const char* GLYPHS[5][GLYPH_H] = {
  {".###.", "#...#", "....#", "..##.", "..#..", ".....", "..#.."},  // ?
  {".####", "#....", "#....", ".###.", "....#", "....#", "####."},  // S
  {"#...#", "#...#", "#...#", "#.#.#", "#.#.#", "##.##", "#...#"},  // W
  {"####.", "#...#", "#...#", "####.", "#.#..", "#..#.", "#...#"},  // R
  {"####.", "#...#", "#...#", "####.", "#...#", "#...#", "####."},  // B
};

const char* ICON_READY[ROWS] = {
  ".............",
  ".............",
  ".....###.....",
  "....#####....",
  "....#####....",
  ".....###.....",
  ".............",
  "............."};
const char* ICON_NO_SENSOR[ROWS] = {
  "...#.....#...",
  "....#...#....",
  ".....#.#.....",
  "......#......",
  ".....#.#.....",
  "....#...#....",
  "...#.....#...",
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

void drawGlyph(uint8_t* frame, int id, int col0) {
  if (id < 0 || id > 4) id = 0;
  for (int r = 0; r < GLYPH_H; r++) {
    for (int c = 0; c < GLYPH_W; c++) {
      frame[r * COLS + col0 + c] = (GLYPHS[id][r][c] == '#') ? 7 : 0;
    }
  }
}

// Called from python/main.py with the two models' answers (ids above).
void show_pred(int model1, int model2) {
  uint8_t frame[ROWS * COLS] = {0};
  drawGlyph(frame, model1, 0);
  drawGlyph(frame, model2, COLS - GLYPH_W);
  frame[7 * COLS + 6] = 3;  // dim dot in the middle-bottom = "two answers"
  matrix.draw(frame);
}

void setupSensor() {
  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);  // ankle impacts when running can pass 8 g
  mpu.setGyroRange(MPU6050_RANGE_2000_DEG);       // the shank swings fast when running
  mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);     // keep heel-strike peaks sharp
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);  // frame values 0..7
  matrix.clear();

  Bridge.begin();
  Bridge.provide("show_pred", show_pred);

  Wire.begin();
  Wire.setClock(400000);
  delay(500);
  found = mpu.begin();
  if (found) {
    setupSensor();
    drawIcon(ICON_READY);      // dot in the middle = sensor found, all good
  } else {
    drawIcon(ICON_NO_SENSOR);  // X on the matrix = MPU6050 not found
  }
}

void loop() {
  if (!found) {
    Bridge.notify("status", "MPU6050 NOT FOUND - check SDA/SCL wiring and that VCC is on 3V3");
    delay(1000);
    found = mpu.begin();
    if (found) {
      setupSensor();
      drawIcon(ICON_READY);
      Bridge.notify("status", "MPU6050 found");
    }
    return;
  }
  unsigned long now = millis();
  if (now - lastSample < SAMPLE_MS) {
    return;
  }
  lastSample = now;

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Gravity alone is 9.8 m/s^2, so |a| near 0 means the reads are failing (a wire
  // came loose and the library hands back zeros). After half a second of that,
  // stop sending fake zeros, show the X, and go back to searching for the sensor.
  float amag2 = a.acceleration.x * a.acceleration.x + a.acceleration.y * a.acceleration.y +
                a.acceleration.z * a.acceleration.z;
  if (amag2 < 1.0) {
    if (++zeroReads >= 25) {
      zeroReads = 0;
      found = false;
      drawIcon(ICON_NO_SENSOR);
      Bridge.notify("status", "MPU6050 STOPPED RESPONDING (reads 0 g) - a wire is loose, check SDA/SCL/VCC/GND");
    }
    return;
  }
  zeroReads = 0;
  Bridge.notify("imu",
                a.acceleration.x, a.acceleration.y, a.acceleration.z,  // m/s^2
                g.gyro.x, g.gyro.y, g.gyro.z);                         // rad/s
}
