// Ping-pong paddle -- runs on the UNO Q's microcontroller.
//
// Reads the MPU6050 (accel + gyro) and the light sensor 100 times a second and
// sends each reading to the Linux side (python/main.py) with Bridge.notify.
// Python forwards it to the laptop over MQTT, and sends back icons to show on
// the LED matrix (hit / miss / smash / point / win / lose).
//
// Wiring (3.3 V logic -- power the sensors from 3V3, NOT 5V):
//   MPU6050 (GY-521): VCC->3V3, GND->GND, SDA->SDA, SCL->SCL (pins next to AREF)
//   Light sensor:     3V3 -> photoresistor -> A0 -> 10k -> GND
//
// Libraries (add in App Lab): Adafruit MPU6050, Adafruit Unified Sensor, Adafruit BusIO.

#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>

const unsigned long SAMPLE_MS = 20;  // 50 Hz (try 10 = 100 Hz once everything works)
const int ROWS = 8;
const int COLS = 13;

Adafruit_MPU6050 mpu;
Arduino_LED_Matrix matrix;
bool found = false;
unsigned long lastSample = 0;

// 8 x 13 icons: '#' = LED on, '.' = off.
const char* ICON_HIT[ROWS] = {
  ".............",
  "...........#.",
  "..........##.",
  ".........##..",
  ".##.....##...",
  "..##...##....",
  "...##.##.....",
  "....###......"};
const char* ICON_MISS[ROWS] = {
  "...#.....#...",
  "....#...#....",
  ".....#.#.....",
  "......#......",
  ".....#.#.....",
  "....#...#....",
  "...#.....#...",
  "............."};
const char* ICON_SMASH[ROWS] = {
  ".......##....",
  "......##.....",
  ".....##......",
  "....#######..",
  "......##.....",
  ".....##......",
  "....##.......",
  "...#........."};
const char* ICON_POINT[ROWS] = {
  "......#......",
  "......#......",
  "....#####....",
  "..#########..",
  "....#####....",
  "...##...##...",
  "..##.....##..",
  "............."};
const char* ICON_WIN[ROWS] = {
  "...#######...",
  "..#.#####.#..",
  "..#.#####.#..",
  "...#######...",
  ".....###.....",
  "......#......",
  "....#####....",
  "............."};
const char* ICON_LOSE[ROWS] = {
  "....#####....",
  "...#.....#...",
  "..#.#...#.#..",
  "..#.......#..",
  "..#..###..#..",
  "..#.#...#.#..",
  "...#.....#...",
  "....#####...."};
const char* ICON_READY[ROWS] = {
  ".............",
  ".............",
  ".....###.....",
  "....#####....",
  "....#####....",
  ".....###.....",
  ".............",
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

// Called from python/main.py: 0 clear, 1 hit, 2 miss, 3 smash, 4 point, 5 win, 6 lose, 7 ready
void show_icon(int id) {
  switch (id) {
    case 1: drawIcon(ICON_HIT); break;
    case 2: drawIcon(ICON_MISS); break;
    case 3: drawIcon(ICON_SMASH); break;
    case 4: drawIcon(ICON_POINT); break;
    case 5: drawIcon(ICON_WIN); break;
    case 6: drawIcon(ICON_LOSE); break;
    case 7: drawIcon(ICON_READY); break;
    default: matrix.clear(); break;
  }
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);  // frame values 0..7
  matrix.clear();

  Bridge.begin();
  Bridge.provide("show_icon", show_icon);

  Wire.begin();
  Wire.setClock(400000);  // fast I2C so a full read fits easily in 10 ms
  delay(500);
  found = mpu.begin();
  if (found) {
    mpu.setAccelerometerRange(MPU6050_RANGE_16_G);  // swings are violent
    mpu.setGyroRange(MPU6050_RANGE_2000_DEG);       // ...and fast (2000 deg/s = 34.9 rad/s)
    mpu.setFilterBandwidth(MPU6050_BAND_94_HZ);     // keep the swing peaks sharp
    drawIcon(ICON_READY);  // dot in the middle = sensor found, all good
  } else {
    drawIcon(ICON_MISS);   // X on the matrix = MPU6050 not found
  }
}

void loop() {
  if (!found) {
    // Keep sending the light sensor (gyro/accel = 0) so you can still see the
    // sketch -> Python link working, and keep retrying the MPU6050.
    Bridge.notify("status", "MPU6050 NOT FOUND - check SDA/SCL wiring and that VCC is on 3V3");
    Bridge.notify("imu", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, analogRead(A0));
    delay(1000);
    found = mpu.begin();
    if (found) {
      mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
      mpu.setGyroRange(MPU6050_RANGE_2000_DEG);
      mpu.setFilterBandwidth(MPU6050_BAND_94_HZ);
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
  int light = analogRead(A0);
  Bridge.notify("imu",
                a.acceleration.x, a.acceleration.y, a.acceleration.z,  // m/s^2
                g.gyro.x, g.gyro.y, g.gyro.z,                          // rad/s
                light);
}
