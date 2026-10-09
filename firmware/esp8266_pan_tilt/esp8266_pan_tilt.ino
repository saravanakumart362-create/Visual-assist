// Pan-tilt servo controller for Visual Assist (ESP8266 / NodeMCU).
//
// The Raspberry Pi sends one command per line over USB serial at 115200 baud:
//
//   PAN <0-180>    turn the head left/right (90 = straight ahead)
//   TILT <0-180>   tilt the head up/down    (90 = level)
//   CENTER         both servos to 90
//
// Each command is answered with "OK" or "ERR".
//
// Wiring: pan signal -> D5, tilt signal -> D6. Power the servos from a 5 V
// supply (not the ESP8266's 3.3 V pin) and join all grounds.

#include <Servo.h>

const uint8_t PAN_PIN = D5;
const uint8_t TILT_PIN = D6;

// SG90 pulse range in microseconds.
const int PULSE_MIN = 500;
const int PULSE_MAX = 2500;

Servo pan;
Servo tilt;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);
  pan.attach(PAN_PIN, PULSE_MIN, PULSE_MAX);
  tilt.attach(TILT_PIN, PULSE_MIN, PULSE_MAX);
  pan.write(90);
  tilt.write(90);
}

bool handle(String line) {
  line.trim();
  line.toUpperCase();
  if (line == "CENTER") {
    pan.write(90);
    tilt.write(90);
    return true;
  }
  int space = line.indexOf(' ');
  if (space < 0) {
    return false;
  }
  String name = line.substring(0, space);
  int angle = constrain(line.substring(space + 1).toInt(), 0, 180);
  if (name == "PAN") {
    pan.write(angle);
    return true;
  }
  if (name == "TILT") {
    tilt.write(angle);
    return true;
  }
  return false;
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    Serial.println(handle(line) ? "OK" : "ERR");
  }
}
