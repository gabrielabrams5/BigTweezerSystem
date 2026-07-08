// 3D-Tweezer firmware, thin-shell version.
//
// The Python side (classes/field_synth.py) owns the coil geometry and does
// all field/gradient/roll synthesis, then sends the six per-coil currents
// directly over serial. The Arduino is now a dumb PWM writer plus the
// AD9850 acoustic-DDS passthrough.
//
// Packet layout (7 floats, order MUST match Python arduino_class.send):
//   [0..5] = I1, I2, I3, I4, I5, I6   -- per-coil signed PWM duty in [-1, 1]
//   [6]    = acoustic_freq (Hz; 0 disables the DDS)
//
// Coil pinout (unchanged from prior versions):
//   C1: PWMR=D2,  PWML=D3,  ENR=D26, ENL=D27
//   C2: PWMR=D44, PWML=D5,  ENR=D24, ENL=D25
//   C3: PWMR=D6,  PWML=D7,  ENR=D22, ENL=D23
//   C4: PWMR=D8,  PWML=D9,  ENR=D32, ENL=D33
//   C5: PWMR=D10, PWML=D11, ENR=D30, ENL=D31
//   C6: PWMR=D12, PWML=D46, ENR=D28, ENL=D29
//
// AD9850 DDS: W_CLK=D34, FQ_UD=D36, DATA=D38, RESET=D40

#include <AD9850.h>
#include "SerialTransfer.h"

SerialTransfer myTransfer;

// 7-float packet buffer. Global so state persists between rxObj calls; loop
// keeps applying the most recent packet's values until a new one arrives.
float action[7];

// Coil 1 : top ring, azimuth 0
const int Coil1_PWMR = 2;
const int Coil1_PWML = 3;
const int Coil1_ENR = 26;
const int Coil1_ENL = 27;

// Coil 2 : top ring, azimuth 120
const int Coil2_PWMR = 44;
const int Coil2_PWML = 5;
const int Coil2_ENR = 24;
const int Coil2_ENL = 25;

// Coil 3 : top ring, azimuth 240
const int Coil3_PWMR = 6;
const int Coil3_PWML = 7;
const int Coil3_ENR = 22;
const int Coil3_ENL = 23;

// Coil 4 : bottom ring, azimuth 0
const int Coil4_PWMR = 8;
const int Coil4_PWML = 9;
const int Coil4_ENR = 32;
const int Coil4_ENL = 33;

// Coil 5 : bottom ring, azimuth 120
const int Coil5_PWMR = 10;
const int Coil5_PWML = 11;
const int Coil5_ENR = 30;
const int Coil5_ENL = 31;

// Coil 6 : bottom ring, azimuth 240
const int Coil6_PWMR = 12;
const int Coil6_PWML = 46;
const int Coil6_ENR = 28;
const int Coil6_ENL = 29;

// AD9850 acoustic DDS
const int W_CLK_PIN  = 34;
const int FQ_UD_PIN  = 36;
const int DATA_PIN   = 38;
const int RESET_PIN  = 40;

int phase = 0;

void setup() {
  cli();
  // Fast PWM for the coil-driver pins. 0x01 -> ~31 kHz on Timers 1-5.
  TCCR1B = (TCCR1B & 0b11111000) | 0x01;
  TCCR2B = (TCCR2B & 0b11111000) | 0x01;
  TCCR3B = (TCCR3B & 0b11111000) | 0x01;
  TCCR4B = (TCCR4B & 0b11111000) | 0x01;
  TCCR5B = (TCCR5B & 0b11111000) | 0x01;
  sei();

  Serial.begin(115200);
  myTransfer.begin(Serial);

  DDS.begin(W_CLK_PIN, FQ_UD_PIN, DATA_PIN, RESET_PIN);
  DDS.calibrate(124999500);

  // Coil driver pin modes
  pinMode(Coil1_PWMR, OUTPUT); pinMode(Coil1_PWML, OUTPUT);
  pinMode(Coil1_ENR,  OUTPUT); pinMode(Coil1_ENL,  OUTPUT);
  pinMode(Coil2_PWMR, OUTPUT); pinMode(Coil2_PWML, OUTPUT);
  pinMode(Coil2_ENR,  OUTPUT); pinMode(Coil2_ENL,  OUTPUT);
  pinMode(Coil3_PWMR, OUTPUT); pinMode(Coil3_PWML, OUTPUT);
  pinMode(Coil3_ENR,  OUTPUT); pinMode(Coil3_ENL,  OUTPUT);
  pinMode(Coil4_PWMR, OUTPUT); pinMode(Coil4_PWML, OUTPUT);
  pinMode(Coil4_ENR,  OUTPUT); pinMode(Coil4_ENL,  OUTPUT);
  pinMode(Coil5_PWMR, OUTPUT); pinMode(Coil5_PWML, OUTPUT);
  pinMode(Coil5_ENR,  OUTPUT); pinMode(Coil5_ENL,  OUTPUT);
  pinMode(Coil6_PWMR, OUTPUT); pinMode(Coil6_PWML, OUTPUT);
  pinMode(Coil6_ENR,  OUTPUT); pinMode(Coil6_ENL,  OUTPUT);
}


// H-bridge writers. DC in [-1, 1]. Positive drives PWMR, negative drives PWML.
// The signed convention matches how field_synth returns per-coil currents:
// positive = axis-aligned field contribution, negative = reversed. If a
// specific coil is wound backward relative to that convention, invert it in
// calibration.json (negative gain on that coil) rather than editing this file.

void set1(float DC1) {
  digitalWrite(Coil1_ENR, HIGH); digitalWrite(Coil1_ENL, HIGH);
  if (DC1 > 0) { analogWrite(Coil1_PWMR, abs(DC1) * 255); analogWrite(Coil1_PWML, 0); }
  else if (DC1 < 0) { analogWrite(Coil1_PWMR, 0); analogWrite(Coil1_PWML, abs(DC1) * 255); }
  else { analogWrite(Coil1_PWMR, 0); analogWrite(Coil1_PWML, 0); }
}
void set2(float DC2) {
  digitalWrite(Coil2_ENR, HIGH); digitalWrite(Coil2_ENL, HIGH);
  if (DC2 > 0) { analogWrite(Coil2_PWMR, abs(DC2) * 255); analogWrite(Coil2_PWML, 0); }
  else if (DC2 < 0) { analogWrite(Coil2_PWMR, 0); analogWrite(Coil2_PWML, abs(DC2) * 255); }
  else { analogWrite(Coil2_PWMR, 0); analogWrite(Coil2_PWML, 0); }
}
void set3(float DC3) {
  digitalWrite(Coil3_ENR, HIGH); digitalWrite(Coil3_ENL, HIGH);
  if (DC3 > 0) { analogWrite(Coil3_PWMR, abs(DC3) * 255); analogWrite(Coil3_PWML, 0); }
  else if (DC3 < 0) { analogWrite(Coil3_PWMR, 0); analogWrite(Coil3_PWML, abs(DC3) * 255); }
  else { analogWrite(Coil3_PWMR, 0); analogWrite(Coil3_PWML, 0); }
}
void set4(float DC4) {
  digitalWrite(Coil4_ENR, HIGH); digitalWrite(Coil4_ENL, HIGH);
  if (DC4 > 0) { analogWrite(Coil4_PWMR, abs(DC4) * 255); analogWrite(Coil4_PWML, 0); }
  else if (DC4 < 0) { analogWrite(Coil4_PWMR, 0); analogWrite(Coil4_PWML, abs(DC4) * 255); }
  else { analogWrite(Coil4_PWMR, 0); analogWrite(Coil4_PWML, 0); }
}
void set5(float DC5) {
  digitalWrite(Coil5_ENR, HIGH); digitalWrite(Coil5_ENL, HIGH);
  if (DC5 > 0) { analogWrite(Coil5_PWMR, abs(DC5) * 255); analogWrite(Coil5_PWML, 0); }
  else if (DC5 < 0) { analogWrite(Coil5_PWMR, 0); analogWrite(Coil5_PWML, abs(DC5) * 255); }
  else { analogWrite(Coil5_PWMR, 0); analogWrite(Coil5_PWML, 0); }
}
void set6(float DC6) {
  digitalWrite(Coil6_ENR, HIGH); digitalWrite(Coil6_ENL, HIGH);
  if (DC6 > 0) { analogWrite(Coil6_PWMR, abs(DC6) * 255); analogWrite(Coil6_PWML, 0); }
  else if (DC6 < 0) { analogWrite(Coil6_PWMR, 0); analogWrite(Coil6_PWML, abs(DC6) * 255); }
  else { analogWrite(Coil6_PWMR, 0); analogWrite(Coil6_PWML, 0); }
}


void loop() {
  if (myTransfer.available()) {
    uint16_t message = 0;
    myTransfer.rxObj(action, message);
  }

  // Acoustic frequency passthrough. 0 -> stop the DDS output.
  float acoustic_freq = action[6];
  if (acoustic_freq != 0.0f) {
    DDS.setfreq(acoustic_freq, phase);
  } else {
    DDS.down();
  }

  // Coil drives -- straight from packet, no field math.
  set1(action[0]);
  set2(action[1]);
  set3(action[2]);
  set4(action[3]);
  set5(action[4]);
  set6(action[5]);
}
