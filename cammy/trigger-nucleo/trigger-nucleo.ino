// NUCLEO-G474RE, STM32duino core. See README.md for wiring and protocol.
// When building, copy this file to its own sketch folder (see README.md).
#include <Arduino.h>
#include <stdlib.h>
#include <string.h>

#if !defined(ARDUINO_NUCLEO_G474RE)
#error "Select the NUCLEO-G474RE board (PA4 and PA6 must be DAC outputs)."
#endif

constexpr uint32_t SURFACE_PIN = D2;       // Active low.
constexpr uint32_t FLUORESCENCE_PIN = D3;  // Active high.
constexpr uint32_t CAMERA_PIN = D4;        // High for the entire exposure.
constexpr uint32_t MAX_INTERVAL_US = 0x7fffffff;
constexpr uint32_t MAX_STEPS = 20;
constexpr uint32_t DAC_REFERENCE_MV = 3300;

// Protocol values: mode 0=surface, 1=fluorescence, 2=alternate.
// Illumination is in hundredths of a percent (10000 = full exposure).
uint32_t mode, settle_us, surface_mv, fluorescence_mv;
uint32_t illumination[2], n_steps;
uint32_t exposure_us[MAX_STEPS], low_us[MAX_STEPS];
bool configured = false;
enum Phase { STOPPED, SETTLING, EXPOSING, GAP };
Phase phase = STOPPED;
uint32_t phase_start, step_index, bank, light_duration, gap_duration;
char command[768];
size_t command_length = 0;
bool command_overflow = false;

void lights_off() {
  digitalWrite(SURFACE_PIN, HIGH);
  digitalWrite(FLUORESCENCE_PIN, LOW);
}

void stop_outputs() {
  digitalWrite(CAMERA_PIN, LOW);
  lights_off();
  phase = STOPPED;
}

void prepare_light() {
  light_duration = (uint64_t(exposure_us[step_index]) * illumination[bank] + 5000) / 10000;
  // Switch off the other bank before enabling this one. Rewriting the same
  // levels leaves single-bank 100% illumination on continuously across frames.
  if (bank == 0) {
    digitalWrite(FLUORESCENCE_PIN, LOW);
    digitalWrite(SURFACE_PIN, light_duration == 0 ? HIGH : LOW);
  } else {
    digitalWrite(SURFACE_PIN, HIGH);
    digitalWrite(FLUORESCENCE_PIN, light_duration == 0 ? LOW : HIGH);
  }
  phase_start = micros();
  phase = SETTLING;
}

uint32_t dac_value(uint32_t output_mv) {
  // Desired voltage is AFTER the external 3.3x amplifier, not at the MCU pin.
  return (uint64_t(output_mv) * 4095 * 10 + DAC_REFERENCE_MV * 33 / 2) /
         (DAC_REFERENCE_MV * 33);
}

bool configure(char *input) {
  uint32_t values[8 + 2 * MAX_STEPS];
  size_t count = 0;
  // Strict parsing rejects empty fields, signs, overflow, and trailing junk.
  while (*input) {
    if (count == 8 + 2 * MAX_STEPS || *input < '0' || *input > '9') return false;
    char *end;
    unsigned long value = strtoul(input, &end, 10);
    if (value > MAX_INTERVAL_US || (*end != ',' && *end != '\0')) return false;
    values[count++] = value;
    if (*end == '\0') break;
    input = end + 1;
    if (*input == '\0') return false;
  }
  if (count < 8 || values[0] != 1 || values[1] > 2 ||
      values[3] < 700 || values[3] > 10000 ||
      values[4] < 700 || values[4] > 10000 ||
      values[5] > 10000 || values[6] > 10000 ||
      values[7] < 1 || values[7] > MAX_STEPS || count != 8 + 2 * values[7]) return false;
  for (size_t i = 0; i < values[7]; ++i) {
    uint32_t width = values[8 + 2 * i], low = values[9 + 2 * i];
    if (width == 0 || low == 0 || low < values[2] || width > MAX_INTERVAL_US - low) return false;
  }
  mode = values[1];
  settle_us = values[2];
  surface_mv = values[3];
  fluorescence_mv = values[4];
  illumination[0] = values[5];
  illumination[1] = values[6];
  n_steps = values[7];
  for (size_t i = 0; i < n_steps; ++i) {
    exposure_us[i] = values[8 + 2 * i];
    low_us[i] = values[9 + 2 * i];
  }
  configured = true;
  return true;
}

void print_help() {
  // Prefix instructions so serial clients can distinguish them from replies.
  Serial.println("# Cammy NUCLEO-G474RE light/camera controller");
  Serial.println("# Serial monitor: 115200 baud, Newline or Both NL & CR, one command per line.");
  Serial.println("# HELP (or an empty line): stop outputs and print these instructions.");
  Serial.println("# HELLO: stop outputs and report firmware identity. STOP: all TTL outputs off.");
  Serial.println("# CONFIG: configure while stopped. START: begin the configured sequence.");
  Serial.println("# D2 surface LOW=on; D3 fluorescence HIGH=on; D4 HIGH=exposing.");
  Serial.println("# PA4 surface and PA6 fluorescence: 12-bit DAC brightness.");
  Serial.println("# Configuration format (integers, no spaces):");
  Serial.println("# CONFIG,1,mode,settle_us,surface_mV,fluorescence_mV,surface_fill,fluorescence_fill,n,exposure_us,low_us,...");
  Serial.println("# mode: 0=surface, 1=fluorescence, 2=alternate (surface first).");
  Serial.println("# Brightness: 700..10000 mV AFTER the 3.3x amplifier.");
  Serial.println("# Fill: 0..10000; 8000=80% illuminated, 10000=full exposure.");
  Serial.println("# Single-bank mode at 10000 fill keeps that bank on continuously.");
  Serial.println("# n: 1..20 exposure_us,low_us pairs. Low time includes settling time.");
  Serial.println("# Exposure and low must be positive; low >= settle; exposure+low <= 2147483647 us.");
  Serial.println("# Examples: 10 ms exposures, 2 ms low, 200 us settling, 0.7 V brightness.");
  Serial.println("# Surface only, continuous light:");
  Serial.println("# CONFIG,1,0,200,700,700,10000,10000,1,10000,2000");
  Serial.println("# Fluorescence only, continuous light:");
  Serial.println("# CONFIG,1,1,200,700,700,10000,10000,1,10000,2000");
  Serial.println("# Alternating: surface illuminated for 8 ms, fluorescence for 5 ms:");
  Serial.println("# CONFIG,1,2,200,700,700,8000,5000,1,10000,2000");
  Serial.println("# Paste a CONFIG line WITHOUT the '# ' prefix; wait for OK CONFIG, then send START.");
  Serial.println("# D4 stays high for the full exposure; underfill only shortens illumination.");
  Serial.println("# Camera: FrameStart, hardware input, RisingEdge, ExposureMode=TriggerWidth.");
  Serial.println("# Send STOP to finish. ERR means outputs stopped; send HELP to review syntax.");
}

void handle_command() {
  if (strcmp(command, "HELP") == 0 || command[0] == '\0') {
    stop_outputs();
    print_help();
  } else if (strcmp(command, "STOP") == 0 || strcmp(command, "HELLO") == 0) {
    stop_outputs();
    Serial.println(strcmp(command, "STOP") == 0 ? "OK STOP" : "CAMMY_NUCLEO,1");
  } else if (strncmp(command, "CONFIG,", 7) == 0) {
    stop_outputs();
    configured = false;
    Serial.println(configure(command + 7) ? "OK CONFIG" : "ERR CONFIG");
  } else if (strcmp(command, "START") == 0 && configured && phase == STOPPED) {
    analogWrite(PA4, dac_value(surface_mv));
    analogWrite(PA6, dac_value(fluorescence_mv));
    step_index = 0;
    bank = mode == 1 ? 1 : 0;  // Alternation always starts with surface.
    Serial.println("OK START");
    Serial.flush();           // Finish the acknowledgement before timing begins.
    prepare_light();
  } else {
    stop_outputs();
    Serial.println("ERR COMMAND");
  }
}

void read_command() {
  // One byte per loop keeps receipt of a command from delaying timing edges.
  if (!Serial.available()) return;
  char c = Serial.read();
  if (c == '\r') return;
  if (c == '\n') {
    command[command_length] = '\0';
    if (command_overflow) {
      stop_outputs();
      configured = false;
      Serial.println("ERR LENGTH");
    } else {
      handle_command();
    }
    command_length = 0;
    command_overflow = false;
  } else if (command_length < sizeof(command) - 1) {
    command[command_length++] = c;
  } else {
    command_overflow = true;
  }
}

void setup() {
  // Preload the active-low surface output before enabling its driver.
  digitalWrite(SURFACE_PIN, HIGH);
  digitalWrite(FLUORESCENCE_PIN, LOW);
  digitalWrite(CAMERA_PIN, LOW);
  pinMode(SURFACE_PIN, OUTPUT);
  pinMode(FLUORESCENCE_PIN, OUTPUT);
  pinMode(CAMERA_PIN, OUTPUT);
  stop_outputs();
  analogWriteResolution(12);
  analogWrite(PA4, 0);
  analogWrite(PA6, 0);
  Serial.begin(115200);  // ST-Link virtual COM port; no waiting for USB enumeration.
  print_help();
}

void loop() {
  uint32_t elapsed = uint32_t(micros() - phase_start);  // Handles micros() rollover.
  if (phase == SETTLING && elapsed >= settle_us) {
    if (light_duration == 0) lights_off();
    digitalWrite(CAMERA_PIN, HIGH);
    phase_start = micros();
    phase = EXPOSING;
  } else if (phase == EXPOSING) {
    if (elapsed >= light_duration && (mode == 2 || illumination[bank] < 10000)) lights_off();
    if (elapsed >= exposure_us[step_index]) {
      digitalWrite(CAMERA_PIN, LOW);
      phase_start = micros();
      gap_duration = low_us[step_index] - settle_us;
      step_index = (step_index + 1) % n_steps;
      if (mode == 2) bank = 1 - bank;
      phase = GAP;
    }
  } else if (phase == GAP && elapsed >= gap_duration) {
    prepare_light();
  }
  read_command();
}
