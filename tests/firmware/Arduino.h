#pragma once
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

constexpr int D2 = 2, D3 = 3, D4 = 4, PA4 = 14, PA6 = 16;
constexpr int HIGH = 1, LOW = 0, OUTPUT = 1;
struct Edge { uint32_t time; int pin; int value; };
inline uint32_t clock_us = 0;
inline int pins[20] = {}, dac[20] = {};
inline std::vector<Edge> edges;
extern bool surface_active_high;
inline uint32_t micros() { return clock_us; }
inline void pinMode(int, int) {}
inline void digitalWrite(int pin, int value) {
  if (pins[pin] != value) edges.push_back({clock_us, pin, value});
  pins[pin] = value;
  assert(!(pins[D2] == (surface_active_high ? HIGH : LOW) && pins[D3] == HIGH));
}
inline void analogWriteResolution(int bits) { assert(bits == 12); }
inline void analogWrite(int pin, int value) { dac[pin] = value; }
struct FakeSerial {
  std::string input, output;
  void begin(int baud) { assert(baud == 115200); }
  int available() { return input.size(); }
  char read() { char c = input.front(); input.erase(0, 1); return c; }
  void println(const char *line) { output += std::string(line) + "\n"; }
  void flush() {}
};
inline FakeSerial Serial;
