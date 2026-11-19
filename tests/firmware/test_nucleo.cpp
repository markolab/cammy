// clang++ -std=c++17 -Wall -Wextra -Werror -DARDUINO_NUCLEO_G474RE \
//   -Itests/firmware tests/firmware/test_nucleo.cpp -o /tmp/test-nucleo
#include "../../cammy/trigger-nucleo/trigger-nucleo.ino"
#include <iostream>

void send(const std::string &line, const std::string &expected) {
  Serial.output.clear();
  Serial.input = line + "\n";
  while (Serial.available()) loop();
  assert(Serial.output == expected + "\n");
}

void at(uint32_t time) {
  clock_us = time;
  for (int i = 0; i < 3; ++i) loop();  // Permit zero-duration GAP/SETTLING transitions.
}

void assert_off() {
  assert(pins[D2] == HIGH && pins[D3] == LOW && pins[D4] == LOW);
}

int main() {
  setup();
  assert_off();
  assert(dac[PA4] == 0 && dac[PA6] == 0);
  std::string help = Serial.output;
  assert(help.find("115200 baud") != std::string::npos);
  assert(help.find("CONFIG,1,2,200,700,700,8000,5000,1,10000,2000") != std::string::npos);
  assert(help.find("ExposureMode=TriggerWidth") != std::string::npos);
  send("HELP", help.substr(0, help.size() - 1));
  send("", help.substr(0, help.size() - 1));
  send("HELLO", "CAMMY_NUCLEO,1");
  send("START", "ERR COMMAND");

  // Independent widths and underfill: 10 ms surface at 80%, 20 ms fluorescence at 50%.
  send("CONFIG,1,2,200,700,10000,8000,5000,2,10000,2000,20000,3000", "OK CONFIG");
  assert_off();
  send("START", "OK START");
  assert(dac[PA4] == 263 && dac[PA6] == 3760);
  assert(pins[D2] == LOW && pins[D4] == LOW);
  at(199); assert(pins[D4] == LOW);
  at(200); assert(pins[D4] == HIGH);
  at(8199); assert(pins[D2] == LOW && pins[D4] == HIGH);
  at(8200); assert(pins[D2] == HIGH && pins[D4] == HIGH);
  at(10200); assert_off();
  at(12000); assert(pins[D3] == HIGH && pins[D4] == LOW);
  at(12200); assert(pins[D3] == HIGH && pins[D4] == HIGH);
  at(22200); assert(pins[D3] == LOW && pins[D4] == HIGH);
  at(32200); assert_off();
  at(35000); assert(pins[D2] == LOW && pins[D4] == LOW);
  at(35200); assert(pins[D2] == LOW && pins[D4] == HIGH);
  send("HELP", help.substr(0, help.size() - 1)); assert_off();
  send("STOP", "OK STOP");

  // Both single-bank modes: full fill is continuous, partial fill is gated.
  for (int selected = 0; selected < 2; ++selected) {
    for (int fill : {0, 5000, 10000}) {
      uint32_t origin = clock_us;
      send("CONFIG,1," + std::to_string(selected) + ",200,700,700," +
           std::to_string(fill) + "," + std::to_string(fill) + ",1,1000,1000", "OK CONFIG");
      send("START", "OK START");
      at(origin + 200); assert(pins[D4] == HIGH);
      edges.clear();
      at(origin + 700);
      assert((selected == 0 ? pins[D2] == LOW : pins[D3] == HIGH) == (fill == 10000));
      assert(pins[D4] == HIGH);
      at(origin + 1200); assert(pins[D4] == LOW);
      if (fill == 10000) {
        for (auto edge : edges) assert(edge.pin == D4);  // No light-off blip at frame end.
      }
      at(origin + 2000);
      at(origin + 2200); assert(pins[D4] == HIGH);
      send("STOP", "OK STOP"); assert_off();
    }
  }

  // Alternation is independent of timing-list length, including a single width.
  send("CONFIG,1,2,0,700,700,10000,10000,1,1000,1000", "OK CONFIG");
  uint32_t origin = clock_us;
  send("START", "OK START");
  at(origin); assert(pins[D2] == LOW && pins[D4] == HIGH);
  at(origin + 1000); assert_off();
  at(origin + 2000); assert(pins[D3] == HIGH && pins[D4] == HIGH);

  // Bad commands stop outputs; rejected CONFIG cannot be started.
  for (const char *bad : {"1,9,200,700,700,10000,10000,1,1000,1000",
                          "1,2,200,699,700,10000,10000,1,1000,1000",
                          "1,2,200,700,700,10001,10000,1,1000,1000",
                          "1,2,200,700,700,10000,10000,1,1000,100",
                          "1,2,200,700,700,10000,10000,1,0,1000",
                          "1,2,200,700,700,10000,10000,21",
                          "1,2,200,700,700,10000,10000,1,2147483647,1000",
                          "1,2,,700,700,10000,10000,1,1000,1000",
                          "1,2,200,700,700,10000,10000,1,1000,1000,"}) {
    send(std::string("CONFIG,") + bad, "ERR CONFIG"); assert_off();
    send("START", "ERR COMMAND");
  }
  send(std::string(800, 'x'), "ERR LENGTH"); assert_off();
  send("HELLO", "CAMMY_NUCLEO,1");

  // Exposure timing survives uint32_t micros() rollover.
  clock_us = UINT32_MAX - 300;
  send("CONFIG,1,2,200,700,700,8000,5000,1,1000,1000", "OK CONFIG");
  origin = clock_us;
  send("START", "OK START");
  at(origin + 200); assert(pins[D4] == HIGH);
  at(origin + 1000); assert(pins[D2] == HIGH && pins[D4] == HIGH);
  at(origin + 1200); assert_off();
  send("STOP", "OK STOP");
  std::cout << "Nucleo firmware timing and protocol checks passed\n";
}
