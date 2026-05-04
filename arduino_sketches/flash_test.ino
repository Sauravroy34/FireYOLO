// ESP32-CAM Flash LED Blink
const int flashLedPin = 4; // Flash LED pin on AI-Thinker

void setup() {
  pinMode(flashLedPin, OUTPUT);
}

void loop() {
  digitalWrite(flashLedPin, HIGH); // Turn LED ON
  delay(500);                      // Wait 500ms
  digitalWrite(flashLedPin, LOW);  // Turn LED OFF
  delay(500);                      // Wait 500ms
}

