#include <WiFi.h>
#include <WiFiUdp.h>
#include <DHT.h>

// WiFi Settings
const char* ssid     = "Nothing";
const char* password = "rd123456";

// UDP Settings
WiFiUDP udp;
const char* remote_ip   = "10.56.153.172";
const int   remote_port = 4210;

// DHT11 Settings
#define DHTPIN   4
#define buzzer   2
#define DHTTYPE  DHT11
DHT dht(DHTPIN, DHTTYPE);

// ── Non-blocking timing ───────────────────────────────────────────────
// DHT11 needs ≥ 2 s between reads; everything else (UDP receive) runs
// every loop iteration so the buzzer reacts with < 200 ms latency.
const unsigned long DHT_INTERVAL_MS = 2000;
unsigned long lastDhtMs = 0;

void setup() {
  Serial.begin(115200);
  pinMode(buzzer, OUTPUT);
  dht.begin();

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  // Open the UDP socket once at startup so we can always receive
  udp.begin(remote_port);
}

void loop() {
  unsigned long now = millis();

  // ── 1. Check for incoming fire-alert from the dashboard (every loop) ─
  // The Python server sends "2" when YOLO sees fire, "0" otherwise.
  // Because we check every loop (not just after a 2 s delay) latency
  // drops from ~2 s to < 200 ms.
  int pktLen = udp.parsePacket();
  if (pktLen > 0) {
    char buf[16] = {0};
    udp.read(buf, sizeof(buf) - 1);
    String response = String(buf);
    response.trim();
    Serial.println("Received: " + response);

    if (response == "2") {
      Serial.println("🔥 Fire confirmed by camera — buzzing!");
      tone(buzzer, 5000, 500);
    }
  }

  // ── 2. Send DHT11 reading every DHT_INTERVAL_MS (non-blocking) ───────
  if (now - lastDhtMs >= DHT_INTERVAL_MS) {
    lastDhtMs = now;

    float h = dht.readHumidity();
    float t = dht.readTemperature();

    if (isnan(h) || isnan(t)) {
      Serial.println("Failed to read from DHT sensor!");
      return;
    }

    // Format and send UDP packet
    String message = "Temp: " + String(t, 2) + "C, Hum: " + String(h, 2) + "%";
    udp.beginPacket(remote_ip, remote_port);
    udp.print(message);
    udp.endPacket();
    Serial.println("Sent: " + message);

    // Local threshold alert (temperature or low humidity)
    if (t > 35.0 && h < 70.0) {
      Serial.println("THRESHOLD ALERT: Temp=" + String(t) + "C  Hum=" + String(h) + "%");
      tone(buzzer, 5000, 800);
    }
  }
}
