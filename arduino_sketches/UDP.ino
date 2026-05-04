#include <WiFi.h>
#include <WiFiUdp.h>
#include <DHT.h>

// WiFi Settings
const char* ssid = "Nothing";
const char* password = "rd123456";

// UDP Settings
WiFiUDP udp;
const char* remote_ip = "10.56.153.172"; // Replace with your computer's IP
const int remote_port = 4210;

// DHT11 Settings
#define DHTPIN 4
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected");
}

void loop() {
  delay(2000); // DHT11 needs at least 1-2 seconds between readings

  float h = dht.readHumidity();
  float t = dht.readTemperature();

  if (isnan(h) || isnan(t)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }

  // Format data string
  String message = "Temp: " + String(t) + "C, Hum: " + String(h) + "%";
  
  // Send UDP packet
  udp.beginPacket(remote_ip, remote_port);
  udp.print(message);
  udp.endPacket();

  Serial.println("Sent: " + message);
}