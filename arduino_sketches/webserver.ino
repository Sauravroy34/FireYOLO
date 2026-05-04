#include <WebServer.h>
#include <WiFi.h>
#include <esp32cam.h>

const char* WIFI_SSID = "Nothing";
const char* WIFI_PASS = "rd123456";

WebServer server(80);

// MJPEG Stream Boundaries and Headers
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// This function handles the continuous video stream
void handleStream() {
  WiFiClient client = server.client();
  
  // Send the initial HTTP response with the multipart header
  client.println("HTTP/1.1 200 OK");
  client.printf("Content-Type: %s\r\n", STREAM_CONTENT_TYPE);
  client.println("Access-Control-Allow-Origin: *");
  client.println();

  // Loop continuously to send frames
  while (client.connected()) {
    auto frame = esp32cam::capture();
    if (frame == nullptr) {
      Serial.println("CAPTURE FAILED!");
      break;
    }

    client.print(STREAM_BOUNDARY);
    client.printf(STREAM_PART, frame->size());
    frame->writeTo(client);
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println();
  
  // 1. Connect to WiFi
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  // 2. Initialize Camera
  {
    using namespace esp32cam;
    Config cfg;
    cfg.setPins(pins::AiThinker);
    cfg.setResolution(Resolution::find(800, 600)); // Hardcoded to 800x600 (SVGA)
    cfg.setBufferCount(2);
    cfg.setJpeg(80); // Adjust this between 10-100 if you need to balance quality vs framerate

    bool ok = Camera.begin(cfg);
    if (!ok) {
      Serial.println("Camera initialize failure");
      delay(5000);
      ESP.restart();
    }
    Serial.println("Camera initialize success");
  }

  // 3. Start Server
  // We route the root "/" directly to the stream handler. 
  // No HTML menu will be served.
  server.on("/", handleStream); 
  server.begin();

  Serial.println("=================================");
  Serial.print("Stream Ready! Open this in VLC or Browser: http://");
  Serial.println(WiFi.localIP());
  Serial.println("=================================");
}

void loop() {
  server.handleClient();
}
