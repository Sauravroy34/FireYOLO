import socket

UDP_IP = "0.0.0.0"      # ← This is correct (listen on all interfaces)
UDP_PORT = 4210

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# This is the most important fix:
sock.bind((UDP_IP, UDP_PORT))

print(f"✅ UDP Server started. Listening on port {UDP_PORT}...")
print("Waiting for data from ESP32...\n")

while True:
    try:
        data, addr = sock.recvfrom(1024)
        message = data.decode('utf-8').strip()
        
        print(f"Received: '{message}' from {addr[0]}:{addr[1]}")
        
    except Exception as e:
        print(f"Error: {e}")