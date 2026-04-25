import time
import serial
from serial.tools import list_ports

BAUD_RATE = 115200
READ_DELAY_S = 0.1
TOTAL_READINGS = 1000

ports = list(list_ports.comports())
print("Ports found:")
for p in ports:
    print(p.device)

if not ports:
    raise SystemExit("No serial ports found.")

# default to first detected port, but you can change if needed
ser = serial.Serial(ports[0].device, BAUD_RATE, timeout=1)

for x in range(TOTAL_READINGS):
    data = ser.read()
    print(data)
    time.sleep(READ_DELAY_S)

ser.close()