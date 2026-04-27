import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

pdata = pd.read_csv("data/042726_flexathigh.csv")
cap = cv2.VideoCapture('data/042726_flexathigh.mp4')
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Convert to grayscale to maintain a 3D shape (Frames, Height, Width)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray_frame)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"Frame count: {frame_count}")
duration = frame_count / fps
print(f"Duration: {duration} seconds")

cap.release()
usdata = np.array(frames)
print(usdata.shape)  # output: (num_frames, height, width))

plt.imshow(usdata[0], cmap='gray')
plt.title('US Data Frame 0')
plt.show()

# mean per frame method
# meanperframe = np.mean(usdata, axis=(1,2))
# plt.plot(meanperframe)
# plt.title('Mean per frame')
# plt.show()

# average change per frame
averagechangeperframe = np.mean(np.diff(usdata, axis=0), axis=(1,2))
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(averagechangeperframe)) / fps, averagechangeperframe)
plt.xlabel('Time (s)')
plt.title('Average change per frame')
plt.subplot(2, 1, 2)
plt.plot(pdata['Data Set 1:Time(s)'], pdata['Data Set 1:Cuff Pressure(mm Hg)'])
max_pressure = pdata['Data Set 1:Cuff Pressure(mm Hg)'].argmax()
max_pressure_time = pdata['Data Set 1:Time(s)'][max_pressure]
plt.plot(pdata['Data Set 1:Time(s)'][max_pressure], pdata['Data Set 1:Cuff Pressure(mm Hg)'][max_pressure], 'ro')
plt.xlabel('Time (s)')
plt.ylabel('Pressure (mm Hg)')
plt.title('Pressure vs Time')
plt.show()

# now align it by time (not sample index)
us_trace = np.mean(usdata, axis=(1, 2))
us_time = np.arange(len(us_trace)) / fps
pressure_time = pdata['Data Set 1:Time(s)'].to_numpy()
pressure = pdata['Data Set 1:Cuff Pressure(mm Hg)'].to_numpy()

# Resample pressure to ultrasound frame timestamps
valid = (us_time >= pressure_time.min()) & (us_time <= pressure_time.max())
us_time_valid = us_time[valid]
us_trace_valid = us_trace[valid]
pressure_on_us = np.interp(us_time_valid, pressure_time, pressure)

# Shift both signals so t=0 is max cuff pressure time
aligned_time = us_time_valid - max_pressure_time

plt.subplot(2, 1, 1)
plt.plot(aligned_time, us_trace_valid)
plt.xlabel('Time from max pressure (s)')
plt.ylabel('Mean grayscale intensity')
plt.title('US Data (aligned)')
plt.subplot(2, 1, 2)
plt.plot(aligned_time, pressure_on_us)
plt.xlabel('Time from max pressure (s)')
plt.ylabel('Pressure (mm Hg)')
plt.title('Pressure (resampled to US timestamps)')
plt.show()