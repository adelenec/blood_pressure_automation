import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

pdata = pd.read_csv("data/050626_Lee3.csv")
pressure_time = (
    pdata["Latest: Time (s)"].astype(float)
).to_numpy()

cap = cv2.VideoCapture("data/050626_Lee3.mp4")
frames = []
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    # stay 2D but keep (N, H, W) stack shape
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps
print("FPS:", fps)
print("frames:", int(frame_count), "duration (s):", duration)

cap.release()
usdata = np.array(frames)
print(usdata.shape)  # (num_frames, height, width)

plt.imshow(usdata[0], cmap="gray")
plt.title("US frame 0")
plt.show()

# mean per frame — left here if you want to eyeball it
# m = np.mean(usdata, axis=(1, 2))
# plt.plot(m)
# plt.show()

# frame-to-frame change (absolute mean), same idea as in the viewer script
avg_change_per_frame = abs(np.mean(np.diff(usdata, axis=0), axis=(1, 2)))
pvals = pdata["Latest: Cuff Pressure (mm Hg)"].to_numpy()

max_p_idx = int(np.argmax(pvals))
t_pressure_max = float(pressure_time[max_p_idx])
peak_change_idx = int(np.argmax(avg_change_per_frame))
t_us_peak = (peak_change_idx + 0.5) / float(fps)

us_t = np.arange(len(avg_change_per_frame), dtype=float) / float(fps)
y_us = float(avg_change_per_frame[peak_change_idx])
y_p = float(pvals[max_p_idx])

plt.subplot(2, 1, 1)
plt.plot(us_t, avg_change_per_frame)
plt.scatter(
    [t_us_peak], [y_us], s=120, zorder=5, marker="*", color="tab:orange",
    edgecolors="k", linewidths=0.5,
    label=f"US: Δ-frame peak → {t_us_peak:.3f}s",
)
plt.axvline(t_us_peak, color="tab:orange", linestyle="--", alpha=0.7)
plt.xlabel("time (s)")
plt.title("mean |Δ| per frame")
plt.legend(loc="upper right", fontsize=8)

plt.subplot(2, 1, 2)
plt.plot(pressure_time, pvals)
plt.scatter(
    [t_pressure_max], [y_p], s=120, zorder=5, marker="*", color="tab:red",
    edgecolors="k", linewidths=0.5,
    label=f"cuff max → {t_pressure_max:.3f}s",
)
plt.axvline(t_pressure_max, color="tab:red", linestyle="--", alpha=0.7)
plt.xlabel("time (s)")
plt.ylabel("mm Hg")
plt.title("cuff pressure")
plt.legend(loc="upper right", fontsize=8)
plt.show()

# align by peaks only (no cross-correlation): each trace keeps its own clock,
# then we subtract the peak times. Video time = frame_idx / fps; pressure uses
# the CSV column × scale above. US peak index is len-1 vs frames because diff().
us_trace = np.mean(usdata, axis=(1, 2))
n = len(us_trace)
us_time = np.arange(n, dtype=float) / float(fps)

print(
    f"video: {us_time[-1]:.3f}s ({n} fr @ {fps:g} Hz); "
    f"pressure span {pressure_time.ptp():.3f}s ({len(pressure_time)} pts); "
    f"t_us_peak={t_us_peak:.3f}, t_p_max={t_pressure_max:.3f}"
)

t_us_rel = us_time - t_us_peak
t_p_rel = pressure_time - t_pressure_max

plt.subplot(2, 1, 1)
plt.plot(t_us_rel, us_trace)
plt.axvline(0, color="tab:orange", linestyle="--", alpha=0.8, label="US t=0")
plt.scatter(
    [0], [float(np.interp(t_us_peak, us_time, us_trace))],
    s=100, zorder=5, marker="*", color="tab:orange", edgecolors="k", linewidths=0.5,
)
plt.xlabel("Δt from US peak (s)")
plt.ylabel("mean gray")
plt.title("ultrasound (frame clock, centered on Δ peak)")
plt.legend(loc="upper right", fontsize=8)

plt.subplot(2, 1, 2)
plt.plot(t_p_rel, pvals)
plt.axvline(0, color="tab:red", linestyle="--", alpha=0.8, label="pressure t=0")
plt.scatter([0], [y_p], s=100, zorder=5, marker="*", color="tab:red", edgecolors="k", linewidths=0.5)
plt.xlabel("Δt from cuff max (s)")
plt.ylabel("mm Hg")
plt.title("pressure (CSV time, centered on max)")
plt.legend(loc="upper right", fontsize=8)
plt.show()

# same x-axis as video: slide pressure so its max lines up with t_us_peak
plt.figure()
p_on_video_t = pressure_time - t_pressure_max + t_us_peak
plt.plot(us_time, us_trace, label="US mean")
plt.plot(p_on_video_t, pvals, label="pressure (shifted)")
plt.scatter(
    [t_us_peak], [float(np.interp(t_us_peak, us_time, us_trace))],
    s=140, zorder=6, marker="*", color="tab:orange", edgecolors="k", linewidths=0.5,
    label="US anchor",
)
plt.scatter(
    [t_us_peak], [y_p], s=140, zorder=6, marker="*", color="tab:red",
    edgecolors="k", linewidths=0.5, label="pressure anchor",
)
plt.axvline(t_us_peak, color="gray", linestyle=":", alpha=0.9)
plt.xlabel("time (s), video frame clock")
plt.legend()
plt.title("overlay — stars should sit on the same vertical line")
plt.show()
