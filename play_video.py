#!/usr/bin/env python3
"""Scrub a video with a slider and frame index entry."""
import argparse
import tkinter as tk

import cv2
from PIL import Image, ImageTk


def resize_frame(frame, max_w):
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    nh = int(h * (max_w / w))
    return cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_AREA)


class App:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap.release()
            raise SystemExit(f"Could not open: {path}")
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            self.cap.release()
            raise SystemExit(
                "Could not determine frame count (file unreadable or unsupported)."
            )
        self.max_idx = n - 1
        self._photo = None
        self._slider_programmatic = False

        self.root = tk.Tk()
        self.root.title(path)
        self.idx_var = tk.IntVar(value=0)
        self.entry_var = tk.StringVar(value="0")

        self.img_label = tk.Label(self.root)
        self.img_label.pack(padx=8, pady=8)

        self.slider = tk.Scale(
            self.root,
            from_=0,
            to=self.max_idx,
            orient=tk.HORIZONTAL,
            resolution=1,
            length=900,
            variable=self.idx_var,
            command=self._on_slider,
        )
        self.slider.pack(fill=tk.X, padx=8, pady=4)

        row = tk.Frame(self.root)
        row.pack(fill=tk.X, padx=8, pady=4)
        tk.Label(row, text="Frame:").pack(side=tk.LEFT)
        self.entry = tk.Entry(row, textvariable=self.entry_var, width=12)
        self.entry.pack(side=tk.LEFT, padx=4)
        self.entry.bind("<Return>", self._on_entry)

        self.show_frame(0)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self.cap.release()
        self.root.destroy()

    def _on_slider(self, val):
        if self._slider_programmatic:
            return
        idx = int(float(val))
        idx = max(0, min(idx, self.max_idx))
        self.entry_var.set(str(idx))
        self.show_frame(idx)

    def _on_entry(self, _event=None):
        try:
            idx = int(self.entry_var.get().strip())
        except ValueError:
            return "break"
        idx = max(0, min(idx, self.max_idx))
        self._slider_programmatic = True
        self.slider.set(idx)
        self._slider_programmatic = False
        self.entry_var.set(str(idx))
        self.show_frame(idx)
        return "break"

    def show_frame(self, idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        cv2.putText(
            frame,
            str(idx),
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        disp = resize_frame(frame, 1200)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.img_label.configure(image=self._photo)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    App(p.parse_args().video)


if __name__ == "__main__":
    main()
