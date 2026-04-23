import pydicom
from pydicom.encaps import generate_pixel_data_frame

ds = pydicom.dcmread("data/butterfly_no_color.dcm")

# Extract the video stream into a file
with open("output_video.mp4", "wb") as f:
    for frame in generate_pixel_data_frame(ds.PixelData):
        f.write(frame)


