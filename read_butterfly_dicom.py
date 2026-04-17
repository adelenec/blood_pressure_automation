import pydicom
from pydicom.encaps import generate_pixel_data_frame

ds = pydicom.dcmread("/Users/aa/Downloads/butterfly1.dcm")

# Extract the video stream into a file
with open("output_video.mp4", "wb") as f:
    for frame in generate_pixel_data_frame(ds.PixelData):
        f.write(frame)


