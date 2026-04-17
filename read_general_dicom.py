import pydicom
from pydicom.encaps import generate_pixel_data_frame
import numpy as np
import cv2
import io
import matplotlib.pyplot as plt

def load_ultrasound_data(file_path):
    ds = pydicom.dcmread(file_path)
    ts_uid = ds.file_meta.TransferSyntaxUID
    
    # 1. Check for Butterfly Format (MPEG-4 H.264)
    if ts_uid == '1.2.840.10008.1.2.4.102': # Butterfly's H.264 
        print(f"Detected Butterfly (H.264): {file_path}")
        return decode_h264_frames(ds)
    
    # 2. Check for standard formats (Philips)
    try:
        # This works if you have 'pylibjpeg' and 'gdcm' installed
                # access pixel data
        image_data = ds.pixel_array
        print("original pixel_array shape:", getattr(image_data, "shape", None))

        # If image_data is 4D (frames, H, W, C), select the first frame for display
        if isinstance(image_data, np.ndarray) and image_data.ndim == 4:
            print(
                "Detected 4D pixel array (frames,H,W,C). Using first frame index 0 for display."
            )
            image_data = image_data[0]

        # If image_data is 3D but last dim != 3/4, matplotlib may still reject—handle common cases
        if (
            isinstance(image_data, np.ndarray)
            and image_data.ndim == 3
            and image_data.shape[-1] not in (3, 4)
        ):
            # could be (frames, H, W) after mistaken ordering — try taking first slice
            print(
                "3D array with last dim",
                image_data.shape[-1],
                "— interpreting as (frames,H,W). Taking first slice.",
            )
            image_data = image_data[0]

        # visualize image
        plt.imshow(image_data, cmap=plt.cm.gray)
        plt.title("DICOM Image")
        plt.axis("off")
        plt.show()

        return ds.pixel_array
    except NotImplementedError as e:
        print(f"Standard decoding failed for {ts_uid}: {e}")
        return None

def decode_h264_frames(ds):
    """Specific decoder for Butterfly video streams."""
    # Extract the raw video bytes from PixelData
    frame_generator = generate_pixel_data_frame(ds.PixelData)
    video_bytes = next(frame_generator)
    
    # Use OpenCV to read from the memory buffer
    # (Writing to a temp file is often more stable for CV2)
    with open("temp_clip.mp4", "wb") as f:
        f.write(video_bytes)
        
    cap = cv2.VideoCapture("temp_clip.mp4")
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames)

def main():
    # file_path = "/Users/aa/Downloads/butterfly1.dcm" 
    file_path = "/Users/aa/DICOM/IM_0001"
    data = load_ultrasound_data(file_path)
    if data is not None:
        print(f"Loaded data shape: {data.shape}")
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()