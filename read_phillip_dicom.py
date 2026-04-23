import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2

# load

file_path = "data/phillips_no_color.dcm"
# file_path = "/Users/aa/Downloads/butterfly1.dcm" # doesn't work

ds = pydicom.dcmread(file_path)

# access metadata
print("Patient Name:", ds.PatientName)
print("Patient ID:", ds.PatientID)
print("Study Date:", ds.StudyDate)
print("Modality:", ds.Modality)

# access pixel data
image_data = ds.pixel_array
print("original pixel_array shape:", getattr(image_data, "shape", None))

height, width = image_data[0].shape[:2]
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))

for frame in image_data:
    # print(frame.shape)
    out.write(frame)
    cv2.imshow("Frame", frame)

out.release()


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
cv2.imshow("DICOM Image", image_data)
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.imshow(image_data, cmap=plt.cm.gray)
# plt.title("DICOM Image")
# plt.axis("off")
# plt.show()
