import cv2
import importlib
import numpy as np
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
smp = importlib.import_module("segmentation_models_pytorch")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(device).eval()

cap = cv2.VideoCapture("data/phillips_no_color.mp4")
while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()

    mask = (p > 0.5).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = frame.copy()
    for c in cnts:
        if len(c) >= 5 and cv2.contourArea(c) > 100:
            cv2.ellipse(out, cv2.fitEllipse(c), (0, 255, 0), 2)

    cv2.imshow("smp mask", mask)
    cv2.imshow("smp ellipses", out)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()