import cv2
from cv2 import SimpleBlobDetector, SimpleBlobDetector_Params

# Open the MP4 file
# video_path = "data/butterfly_no_color.mp4"
video_path = "data/phillips_no_color.mp4"
cap = cv2.VideoCapture(video_path)

# Set up SimpleBlobDetector parameters
params = SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 1000
params.maxArea = 500000
params.filterByCircularity = False
params.minCircularity = 0.5
params.filterByConvexity = False
# params.minConvexity = 0.7
params.filterByColor = True
params.blobColor = 0 # Darker is better
# Create detector
detector = cv2.SimpleBlobDetector_create(params)
# print("Blob detector created with parameters:", params)

ret, frame = cap.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY, gray)
# cv2.imshow('thresh ', threshold)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)

cv2.drawContours(frame, contours, -1, (0,255,0), 3)

# get bounding box
x,y,w,h = cv2.boundingRect(big_contour)
print(x,y,w,h)
frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 0, 255),2)
# cv2.imshow('Crop Previsualization', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap.set(0, 0) # reset to first frame
while cap.isOpened():
    ret, frame = cap.read()
 
    # Crop to bounding box
    frame = frame[y:y+h, x:x+w]

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.fastNlMeansDenoising(gray, gray, 30, 7, 21)
    cv2.GaussianBlur(gray, (9, 9), 0, dst=gray)
    _, bw_frame = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # _, bw_frame = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    # bw_frame = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 2)

    # for keypoint in keypoints:
    #     print(f"Detected blob at (x={keypoint.pt[0]:.2f}, y={keypoint.pt[1]:.2f}), size={keypoint.size:.2f}")

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints = detector.detect(bw_frame)
    im_with_keypoints = cv2.drawKeypoints(gray, keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, )

    cv2.imshow('frame', im_with_keypoints)
    if cv2.waitKey(1) == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

# # Read first frame
# ret, frame = cap.read()
# _, bw_frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# if ret:
#     print("Read frame!")
#     # Detect blobs
#     keypoints = detector.detect(bw_frame)
    
#     for keypoint in keypoints:
#         print(f"Detected blob at (x={keypoint.pt[0]:.2f}, y={keypoint.pt[1]:.2f}), size={keypoint.size:.2f}")

#     # Draw detected blobs as circles
#     im_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, 
#                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
#     # Display result
#     cv2.imshow("Detected Ellipses", im_with_keypoints)
#     cv2.imshow("Binary Frame", bw_frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# cap.release()