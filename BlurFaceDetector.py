import cv2
import numpy as np

def blur(face, factor=3):
    h, w = face.shape[:2]
    if factor < 1: factor = 1  # Maximum blurring
    if factor > 5: factor = 5  # Minimal blurring
    # Kernel size.
    w_k = int(w / factor)
    h_k = int(h / factor)
    # Insure kernel is an odd number.
    if w_k % 2 == 0: w_k += 1
    if h_k % 2 == 0: h_k += 1
    blurred = cv2.GaussianBlur(face, (int(w_k), int(h_k)), 0, 0)
    return blurred
def FaceDetector(image):
    # Create a network object.
    net = cv2.dnn.readNet('./model/deploy.prototxt',
                                   './model/res10_300x300_ssd_iter_140000.caffemodel', framework="Caffe")
    # Model parameters used to train model.
    mean = [104, 117, 123]
    scale = 1.0
    in_width = 300
    in_height = 300
    # Convert the image into a blob format.
    blob = cv2.dnn.blobFromImage(image, scalefactor=scale, size=(in_width, in_height), mean=mean, swapRB=False, crop=False)
    # Pass the blob to the DNN model.
    net.setInput(blob)
    # Retrieve detections from the DNN model.
    detections = net.forward()
    return detections

def main(video_cap):
    detection_threshold = 0.9

    blurred_faces = False

    while True:
        has_frame, frame = video_cap.read()
        if not has_frame:
            break
        h = frame.shape[0]
        w = frame.shape[1]

        detections = FaceDetector(image=frame)
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > detection_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype('int')

                if blurred_faces:
                    # Extract the face ROI.
                    face = frame[y1:y2, x1:x2]
                    face = blur(face, factor=3)
                    # Replace the detected face with the blurred one.
                    frame[y1:y2, x1:x2] = face
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Press P for Face Detction and B for Face Blurring", frame)
        key = cv2.waitKey(1)

        if key == ord('Q') or key == ord('q') or key == 27:
            break
        elif key == ord('p'):
            blurred_faces = False
        elif key == ord('b'):
            blurred_faces = True

    video_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_cap = cv2.VideoCapture("input-video.mp4")
    main(video_cap)