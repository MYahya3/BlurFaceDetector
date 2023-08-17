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


def face_blur_ellipse(image, net, factor=3, detect_threshold=0.90, write_mask=False):
    img = image.copy()
    img_blur = img.copy()
    elliptical_mask = np.zeros(img.shape, dtype=img.dtype)
    # Prepare image and perform inference.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    h, w = img.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detect_threshold:
            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box
            # The face is defined by the bounding rectangle from the detection.
            face = img[int(y1):int(y2), int(x1):int(x2), :]
            # Blur the rectangular area defined by the bounding box.
            face = blur(face, factor=factor)
            # Copy the `blurred_face` to the blurred image.
            img_blur[int(y1):int(y2), int(x1):int(x2), :] = face
            # Specify the elliptical parameters directly from the bounding box coordinates.
            e_center = (x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2)
            e_size = (x2 - x1, y2 - y1)
            e_angle = 0.0
            # Create an elliptical mask.
            elliptical_mask = cv2.ellipse(elliptical_mask, (e_center, e_size, e_angle),
                                          (255, 255, 255), -1, cv2.LINE_AA)
            # Apply the elliptical mask
            np.putmask(img, elliptical_mask, img_blur)

    return img

def FaceDetector(net, image):
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

def main(video_cap, net):
    detection_threshold = 0.9
    blurred_faces = False
    while True:
        has_frame, frame = video_cap.read()
        if not has_frame:
            break
        h = frame.shape[0]
        w = frame.shape[1]

        if not blurred_faces:
            detections = FaceDetector(image=frame, net=net)
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > detection_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype('int')
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            frame = face_blur_ellipse(image=frame, net=net,factor=3)
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
    # Create a network object.
    net = cv2.dnn.readNet('./model/deploy.prototxt',
                                   './model/res10_300x300_ssd_iter_140000.caffemodel', framework="Caffe")

    video_cap = cv2.VideoCapture(0)
    main(video_cap, net)