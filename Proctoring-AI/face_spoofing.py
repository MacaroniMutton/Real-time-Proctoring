import numpy as np
import cv2
import joblib
from face_detector import get_face_detector, find_faces

def calc_hist(img):
    """
    Calculate histogram of an RGB image.
    """
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        if histr.max() != 0:
            histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

# Load face detector and model
face_model = get_face_detector()
clf = joblib.load('models/face_spoofing.pkl')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

sample_number = 1
count = 0
measures = np.zeros(sample_number, dtype=np.float64)  # Updated from deprecated np.float

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    faces = find_faces(img, face_model)
    measures[count % sample_number] = 0
    height, width = img.shape[:2]

    for x, y, x1, y1 in faces:
        roi = img[y:y1, x:x1]
        if roi.size == 0:
            continue  # Skip empty face regions

        point = (x, y - 5)

        img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

        ycrcb_hist = calc_hist(img_ycrcb)
        luv_hist = calc_hist(img_luv)

        feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel()).reshape(1, -1)

        prediction = clf.predict_proba(feature_vector)
        prob = prediction[0][1]
        measures[count % sample_number] = prob

        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)

        # Determine spoofing status
        if 0 not in measures:
            mean_prob = np.mean(measures)
            text = "False" if mean_prob >= 0.7 else "True"
            color = (0, 0, 255) if text == "False" else (0, 255, 0)

            cv2.putText(img, text, point, cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    count += 1
    cv2.imshow('Face Spoofing Detection', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
