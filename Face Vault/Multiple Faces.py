import cv2
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

face_match = False

reference_dir = "Images"

reference_images = {}

for filename in os.listdir(reference_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        name = os.path.splitext(filename)[0]
        reference_images[name] = cv2.imread(os.path.join(reference_dir, filename))
        print(f"Loaded reference image for {name}")

def check_face(frame):
    global face_match
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print(f"Number of faces detected: {len(faces)}")
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            for name, reference_img in reference_images.items():
                result = cv2.matchTemplate(gray[y:y+h, x:x+w], cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
                if result.max() > 0.7:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    face_match = True
                    break
            if face_match:
                break
        if not face_match:
            print("Face detected, but no match found.")
    else:
        face_match = False
    print("check_face function called.")

while True:
    ret, frame = cap.read()

    if ret:
        check_face(frame)

        # Calculate text position based on frame size
        text_x = int(frame.shape[1] * 0.05)
        text_y = int(frame.shape[0] * 0.9)

        if face_match:
            cv2.putText(frame, "", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()