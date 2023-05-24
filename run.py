import math

import cv2
import mediapipe as mp
import numpy as np

from src.utils import draw_landmarks

COLOR_RED = (0, 0, 255)
THICKNESS = 2
FONT_SIZE = 1
FONT_COLOR = (255, 255, 255)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
finger_tip_coords = []


def start_camera():
    """Запускает веб-камеру и возвращает объект VideoCapture"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру.")
        exit()
    return cap


def show_frame(cap, finger_tip_coords):
    """Считывает и отображает текущий кадр с веб-камеры"""
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра.")
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Указательный палец
                finger_landmarks = hand_landmarks.landmark[8]
                finger_x = finger_landmarks.x
                finger_y = finger_landmarks.y
        else:
            finger_x, finger_y = None, None

        if finger_x:
            finger_x, finger_y = int(finger_x * image_width), int(finger_y * image_height)
            finger = (finger_x, finger_y)
            cv2.circle(frame, finger, 2, (0, 255, 0), THICKNESS)

            coord_x.append(finger_x)
            coord_y.append(finger_y)

        cv2.imshow('frame', frame)

        return image_height, image_width


def stop_camera(cap):
    """Останавливает веб-камеру и освобождает ресурсы"""
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    coord_x = []
    coord_y = []

    cap = start_camera()
    while True:

        x, y = show_frame(cap, finger_tip_coords)
        if cv2.waitKey(1) == ord('q'):
            break

    stop_camera(cap)

    image = np.ones((x, y, 3), np.uint8) * 255
    for c_x, c_y in zip(coord_x, coord_y):
        cv2.circle(image, (c_x, c_y), 2, (0, 255, 0), THICKNESS)

    for i in range(len(coord_x) - 1):
        cv2.line(image, (coord_x[i], coord_y[i]), (coord_x[i + 1], coord_y[i + 1]), (0, 0, 255), 1)

    cv2.imshow("Points Image", image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    cv2.imwrite("points_image.jpg", image)
