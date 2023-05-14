import cv2
import mediapipe as mp

# Задаем константы для рисования на изображении
COLOR_RED = (0, 0, 255)
THICKNESS = 2
FONT_SIZE = 1
FONT_COLOR = (255, 255, 255)

mp_hands = mp.solutions.hands
# Запускаем отрисовщик
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
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра.")
            return

        # Преобразуем изображение из цветовой схемы BGR в RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Определяем позы рук
        results = hands.process(image)

        # Если руки найдены, то выводим ключевые точки на экран
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                finger_tip_coords.append(
                    (int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0]))
                )

        # Рисуем линию между последними двумя точками

        if len(finger_tip_coords) >= 2:
            for finger in finger_tip_coords[:-20]:
                # cv2.с(frame,  finger_tip_coords[i + 1], COLOR_RED, THICKNESS)
                cv2.circle(frame, finger, 2, (0, 255, 0), THICKNESS)

        # Отображаем текущий кадр с веб-камеры
        cv2.imshow('frame', frame)


def stop_camera(cap):
    """Останавливает веб-камеру и освобождает ресурсы"""
    cap.release()
    cv2.destroyAllWindows()


# Запускаем веб-камеру
cap = start_camera()

# Читаем изображение из видеопотока до тех пор, пока пользователь не нажмет клавишу "q"
while True:
    # Отображаем текущий кадр с веб-камеры
    show_frame(cap, finger_tip_coords)

    # Если пользователь нажимает клавишу "q", выходим из цикла
    if cv2.waitKey(1) == ord('q'):
        break

# Останавливаем веб-камеру и освобождаем ресурсы
stop_camera(cap)
