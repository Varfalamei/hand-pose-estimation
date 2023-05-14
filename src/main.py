import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

# Запускаем отрисовщик
mp_drawing = mp.solutions.drawing_utils


def start_camera():
    """Запускает веб-камеру и возвращает объект VideoCapture"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру.")
        exit()
    return cap


def show_frame(cap):
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
    show_frame(cap)

    # Если пользователь нажимает клавишу "q", выходим из цикла
    if cv2.waitKey(1) == ord('q'):
        break

# Останавливаем веб-камеру и освобождаем ресурсы
stop_camera(cap)
