import cv2
import mediapipe as mp

# Инициализация детектора ключевых точек руки
mp_hands = mp.solutions.hands

# Инициализация отображения ключевых точек руки
mp_drawing = mp.solutions.drawing_utils

# Загрузка видеофайла
cap = cv2.VideoCapture('path/to/video')

# Запуск цикла обработки кадров
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        # Чтение кадра из видеофайла
        success, image = cap.read()
        if not success:
            break

        # Конвертация изображения в RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Обработка изображения детектором ключевых точек руки
        results = hands.process(image)

        # Отображение ключевых точек на изображении
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Отображение обработанного кадра
        cv2.imshow('Hand Tracking', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
