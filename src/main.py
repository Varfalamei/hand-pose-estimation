import cv2


def start_camera():
    """Запускает веб-камеру и возвращает объект VideoCapture"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру.")
        exit()
    return cap


def show_frame(cap):
    """Считывает и отображает текущий кадр с веб-камеры"""
    ret, frame = cap.read()
    if not ret:
        print("Ошибка чтения кадра.")
        return
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
