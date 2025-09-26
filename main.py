# face_from_existing_frames_fixed.py
import cv2
import face_recognition
import subprocess
import psutil
import logging
import os

# ---------------- CONFIG ----------------
FRAMES_DIR = r"D:\\For_me\\2Lab\\video_frames"  # папка с кадрами
TARGET_COMMAND = ["notepad.exe"]

TOLERANCE = 0.6
RECOGNITION_REQUIRED_FRAMES = 3
UNKNOWN_REQUIRED_FRAMES = 3
REFERENCE_FRAMES = 20  # сколько первых кадров использовать для эталона
# ----------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

def kill_process_and_children(proc):
    """Безопасное завершение процесса и его дочерних процессов"""
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except:
                pass
        psutil.wait_procs(children, timeout=3)
        try:
            parent.terminate()
            parent.wait(3)
        except:
            try:
                parent.kill()
            except:
                pass
    except:
        pass

def main():
    # Получаем список файлов с кадрами и сортируем по имени
    frame_files = sorted(
        [os.path.join(FRAMES_DIR, f) for f in os.listdir(FRAMES_DIR)
         if f.lower().endswith(('.jpg', '.png'))]
    )
    if not frame_files:
        logging.error("Нет кадров для обработки в папке.")
        return

    reference_encodings = []
    running_proc = None
    known_count = 0
    unknown_count = 0

    logging.info("Начинаем распознавание лиц по кадрам...")

    for i, frame_file in enumerate(frame_files, start=1):
        frame = cv2.imread(frame_file)
        if frame is None:
            logging.warning(f"[Frame {i}] Не удалось загрузить {frame_file}")
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small, model="hog")

        face_encodings = []
        for loc in face_locations:
            enc = face_recognition.face_encodings(rgb_small, known_face_locations=[loc])
            if enc:
                face_encodings.append(enc[0])

        # создаем эталон из первых REFERENCE_FRAMES кадров
        if i <= REFERENCE_FRAMES and face_encodings:
            reference_encodings.extend(face_encodings)
            logging.info(f"[Frame {i}] Добавлен кадр в эталон лица")
            continue

        recognized = False
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(reference_encodings, encoding, tolerance=TOLERANCE)
            if True in matches:
                recognized = True
                break

        if recognized:
            known_count += 1
            unknown_count = 0
            logging.info(f"[Frame {i}] ✅ Лицо распознано")
        else:
            unknown_count += 1
            known_count = 0
            logging.info(f"[Frame {i}] ❌ Лицо НЕ распознано")

        # Запуск процесса при подтвержденном лице
        if known_count >= RECOGNITION_REQUIRED_FRAMES and running_proc is None:
            logging.info(f"[Frame {i}] Лицо подтверждено — запускаю {TARGET_COMMAND}")
            try:
                running_proc = subprocess.Popen(TARGET_COMMAND)
            except Exception as e:
                logging.error(f"Ошибка запуска: {e}")
                running_proc = None

        # Закрытие процесса при неизвестном лице
        if unknown_count >= UNKNOWN_REQUIRED_FRAMES and running_proc:
            logging.warning(f"[Frame {i}] Постороннее лицо — закрываю процесс")
            try:
                kill_process_and_children(running_proc)
            except Exception as e:
                logging.error(f"Ошибка завершения: {e}")
            running_proc = None

    # Завершаем процесс после окончания обработки
    if running_proc:
        kill_process_and_children(running_proc)

    logging.info("Обработка кадров завершена.")

if __name__ == "__main__":
    main()
