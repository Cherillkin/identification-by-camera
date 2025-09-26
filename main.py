# face_calc_guard.py
import os
import sys
import time
import threading
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from typing import Tuple, List, Optional

# ================== НАСТРОЙКИ РАСПОЗНАВАНИЯ ==================
DATA_AUTH_DIR  = "video_frames"

MODEL_PATH = "face_model.yml"
AUTHORIZED_CLASS_LABEL = 1
OTHER_CLASS_LABEL = 0

CHECK_INTERVAL_SEC = 10          # период плановой проверки, сек

# Окно перепроверки (устойчивость)
RETRY_WINDOW_SEC = 3.0           # длительность окна, сек
RETRY_FPS = 10                   # кадров/сек в окне
REQUIRED_OK_IN_WINDOW = 6        # нужно столько «успешных» кадров...
WINDOW_SIZE = 10                 # из максимум стольких попыток

# Порог LBPH
LBPH_BASE_THRESHOLD = 80.0       # запасной порог, если калибровка не сработает
CALIBRATION_MARGIN = 8.0         # итоговый порог = медиана(authorized) + этот запас
RUNTIME_THRESHOLD_FILE = "runtime_threshold.txt"

# Извлечение лица
FACE_SIZE = (200, 200)
FACE_PAD_RATIO = 0.18

# Камера
WARMUP_SEC = 1.2
BLACK_FRAME_MEAN = 8.0
HAAR_NAME = "haarcascade_frontalface_default.xml"

# Калибровка при старте
CALIBRATE_ON_START = True
CALIBRATION_SECONDS = 3.0

# ================== НАСТРОЙКИ КАЛЬКУЛЯТОРА ==================
WIN_TITLE = "Калькулятор (доступ только для владельца)"
WIN_SIZE = "330x500"
# ============================================================

def _log(msg: str):
    print(f"[FaceGuard] {msg}", flush=True)

# ---------- предобработка/детекция ----------
def preprocess_gray(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray

def pad_rect(x: int, y: int, w: int, h: int, pad: int, W: int, H: int) -> Tuple[int,int,int,int]:
    x2, y2 = x + w, y + h
    x = max(0, int(x - pad))
    y = max(0, int(y - pad))
    x2 = min(W, int(x2 + pad))
    y2 = min(H, int(y2 + pad))
    return x, y, x2 - x, y2 - y

def detect_largest_face(cascade: cv2.CascadeClassifier, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    gray = preprocess_gray(frame_bgr)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    return np.array(faces[0])

def crop_face_region(frame_bgr: np.ndarray, face_rect: np.ndarray) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    x, y, w, h = map(int, face_rect.tolist())
    pad = int(max(w, h) * FACE_PAD_RATIO)
    x, y, w, h = pad_rect(x, y, w, h, pad, W, H)
    face = frame_bgr[y:y+h, x:x+w]
    gray = preprocess_gray(face)
    gray = cv2.resize(gray, FACE_SIZE)
    return gray

# ---------- данные и модель ----------
def list_folder_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    names = sorted(os.listdir(folder))
    return [os.path.join(folder, n) for n in names if os.path.isfile(os.path.join(folder, n))]

def load_folder_faces(cascade, folder: str, label: int) -> Tuple[List[np.ndarray], List[int]]:
    imgs, labels = [], []
    for path in list_folder_images(folder):
        img = cv2.imread(path)
        if img is None:
            continue
        face = detect_largest_face(cascade, img)
        if face is None:
            _log(f"В '{path}' лицо не найдено — пропускаю.")
            continue
        gray = crop_face_region(img, face)
        imgs.append(gray)
        labels.append(label)
    return imgs, labels

def prepare_training_data(cascade) -> Tuple[List[np.ndarray], np.ndarray]:
    auth_imgs, auth_labels = load_folder_faces(cascade, DATA_AUTH_DIR, AUTHORIZED_CLASS_LABEL)
    other_imgs, other_labels = load_folder_faces(cascade, DATA_OTHER_DIR, OTHER_CLASS_LABEL)
    if not auth_imgs:
        _log(f"Нет валидных фото в {DATA_AUTH_DIR} — обучить модель невозможно.")
        sys.exit(1)
    imgs = auth_imgs + other_imgs
    labels = auth_labels + other_labels
    _log(f"Обучение: authorized={len(auth_imgs)}; others={len(other_imgs)}")
    return imgs, np.array(labels, dtype=np.int32)

def ensure_model(cascade) -> "cv2.face_LBPHFaceRecognizer":
    if not hasattr(cv2, "face"):
        _log("Нет модуля 'cv2.face'. Установите: pip install opencv-contrib-python")
        sys.exit(1)
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    if os.path.isfile(MODEL_PATH):
        recognizer.read(MODEL_PATH)
        _log(f"Модель загружена из {MODEL_PATH}")
        return recognizer
    imgs, labels = prepare_training_data(cascade)
    recognizer.train(imgs, labels)
    recognizer.save(MODEL_PATH)
    _log(f"Модель обучена и сохранена в {MODEL_PATH}")
    return recognizer

# ---------- камера ----------
def warmup_camera(cap: cv2.VideoCapture, seconds=WARMUP_SEC) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        grabbed, frame = cap.read()
        if grabbed and frame is not None and float(frame.mean()) > BLACK_FRAME_MEAN:
            return True
        time.sleep(0.03)
    return False

def open_camera_candidates():
    if os.name == "nt":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    else:
        backends = [cv2.CAP_V4L2]
    for be in backends:
        for idx in (0, 1, 2, 3):
            yield (be, idx)
    for idx in (0, 1, 2, 3):
        yield (None, idx)

def open_camera() -> Optional[cv2.VideoCapture]:
    for backend, idx in open_camera_candidates():
        try:
            cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                if warmup_camera(cap):
                    _log(f"Камера открыта: index={idx}, backend={backend if backend is not None else 'default'}")
                    return cap
                cap.release()
        except Exception:
            pass
    _log("Камера не открыта (возможно, выключена/занята).")
    return None

def reopen_if_black_or_dead(cap: Optional[cv2.VideoCapture]) -> Optional[cv2.VideoCapture]:
    if cap is None or not cap.isOpened():
        return open_camera()
    ok, frame = cap.read()
    if not ok or frame is None or float(frame.mean()) <= BLACK_FRAME_MEAN:
        try: cap.release()
        except Exception: pass
        return open_camera()
    return cap

# ---------- распознавание ----------
def predict_once(cascade, recognizer, frame_bgr: np.ndarray) -> Tuple[str, float, Optional[np.ndarray]]:
    """
    Возвращает (label_name, confidence, face_rect)
    label_name ∈ {"authorized","others","no_face"}
    """
    face_rect = detect_largest_face(cascade, frame_bgr)
    if face_rect is None:
        return "no_face", float("inf"), None
    gray = crop_face_region(frame_bgr, face_rect)
    label, confidence = recognizer.predict(gray)
    name = "authorized" if label == AUTHORIZED_CLASS_LABEL else "others"
    return name, float(confidence), face_rect

def load_runtime_threshold() -> Optional[float]:
    if os.path.isfile(RUNTIME_THRESHOLD_FILE):
        try:
            with open(RUNTIME_THRESHOLD_FILE, "r", encoding="utf-8") as f:
                return float(f.read().strip())
        except Exception:
            return None
    return None

def save_runtime_threshold(th: float):
    try:
        with open(RUNTIME_THRESHOLD_FILE, "w", encoding="utf-8") as f:
            f.write(str(th))
        _log(f"Порог сохранён: {th:.1f}")
    except Exception:
        pass

def calibrate_threshold(cascade, recognizer, cap: Optional[cv2.VideoCapture]) -> Optional[float]:
    if cap is None or not cap.isOpened():
        _log("Калибровка: камера недоступна.")
        return None
    _log("Калибровка: посмотрите прямо в камеру…")
    t0 = time.time()
    vals: List[float] = []
    while time.time() - t0 < CALIBRATION_SECONDS:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        name, conf, _ = predict_once(cascade, recognizer, frame)
        if name == "authorized":
            vals.append(conf)
    if len(vals) < 5:
        _log("Калибровка: мало кадров authorized, порог не обновлён.")
        return None
    med = float(np.median(vals))
    th = med + CALIBRATION_MARGIN
    _log(f"Калибровка: медиана={med:.1f} => порог={th:.1f} (медиана + {CALIBRATION_MARGIN})")
    return th

def check_window(cascade, recognizer, cap_ref: Optional[cv2.VideoCapture], threshold: float) -> Tuple[bool, Optional[cv2.VideoCapture]]:
    """
    True — проверка пройдена/перенесена, False — нарушение.
    Успешный кадр = (label == authorized) И (confidence <= threshold).
    Требуется >= REQUIRED_OK_IN_WINDOW успешных кадров в окне.
    """
    cap = cap_ref
    deadline = time.time() + RETRY_WINDOW_SEC
    period = 1.0 / max(1, RETRY_FPS)

    ok_count = 0
    total = 0

    while time.time() < deadline and total < WINDOW_SIZE:
        if cap is None or not cap.isOpened():
            cap = reopen_if_black_or_dead(cap)
            if cap is None:
                _log("Камера недоступна — перенос проверки.")
                return True, cap
            continue

        grabbed, frame = cap.read()
        if not grabbed or frame is None or float(frame.mean()) <= BLACK_FRAME_MEAN:
            cap = reopen_if_black_or_dead(cap)
            continue

        name, conf, _ = predict_once(cascade, recognizer, frame)
        passed_this = (name == "authorized" and conf <= threshold)

        total += 1
        if passed_this:
            ok_count += 1

        _log(f"Кадр {total}/{WINDOW_SIZE}: {name} (conf={conf:.1f} thr={threshold:.1f}) "
             f"{'OK' if passed_this else 'NO'}  ok_count={ok_count}")

        if ok_count >= REQUIRED_OK_IN_WINDOW:
            return True, cap

        time.sleep(period)

    return ok_count >= REQUIRED_OK_IN_WINDOW, cap

# ================== КАЛЬКУЛЯТОР (Tkinter) ==================
class Calculator(tk.Tk):
    def __init__(self, on_close_cb=None):
        super().__init__()
        self.title(WIN_TITLE)
        self.geometry(WIN_SIZE)
        self.resizable(False, False)
        self.expression = tk.StringVar(value="")

        self.status_var = tk.StringVar(value="Доступ разрешён. Идёт периодическая проверка лица…")
        self._build_ui()

        self._on_close_cb = on_close_cb
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        display = tk.Entry(self, textvariable=self.expression, font=("Segoe UI", 20),
                           justify="right", bd=8, relief="groove")
        display.pack(fill="x", padx=10, pady=(10,4), ipady=10)

        status = tk.Label(self, textvariable=self.status_var, anchor="w")
        status.pack(fill="x", padx=12, pady=(0,8))

        self.buttons = {}
        btns = [
            ("C",   self.clear), ("⌫",  self.backspace), ("(",  lambda: self.add("(")), (")", lambda: self.add(")")),
            ("7",   lambda: self.add("7")), ("8", lambda: self.add("8")), ("9", lambda: self.add("9")), ("/", lambda: self.add("/")),
            ("4",   lambda: self.add("4")), ("5", lambda: self.add("5")), ("6", lambda: self.add("6")), ("*", lambda: self.add("*")),
            ("1",   lambda: self.add("1")), ("2", lambda: self.add("2")), ("3", lambda: self.add("3")), ("-", lambda: self.add("-")),
            ("0",   lambda: self.add("0")), (".", lambda: self.add(".")), ("^", lambda: self.add("^")),  ("+", lambda: self.add("+")),
            ("=",   self.equals),
        ]

        grid = tk.Frame(self)
        grid.pack(expand=True, fill="both", padx=10, pady=(0,10))

        r, c = 0, 0
        for text, cmd in btns:
            b = tk.Button(grid, text=text, command=cmd, font=("Segoe UI", 14), bd=3, relief="raised")
            b.grid(row=r, column=c, sticky="nsew", padx=4, pady=4, ipadx=4, ipady=10)
            self.buttons[text] = b
            c += 1
            if c == 4:
                c = 0
                r += 1

        for i in range(r+1):
            grid.rowconfigure(i, weight=1)
        for j in range(4):
            grid.columnconfigure(j, weight=1)

    def set_status(self, text: str):
        self.status_var.set(text)

    def add(self, char):
        self.expression.set(self.expression.get() + char)

    def clear(self):
        self.expression.set("")

    def backspace(self):
        self.expression.set(self.expression.get()[:-1])

    def equals(self):
        expr = self.expression.get()
        try:
            if not expr:
                raise ValueError("Пустое выражение")
            # поддержка степени: ^ -> **
            expr_eval = expr.replace("^", "**")
            # проверка допустимых символов
            allowed = "0123456789+-*/(). ^"
            if any(ch not in allowed for ch in expr_eval.replace("**", "^")):
                raise ValueError("Недопустимые символы")
            result = eval(expr_eval, {"__builtins__": {}}, {})
            self.expression.set(str(result))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неверное выражение.\n{e}")

    def on_close(self):
        try:
            if self._on_close_cb:
                self._on_close_cb()
        finally:
            self.destroy()

# ================== СВЯЗКА: лицо + калькулятор ==================
class FaceWatcher:
    """Фоновый сторож: каждые CHECK_INTERVAL_SEC проверяет лицо; при провале — закрывает GUI."""
    def __init__(self, cascade, recognizer, threshold: float, app: Calculator):
        self.cascade = cascade
        self.recognizer = recognizer
        self.threshold = threshold
        self.app = app

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self.cap = open_camera()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self.cap:
            try: self.cap.release()
            except Exception: pass

    def _worker(self):
        try:
            while not self._stop.is_set():
                # Плановая проверка
                self.app.after(0, lambda: self.app.set_status("Проверка личности…"))
                passed, new_cap = check_window(self.cascade, self.recognizer, self.cap, self.threshold)
                self.cap = new_cap
                if not passed:
                    # Закрываем приложение из GUI-потока
                    def _close():
                        try:
                            messagebox.showwarning("Доступ запрещён", "Вы не являетесь владельцем. Программа будет закрыта.")
                        finally:
                            self.app.on_close()
                    self.app.after(0, _close)
                    break
                else:
                    self.app.after(0, lambda: self.app.set_status("Доступ разрешён. Следующая проверка через 30 сек."))
                # Ждём до следующей проверки
                for _ in range(int(CHECK_INTERVAL_SEC*10)):
                    if self._stop.is_set(): break
                    time.sleep(0.1)
        finally:
            if self.cap:
                try: self.cap.release()
                except Exception: pass

# ================== MAIN ==================
def main():
    # Каскад
    cascade_path = os.path.join(cv2.data.haarcascades, HAAR_NAME)
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        _log("Не удалось загрузить Haar Cascade. Установите/переустановите OpenCV.")
        sys.exit(1)

    # Модель
    recognizer = ensure_model(cascade)

    # Камера для калибровки и первичной проверки
    cap = open_camera()

    # Порог
    threshold = load_runtime_threshold() or LBPH_BASE_THRESHOLD
    if CALIBRATE_ON_START:
        t = calibrate_threshold(cascade, recognizer, cap)
        if t is not None:
            threshold = t
            save_runtime_threshold(threshold)

    # Начальная проверка перед запуском калькулятора
    _log("Первичная проверка личности перед запуском калькулятора…")
    ok, cap = check_window(cascade, recognizer, cap, threshold)
    if cap:
        try: cap.release()
        except Exception: pass
    if not ok:
        _log("Доступ запрещён. Приложение не будет запущено.")
        try:
            messagebox.showerror("Доступ запрещён", "Вы не являетесь владельцем. Программа закрыта.")
        except Exception:
            pass
        sys.exit(1)

    # Запуск калькулятора + фоновый сторож
    app = Calculator()
    watcher = FaceWatcher(cascade, recognizer, threshold, app)
    app._on_close_cb = watcher.stop  # корректно остановим сторожа и освободим камеру
    watcher.start()

    app.mainloop()

if __name__ == "__main__":
    main()
