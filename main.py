import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import filedialog, Scale, HORIZONTAL
from PIL import Image, ImageTk
import io
import os
import urllib.request
import torch.nn as nn
from collections import OrderedDict
from torchvision import transforms
import random

# Импортируем модуль с предобученной моделью YOLOv4
from yolo_model import load_yolo_model, detect_objects
# Импортируем имена классов COCO
from coco_names import COCO_NAMES

# COCO dataset names
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Создаем словарь цветов для классов
COLORS = {}
random.seed(42)  # Для воспроизводимости
for class_name in COCO_NAMES:
    COLORS[class_name] = (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    )


class YOLOv4FeatureVisualizer:
    def __init__(self, master):
        self.master = master
        self.master.title("YOLOv4 Feature Map Visualizer")
        self.master.geometry("1200x900")

        # Variables
        self.model = None
        self.pretrained_model = None  # Предобученная модель YOLOv4
        self.img = None
        self.feature_maps = {}
        self.current_layer_idx = 0
        self.layers_names = []
        self.detections = []  # Для хранения обнаруженных объектов
        self.detection_img = None  # Для хранения изображения с размеченными объектами
        self.confidence_threshold = 0.4  # Порог уверенности по умолчанию

        # Create frames
        self.create_frames()

        # Create widgets
        self.create_widgets()

        # Initialize model
        self.load_yolov4_model()

    def create_frames(self):
        # Main layout with top control panel and bottom visualization area
        self.control_frame = tk.Frame(self.master, height=70)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Frame для настроек
        self.settings_frame = tk.Frame(self.master, height=50)
        self.settings_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.vis_frame = tk.Frame(self.master)
        self.vis_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for original image and detections
        self.left_frame = tk.Frame(self.vis_frame, width=500)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frames for original and detection images
        self.orig_img_frame = tk.Frame(self.left_frame)
        self.orig_img_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.detect_img_frame = tk.Frame(self.left_frame)
        self.detect_img_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Right frame for feature maps
        self.right_frame = tk.Frame(self.vis_frame, width=700)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Frame for the layer navigation
        self.layer_nav_frame = tk.Frame(self.right_frame, height=50)
        self.layer_nav_frame.pack(side=tk.TOP, fill=tk.X)

        # Frame for the feature maps
        self.feature_map_frame = tk.Frame(self.right_frame)
        self.feature_map_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_widgets(self):
        # Control panel widgets
        self.load_img_btn = tk.Button(self.control_frame, text="Загрузить изображение", command=self.load_image)
        self.load_img_btn.pack(side=tk.LEFT, padx=5)

        self.process_btn = tk.Button(self.control_frame, text="Запустить анализ", command=self.process_image)
        self.process_btn.pack(side=tk.LEFT, padx=5)

        self.detect_btn = tk.Button(self.control_frame, text="Обнаружить объекты", command=self.detect_objects)
        self.detect_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.control_frame, text="Статус: Готов")
        self.status_label.pack(side=tk.RIGHT, padx=5)

        # Confidence threshold slider
        self.conf_label = tk.Label(self.settings_frame, text="Порог уверенности:")
        self.conf_label.pack(side=tk.LEFT, padx=5)

        self.conf_slider = Scale(self.settings_frame, from_=0.1, to=0.9, resolution=0.05,
                                 orient=HORIZONTAL, length=200,
                                 command=self.update_confidence)
        self.conf_slider.set(self.confidence_threshold)
        self.conf_slider.pack(side=tk.LEFT, padx=5)

        self.conf_value_label = tk.Label(self.settings_frame, text=f"{self.confidence_threshold:.2f}")
        self.conf_value_label.pack(side=tk.LEFT, padx=5)

        # Original image label
        self.img_label = tk.Label(self.orig_img_frame, text="Оригинальное изображение")
        self.img_label.pack(side=tk.TOP, pady=5)

        self.original_img_canvas = tk.Canvas(self.orig_img_frame, bg="gray", height=300, width=400)
        self.original_img_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Detection image label
        self.detect_label = tk.Label(self.detect_img_frame, text="Обнаруженные объекты")
        self.detect_label.pack(side=tk.TOP, pady=5)

        self.detect_img_canvas = tk.Canvas(self.detect_img_frame, bg="gray", height=300, width=400)
        self.detect_img_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Detection info Text widget
        self.detect_info_text = tk.Text(self.detect_img_frame, height=5, width=48)
        self.detect_info_text.pack(fill=tk.X, expand=False, padx=10, pady=5)
        self.detect_info_text.insert(tk.END, "Информация об обнаруженных объектах будет отображаться здесь")
        self.detect_info_text.config(state=tk.DISABLED)

        # Layer navigation widgets
        self.prev_layer_btn = tk.Button(self.layer_nav_frame, text="← Пред. слой", command=self.prev_layer)
        self.prev_layer_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.layer_label = tk.Label(self.layer_nav_frame, text="Слой: -")
        self.layer_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.next_layer_btn = tk.Button(self.layer_nav_frame, text="След. слой →", command=self.next_layer)
        self.next_layer_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.layer_info_label = tk.Label(self.layer_nav_frame, text="Тип: -", justify=tk.LEFT)
        self.layer_info_label.pack(side=tk.LEFT, padx=20, pady=5)

        # Feature map canvas
        self.feature_map_label = tk.Label(self.feature_map_frame, text="Карты признаков")
        self.feature_map_label.pack(side=tk.TOP, pady=5)

        self.feature_canvas = tk.Canvas(self.feature_map_frame, bg="gray", height=500)
        self.feature_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def update_confidence(self, value):
        """Обновление порога уверенности"""
        self.confidence_threshold = float(value)
        self.conf_value_label.config(text=f"{self.confidence_threshold:.2f}")
        # Если есть обнаруженные объекты, пересчитываем с новым порогом
        if hasattr(self, 'detections_before_threshold'):
            self.apply_confidence_threshold()

    def apply_confidence_threshold(self):
        """Применяет порог уверенности к обнаруженным объектам"""
        if not hasattr(self, 'detections_before_threshold'):
            return

        # Фильтруем детекции по порогу уверенности
        self.detections = [det for det in self.detections_before_threshold
                           if det['conf'] >= self.confidence_threshold]

        # Отображаем детекции на изображении
        self.display_detections()

    def load_yolov4_model(self):
        """Загрузка предобученной модели YOLOv4"""
        try:
            self.status_label.config(text="Статус: Загрузка модели YOLOv4...")

            # Создаем базовую структуру YOLOv4 (упрощенная версия для визуализации)
            self.model = YOLOv4Base()
            
            # Загружаем предобученную модель YOLOv4
            self.status_label.config(text="Статус: Загрузка предобученной модели YOLOv4...")
            self.pretrained_model = load_yolo_model()
            self.status_label.config(text="Статус: Предобученная модель YOLOv4 загружена")

            self.status_label.config(text="Статус: Модель загружена")
            self.layers_names = list(self.model.features._modules.keys())

        except Exception as e:
            self.status_label.config(text=f"Ошибка: {str(e)}")
            print(f"Ошибка загрузки модели: {str(e)}")

    def load_image(self):
        """Загрузка изображения"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")]
            )

            if not file_path:
                return

            self.img = cv2.imread(file_path)
            if self.img is None:
                raise ValueError("Не удалось загрузить изображение")

            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

            # Отображаем исходное изображение
            self.display_original_image()

            self.status_label.config(text="Статус: Изображение загружено")

        except Exception as e:
            self.status_label.config(text=f"Ошибка: {str(e)}")
            print(f"Ошибка загрузки изображения: {str(e)}")

    def display_original_image(self):
        """Отображает исходное изображение на canvas"""
        if self.img is None:
            return

        h, w = self.img.shape[:2]
        canvas_w = self.original_img_canvas.winfo_width()
        canvas_h = self.original_img_canvas.winfo_height()

        # Подгоняем изображение под размер канваса
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img_resized = cv2.resize(self.img, (new_w, new_h))

        # Конвертируем в PIL Image
        img_pil = Image.fromarray(img_resized)
        self.tk_img = ImageTk.PhotoImage(image=img_pil)

        # Отображаем на канвасе
        self.original_img_canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            image=self.tk_img, anchor=tk.CENTER
        )

    def detect_objects(self):
        """Обнаружение объектов на изображении с использованием предобученной модели YOLOv4"""
        if self.img is None:
            self.status_label.config(text="Ошибка: Сначала загрузите изображение")
            return

        try:
            self.status_label.config(text="Статус: Обнаружение объектов...")

            # Используем предобученную модель YOLOv4 для обнаружения объектов
            detections = detect_objects(self.pretrained_model, self.img, conf_thres=0.1)  # Низкий порог для сбора всех детекций

            # Сохраняем детекции
            self.detections_before_threshold = []
            if detections[0] is not None:
                for detection in detections[0]:
                    x1, y1, x2, y2, obj_conf, class_conf, class_pred = detection

                    # Масштабируем координаты под размер исходного изображения
                    h, w = self.img.shape[:2]

                    # Добавляем проверку на бесконечность и NaN
                    try:
                        x1_val = x1.item() * w / 416
                        y1_val = y1.item() * h / 416
                        x2_val = x2.item() * w / 416
                        y2_val = y2.item() * h / 416
                        
                        # Проверка на бесконечность и NaN
                        if (not np.isfinite(x1_val) or not np.isfinite(y1_val) or 
                            not np.isfinite(x2_val) or not np.isfinite(y2_val)):
                            continue
                            
                        x1 = max(0, min(w-1, int(x1_val)))
                        y1 = max(0, min(h-1, int(y1_val)))
                        x2 = max(0, min(w-1, int(x2_val)))
                        y2 = max(0, min(h-1, int(y2_val)))
                    except (ValueError, OverflowError):
                        # Пропускаем некорректные значения
                        continue

                    class_idx = int(class_pred.item())
                    class_name = COCO_NAMES[class_idx]

                    # Добавляем в список детекций
                    self.detections_before_threshold.append({
                        'box': (x1, y1, x2, y2),
                        'conf': float(obj_conf * class_conf),
                        'class_idx': class_idx,
                        'class_name': class_name
                    })

            # Применяем порог уверенности
            self.apply_confidence_threshold()

            self.status_label.config(text="Статус: Обнаружение объектов завершено")

        except Exception as e:
            self.status_label.config(text=f"Ошибка: {str(e)}")
            print(f"Ошибка обнаружения объектов: {str(e)}")

    def display_detections(self):
        """Отображение детекций на изображении"""
        if not hasattr(self, 'detections'):
            # Если детекций нет, просто показываем оригинальное изображение
            self.detection_img = self.img.copy()

            # Обновляем текстовую информацию
            self.detect_info_text.config(state=tk.NORMAL)
            self.detect_info_text.delete(1.0, tk.END)
            self.detect_info_text.insert(tk.END, "Объекты не обнаружены")
            self.detect_info_text.config(state=tk.DISABLED)

        elif not self.detections:
            # Если детекций нет, просто показываем оригинальное изображение
            self.detection_img = self.img.copy()

            # Обновляем текстовую информацию
            self.detect_info_text.config(state=tk.NORMAL)
            self.detect_info_text.delete(1.0, tk.END)
            self.detect_info_text.insert(tk.END, f"Объекты не обнаружены (порог: {self.confidence_threshold:.2f})")
            self.detect_info_text.config(state=tk.DISABLED)

        else:
            # Копируем изображение для рисования
            self.detection_img = self.img.copy()

            # Рисуем боксы и метки
            for i, det in enumerate(self.detections):
                x1, y1, x2, y2 = det['box']
                class_name = det['class_name']
                conf_value = det['conf']

                # Получаем цвет для данного класса
                color = COLORS.get(class_name, (0, 255, 0))

                # Преобразуем формат цвета для OpenCV (BGR)
                color_bgr = (color[2], color[1], color[0])

                # Рисуем бокс с более заметным цветом
                self.detection_img = cv2.rectangle(self.detection_img, (x1, y1), (x2, y2), color_bgr, 2)

                # Создаем метку с номером, названием и уверенностью
                label = f"{i + 1}. {class_name} ({conf_value:.2f})"

                # Рисуем фон для текста для лучшей видимости
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                self.detection_img = cv2.rectangle(
                    self.detection_img,
                    (x1, y1 - 25),
                    (x1 + text_size[0], y1),
                    color_bgr, -1  # Заполненный прямоугольник
                )

                # Рисуем текст
                self.detection_img = cv2.putText(
                    self.detection_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

            # Обновляем текстовую информацию
            self.detect_info_text.config(state=tk.NORMAL)
            self.detect_info_text.delete(1.0, tk.END)

            self.detect_info_text.insert(tk.END,
                                         f"Обнаружено объектов: {len(self.detections)} (порог: {self.confidence_threshold:.2f})\n\n")

            # Создаем словарь для подсчета объектов по классам
            class_counts = {}
            for det in self.detections:
                class_name = det['class_name']
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

            # Выводим статистику по классам
            self.detect_info_text.insert(tk.END, "Статистика по классам:\n")
            for class_name, count in class_counts.items():
                self.detect_info_text.insert(tk.END, f"- {class_name}: {count}\n")

            self.detect_info_text.insert(tk.END, "\nДетализация:\n")
            for i, det in enumerate(self.detections):
                info_text = f"{i + 1}. {det['class_name']} (уверенность: {det['conf']:.2f})\n"
                self.detect_info_text.insert(tk.END, info_text)

            self.detect_info_text.config(state=tk.DISABLED)

        # Отображаем изображение с детекциями
        self.display_detection_image()

    def display_detection_image(self):
        """Отображает изображение с детекциями на canvas"""
        if self.detection_img is None:
            return

        h, w = self.detection_img.shape[:2]
        canvas_w = self.detect_img_canvas.winfo_width()
        canvas_h = self.detect_img_canvas.winfo_height()

        # Подгоняем изображение под размер канваса
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        img_resized = cv2.resize(self.detection_img, (new_w, new_h))

        # Конвертируем в PIL Image
        img_pil = Image.fromarray(img_resized)
        self.tk_detect_img = ImageTk.PhotoImage(image=img_pil)

        # Отображаем на канвасе
        self.detect_img_canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            image=self.tk_detect_img, anchor=tk.CENTER
        )

    def process_image(self):
        """Обработка изображения через YOLOv4 и получение карт признаков"""
        if self.img is None:
            self.status_label.config(text="Ошибка: Сначала загрузите изображение")
            return

        try:
            self.status_label.config(text="Статус: Анализ изображения...")

            # Подготавливаем изображение для модели
            img_tensor = self.prepare_image(self.img)

            # Регистрируем хуки для получения активаций
            self.feature_maps = {}
            hooks = []

            # Функция для хука
            def get_activation(name):
                def hook(model, input, output):
                    self.feature_maps[name] = output.detach().cpu().numpy()

                return hook

            # Регистрируем хуки для получения активаций со всех слоев
            for name, layer in self.model.features.named_children():
                hooks.append(layer.register_forward_hook(get_activation(name)))

            # Прогоняем изображение через модель
            with torch.no_grad():
                output, detections = self.model(img_tensor)

            # Убираем хуки
            for hook in hooks:
                hook.remove()

            # Отображаем карты признаков для первого слоя
            self.current_layer_idx = 0
            self.display_feature_maps()

            self.status_label.config(text="Статус: Анализ завершен")

        except Exception as e:
            self.status_label.config(text=f"Ошибка: {str(e)}")
            print(f"Ошибка обработки изображения: {str(e)}")

    def prepare_image(self, img):
        """Подготовка изображения для модели YOLOv4"""
        img_resized = cv2.resize(img, (416, 416))
        img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor / 255.0  # Нормализация
        return img_tensor

    def display_feature_maps(self):
        """Отображает карты признаков для текущего слоя"""
        if not self.feature_maps:
            return

        if self.current_layer_idx < 0 or self.current_layer_idx >= len(self.layers_names):
            return

        layer_name = self.layers_names[self.current_layer_idx]
        self.layer_label.config(text=f"Слой: {layer_name}")

        # Получаем информацию о слое
        layer_type = type(self.model.features._modules[layer_name]).__name__
        self.layer_info_label.config(text=f"Тип: {layer_type}")

        # Получаем карты признаков для текущего слоя
        feature_map = self.feature_maps[layer_name]

        # Очищаем канвас
        self.feature_canvas.delete("all")

        # Отображаем карты признаков в виде сетки
        self.display_feature_grid(feature_map)

    def display_feature_grid(self, feature_map):
        """Отображает сетку карт признаков"""
        batch, channels, height, width = feature_map.shape

        # Определим размер сетки
        grid_size = min(4, channels)  # Максимум 4x4 карты признаков
        canvas_w = self.feature_canvas.winfo_width()
        canvas_h = self.feature_canvas.winfo_height()

        # Создаем figure для matplotlib
        fig = plt.Figure(figsize=(10, 10), dpi=100)

        # Количество карт для отображения
        num_maps = min(16, channels)  # Максимум 16 карт признаков

        # Создаем подграфики для каждой карты признаков
        for i in range(num_maps):
            ax = fig.add_subplot(4, 4, i + 1)

            # Получаем карту признаков
            f_map = feature_map[0, i, :, :]

            # Нормализуем для отображения
            f_map = (f_map - f_map.min()) / (f_map.max() - f_map.min() + 1e-10)

            # Отображаем
            ax.imshow(f_map, cmap='viridis')
            ax.set_title(f'Filter {i}')
            ax.axis('off')

        fig.tight_layout()

        # Конвертируем figure в изображение
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)

        # Подгоняем размер
        scale = min(canvas_w / img.width, canvas_h / img.height) * 0.9
        new_w, new_h = int(img.width * scale), int(img.height * scale)
        img_resized = img.resize((new_w, new_h))

        # Отображаем на канвасе
        self.feat_map_img = ImageTk.PhotoImage(image=img_resized)
        self.feature_canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            image=self.feat_map_img, anchor=tk.CENTER
        )

        # Закрываем буфер
        buf.close()

    def next_layer(self):
        """Переход к следующему слою"""
        if not self.feature_maps:
            return

        if self.current_layer_idx < len(self.layers_names) - 1:
            self.current_layer_idx += 1
            self.display_feature_maps()

    def prev_layer(self):
        """Переход к предыдущему слою"""
        if not self.feature_maps:
            return

        if self.current_layer_idx > 0:
            self.current_layer_idx -= 1
            self.display_feature_maps()


class YOLOv4Base(nn.Module):
    """Упрощенная версия YOLOv4 для визуализации"""

    def __init__(self):
        super(YOLOv4Base, self).__init__()

        # Создаем упрощенную версию слоев YOLOv4 для визуализации
        self.features = nn.Sequential(OrderedDict([
            # Backbone (CSPDarknet53)
            # Начальный блок
            ('conv1', nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(32)),
            ('relu1', nn.LeakyReLU(0.1)),

            # Downsample 1
            ('conv2', nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.LeakyReLU(0.1)),

            # CSP Block 1
            ('conv3', nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.LeakyReLU(0.1)),

            # Downsample 2
            ('conv4', nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn4', nn.BatchNorm2d(128)),
            ('relu4', nn.LeakyReLU(0.1)),

            # CSP Block 2
            ('conv5', nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn5', nn.BatchNorm2d(64)),
            ('relu5', nn.LeakyReLU(0.1)),

            # Downsample 3
            ('conv6', nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn6', nn.BatchNorm2d(256)),
            ('relu6', nn.LeakyReLU(0.1)),

            # CSP Block 3
            ('conv7', nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn7', nn.BatchNorm2d(128)),
            ('relu7', nn.LeakyReLU(0.1)),

            # Downsample 4
            ('conv8', nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn8', nn.BatchNorm2d(512)),
            ('relu8', nn.LeakyReLU(0.1)),

            # CSP Block 4
            ('conv9', nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn9', nn.BatchNorm2d(256)),
            ('relu9', nn.LeakyReLU(0.1)),

            # Downsample 5
            ('conv10', nn.Conv2d(256, 1024, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn10', nn.BatchNorm2d(1024)),
            ('relu10', nn.LeakyReLU(0.1)),

            # CSP Block 5
            ('conv11', nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn11', nn.BatchNorm2d(512)),
            ('relu11', nn.LeakyReLU(0.1)),

            # SPP Block
            ('conv12', nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn12', nn.BatchNorm2d(256)),
            ('relu12', nn.LeakyReLU(0.1)),

            # Начало PANet
            ('conv13', nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False)),
            ('bn13', nn.BatchNorm2d(128)),
            ('relu13', nn.LeakyReLU(0.1)),
        ]))

        # Голова для детекции объектов (упрощенная)
        self.detector = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 255, kernel_size=1, stride=1, padding=0)  # 255 = 3 * (80 + 5)
        )

    def forward(self, x):
        features = self.features(x)
        return features


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOv4FeatureVisualizer(root)
    root.mainloop()