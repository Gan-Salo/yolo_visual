import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import urllib.request
from collections import OrderedDict

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.grid_size = 0
        
    def forward(self, x, img_dim=None):
        # Получаем размеры входа
        bs, _, ny, nx = x.shape
        
        # Преобразуем формат
        x = x.view(bs, self.num_anchors, self.num_classes + 5, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        
        # Если не в режиме обучения, преобразуем выходы
        if not self.training:
            if self.grid_size != nx:
                self.grid_size = nx
                # Создаем сетку координат с явным указанием indexing='ij'
                yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
                self.grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float().to(x.device)
                
            # Применяем сигмоиду и добавляем смещение сетки
            x[..., 0:2] = (torch.sigmoid(x[..., 0:2]) + self.grid) / self.grid_size
            # Применяем экспоненту для ширины и высоты
            x[..., 2:4] = torch.exp(x[..., 2:4]) * torch.tensor(self.anchors).to(x.device).view(1, self.num_anchors, 1, 1, 2) / self.img_dim
            # Сигмоида для уверенности и классов
            x[..., 4:] = torch.sigmoid(x[..., 4:])
            # Объединяем по сетке
            x = x.view(bs, -1, self.num_classes + 5)
            
        return x

class Darknet(nn.Module):
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)
        
    def forward(self, x):
        img_dim = x.shape[2]
        layer_outputs, yolo_outputs = [], []
        
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "convolutional" or module_def["type"] == "upsample" or module_def["type"] == "maxpool":
                x = module(x)
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def["type"] == "yolo":
                x = module[0](x, img_dim)
                yolo_outputs.append(x)
            layer_outputs.append(x)
        
        return yolo_outputs if self.training else torch.cat(yolo_outputs, 1)
    
    def load_darknet_weights(self, weights_path):
        """Загрузка весов из файла .weights"""
        # Открываем файл весов
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # Первые 5 значений - заголовок
            self.header_info = header
            self.seen = header[3]
            weights = np.fromfile(f, dtype=np.float32)  # Остальное - веса
        
        # Проходим по всем слоям и загружаем веса
        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Загрузка весов для слоя с батч-нормализацией
                    bn_layer = module[1]
                    
                    # Загрузка параметров батч-нормализации
                    num_b = bn_layer.bias.numel()
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Загрузка весов смещения для слоя без батч-нормализации
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                
                # Загрузка весов свертки
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

def parse_model_config(path):
    """Парсинг конфигурационного файла модели"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    module_defs = []
    for line in lines:
        if line.startswith('['):  # Новый блок
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            if "=" in line:
                key, value = line.split("=", 1)
                value = value.strip()
                module_defs[-1][key.rstrip()] = value.strip()
    
    return module_defs

def create_modules(module_defs):
    """Создание модулей PyTorch из определений модулей"""
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()
        
        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"]) if "batch_normalize" in module_def else 0
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
        
        elif module_def["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)
        
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())
        
        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())
        
        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])
            
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)
        
        module_list.append(modules)
        output_filters.append(filters)
    
    return hyperparams, module_list

class EmptyLayer(nn.Module):
    """Пустой слой для маршрутизации и сокращения"""
    def __init__(self):
        super(EmptyLayer, self).__init__()

def download_weights_and_config():
    """Загрузка весов и конфигурационного файла YOLOv4"""
    weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
    config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
    
    weights_path = "yolov4.weights"
    config_path = "yolov4.cfg"
    
    # Проверяем, существуют ли файлы
    if not os.path.exists(weights_path):
        print(f"Загрузка весов YOLOv4 из {weights_url}...")
        urllib.request.urlretrieve(weights_url, weights_path)
    
    if not os.path.exists(config_path):
        print(f"Загрузка конфигурации YOLOv4 из {config_url}...")
        urllib.request.urlretrieve(config_url, config_path)
    
    return config_path, weights_path

def load_yolo_model():
    """Загрузка модели YOLO с предобученными весами"""
    try:
        config_path, weights_path = download_weights_and_config()
        
        # Создаем модель
        model = Darknet(config_path)
        
        # Загружаем веса
        model.load_darknet_weights(weights_path)
        
        # Переводим модель в режим оценки
        model.eval()
        
        return model
    except Exception as e:
        print(f"Ошибка загрузки модели: {str(e)}")
        return None

def detect_objects(model, image, conf_thres=0.5, nms_thres=0.4):
    """Обнаружение объектов на изображении с использованием YOLO"""
    try:
        if model is None:
            print("Модель не загружена")
            return [None]  # Возвращаем список с None для совместимости
        
        # Преобразуем изображение для модели
        img = preprocess_image(image)
        
        # Получаем предсказания
        with torch.no_grad():
            detections = model(img)
        
        # Обрабатываем детекции
        detections = post_process_detections(detections, conf_thres, nms_thres)
        
        return detections
    except Exception as e:
        print(f"Ошибка обнаружения объектов: {str(e)}")
        return [None]  # Возвращаем список с None для совместимости

def preprocess_image(image, img_size=416):
    """Предобработка изображения для YOLOv4"""
    # Изменяем размер изображения
    img = cv2.resize(image, (img_size, img_size))
    
    # Преобразуем в тензор PyTorch
    img = img[:, :, ::-1].transpose((2, 0, 1))  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    
    return img

def post_process_detections(detections, conf_thres=0.5, nms_thres=0.4):
    """Постобработка детекций YOLOv4"""
    # Применяем порог уверенности
    detections = non_max_suppression(detections, conf_thres, nms_thres)
    
    return detections

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """Применение non-maximum suppression к предсказаниям"""
    # Преобразуем центральные координаты в координаты углов
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    
    output = [None for _ in range(len(prediction))]
    
    for image_i, image_pred in enumerate(prediction):
        # Фильтруем по порогу уверенности
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        
        # Если нет детекций
        if not image_pred.size(0):
            continue
        
        # Получаем оценки и классы
        class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        
        # Получаем уникальные классы
        unique_labels = detections[:, -1].cpu().unique()
        
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        
        for c in unique_labels:
            # Получаем детекции с текущим классом
            detections_class = detections[detections[:, -1] == c]
            
            # Сортируем по уверенности
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            
            # Применяем NMS
            max_detections = []
            while detections_class.size(0):
                # Добавляем детекцию с наивысшей уверенностью
                max_detections.append(detections_class[0].unsqueeze(0))
                
                # Останавливаемся, если нет больше детекций
                if len(detections_class) == 1:
                    break
                
                # Получаем IoU для всех оставшихся боксов
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                
                # Удаляем детекции с IoU > порога
                detections_class = detections_class[1:][ious < nms_thres]
            
            if max_detections:
                max_detections = torch.cat(max_detections).data
                
                # Добавляем к выходу
                output[image_i] = (
                    max_detections if output[image_i] is None else 
                    torch.cat((output[image_i], max_detections))
                )
    
    return output

def bbox_iou(box1, box2):
    """Вычисление IoU между двумя боксами"""
    # Получаем координаты боксов
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    
    # Получаем координаты пересечения
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    
    # Вычисляем площадь пересечения
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    
    # Вычисляем площади боксов
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    
    # Вычисляем IoU
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    
    return iou
