import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Any
from pathlib import Path

class TrafficSignDetector:
    """Класс для детекции дорожных знаков с помощью YOLO"""
    
    def __init__(self, model_path: str):
        """
        Инициализация детектора
        
        Args:
            model_path: Путь к файлу модели (.pt)
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        # Загружаем модель
        self.model = YOLO(model_path)
        
        # Получаем названия классов из модели
        self.class_names = self.model.names
        
        print(f"✅ Модель загружена: {model_path}")
        print(f"📊 Классов: {len(self.class_names)}")
    
    def detect(self, 
               image: np.ndarray, 
               conf_threshold: float = 0.5,
               iou_threshold: float = 0.4,
               img_size: int = 640) -> Dict[str, Any]:
        """
        Выполняет детекцию дорожных знаков на изображении
        
        Args:
            image: Изображение в формате numpy array (RGB)
            conf_threshold: Порог уверенности для детекции
            iou_threshold: Порог IoU для NMS
            img_size: Размер изображения для модели
            
        Returns:
            Словарь с результатами детекции
        """
        
        # Запускаем модель
        results = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=img_size,
            verbose=False
        )
        
        # Обрабатываем результаты
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Извлекаем данные о детекции
                bbox = boxes.xyxy[i].cpu().numpy()  # координаты бокса
                confidence = float(boxes.conf[i].cpu().numpy())  # уверенность
                class_id = int(boxes.cls[i].cpu().numpy())  # ID класса
                
                # Получаем название класса
                class_name = self.class_names[class_id]
                
                detection = {
                    'bbox': bbox.astype(int),  # [x1, y1, x2, y2]
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name
                }
                
                detections.append(detection)
        
        return {
            'detections': detections,
            'image_shape': image.shape,
            'model_info': {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'img_size': img_size
            }
        }
    
    def detect_from_file(self, image_path: str, **kwargs) -> Dict[str, Any]:
        """
        Детекция из файла изображения
        
        Args:
            image_path: Путь к изображению
            **kwargs: Дополнительные параметры для detect()
            
        Returns:
            Результаты детекции
        """
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Конвертируем BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return self.detect(image_rgb, **kwargs)