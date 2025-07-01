import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
from config import CLASS_COLORS

def create_result_image(image: np.ndarray, 
                       results: Dict[str, Any],
                       show_confidence: bool = True,
                       show_class_names: bool = True) -> np.ndarray:
    """
    Создает изображение с нарисованными результатами детекции
    
    Args:
        image: Исходное изображение (RGB)
        results: Результаты детекции от TrafficSignDetector
        show_confidence: Показывать ли уверенность
        show_class_names: Показывать ли названия классов
        
    Returns:
        Изображение с нарисованными детекциями
    """
    
    # Копируем изображение
    result_image = image.copy()
    
    # Конвертируем RGB -> BGR для OpenCV
    result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    
    # Рисуем каждую детекцию
    for detection in results['detections']:
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        # Координаты бокса
        x1, y1, x2, y2 = bbox
        
        # Цвет для класса
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        
        # Рисуем прямоугольник
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Формируем текст для отображения
        label_parts = []
        if show_class_names:
            label_parts.append(class_name)
        if show_confidence:
            label_parts.append(f"{confidence:.2%}")
        
        label = " | ".join(label_parts)
        
        # Настройки текста
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Размер текста для фона
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        
        # Рисуем фон для текста
        cv2.rectangle(
            result_image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Рисуем текст
        cv2.putText(
            result_image,
            label,
            (x1, y1 - 5),
            font,
            font_scale,
            (0, 0, 0),  # Черный текст
            thickness
        )
    
    # Конвертируем обратно BGR -> RGB
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    return result_image

def create_statistics_chart(detections: List[Dict[str, Any]]) -> go.Figure:
    """
    Создает график статистики обнаруженных классов
    
    Args:
        detections: Список детекций
        
    Returns:
        Plotly фигура с графиком
    """
    
    if not detections:
        # Пустой график если нет детекций
        fig = go.Figure()
        fig.add_annotation(
            text="Дорожные знаки не обнаружены",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Подсчитываем количество каждого класса
    class_counts = {}
    class_confidences = {}
    
    for detection in detections:
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        if class_name not in class_counts:
            class_counts[class_name] = 0
            class_confidences[class_name] = []
        
        class_counts[class_name] += 1
        class_confidences[class_name].append(confidence)
    
    # Подготавливаем данные для графика
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    avg_confidences = [np.mean(class_confidences[cls]) for cls in classes]
    
    # Создаем график
    fig = go.Figure()
    
    # Столбчатая диаграмма количества
    fig.add_trace(go.Bar(
        x=classes,
        y=counts,
        name='Количество',
        marker_color='lightblue',
        yaxis='y',
        text=counts,
        textposition='outside'
    ))
    
    # Линейный график средней уверенности
    fig.add_trace(go.Scatter(
        x=classes,
        y=avg_confidences,
        mode='lines+markers',
        name='Средняя уверенность',
        line=dict(color='red', width=3),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Настройки осей
    fig.update_layout(
        title="📊 Статистика обнаруженных дорожных знаков",
        xaxis_title="Тип знака",
        yaxis=dict(
            title="Количество",
            side="left"
        ),
        yaxis2=dict(
            title="Средняя уверенность",
            side="right",
            overlaying="y",
            tickformat=".0%"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    return fig