import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path

# Локальные импорты
from utils.detection import TrafficSignDetector
from utils.visualization import create_result_image, create_statistics_chart
from config import *

# Настройка страницы
st.set_page_config(
    page_title="🚦 Traffic Signs Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .detection-result {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E86AB;
    }
    .stats-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Загружаем модель один раз и кешируем"""
    detector = TrafficSignDetector(MODEL_PATH)
    return detector

def main():
    # Заголовок приложения
    st.markdown('<h1 class="main-header">🚦 Детекция дорожных знаков</h1>', 
                unsafe_allow_html=True)
    
    # Описание проекта
    st.markdown("""
    <div class="info-box">
    <h3>🎯 О проекте</h3>
    <p>Интеллектуальная система распознавания дорожных знаков для помощи водителям 
    и автономных транспортных средств. Система определяет 15 типов дорожных знаков 
    с точностью 95.9%.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Настройки детекции
        confidence_threshold = st.slider(
            "🎯 Порог уверенности", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.05
        )
        
        iou_threshold = st.slider(
            "🔍 Порог IoU (фильтрация)", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.4, 
            step=0.05
        )
        
        show_confidence = st.checkbox("📊 Показывать уверенность", value=True)
        show_class_names = st.checkbox("🏷️ Показывать названия классов", value=True)
        
        st.markdown("---")
        st.markdown("### 📋 Поддерживаемые знаки:")
        
        for class_name in CLASS_NAMES:
            if "Speed Limit" in class_name:
                st.markdown(f"🔢 {class_name}")
            elif "Light" in class_name:
                st.markdown(f"🚦 {class_name}")
            elif "Stop" in class_name:
                st.markdown(f"🛑 {class_name}")
    
    # Основной интерфейс
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Загрузите изображение")
        
        # Способы загрузки
        upload_option = st.radio(
            "Выберите способ:",
            ["📁 Загрузить файл", "📷 Использовать камеру", "🖼️ Пример изображения"]
        )
        
        uploaded_image = None
        
        if upload_option == "📁 Загрузить файл":
            uploaded_file = st.file_uploader(
                "Выберите изображение",
                type=['png', 'jpg', 'jpeg'],
                help="Поддерживаются форматы: PNG, JPG, JPEG"
            )
            if uploaded_file:
                uploaded_image = Image.open(uploaded_file)
                
        elif upload_option == "📷 Использовать камеру":
            camera_image = st.camera_input("Сделайте фото")
            if camera_image:
                uploaded_image = Image.open(camera_image)
                
        elif upload_option == "🖼️ Пример изображения":
            demo_image = st.selectbox(
                "Выберите пример:",
                ["traffic_scene_1.jpg", "speed_limit_40.jpg", "stop_sign.jpg"]
            )
            # Здесь можно добавить загрузку примеров из папки assets
            st.info("💡 Добавьте примеры изображений в папку assets/demo_images/")
    
    with col2:
        st.header("🎯 Результаты детекции")
        
        if uploaded_image:
            # Показываем оригинальное изображение
            st.subheader("📷 Исходное изображение")
            st.image(uploaded_image, use_container_width=True)
            
            # Кнопка для запуска детекции
            if st.button("🚀 Начать детекцию", type="primary"):
                with st.spinner("🔍 Анализируем изображение..."):
                    
                    # Загружаем модель
                    detector = load_model()
                    
                    # Конвертируем изображение
                    image_array = np.array(uploaded_image)
                    
                    # Запускаем детекцию
                    results = detector.detect(
                        image_array,
                        conf_threshold=confidence_threshold,
                        iou_threshold=iou_threshold
                    )
                    
                    # Создаем изображение с результатами
                    result_image = create_result_image(
                        image_array, 
                        results,
                        show_confidence=show_confidence,
                        show_class_names=show_class_names
                    )
                    
                    # Показываем результат
                    st.markdown('<div class="detection-result">', unsafe_allow_html=True)
                    st.subheader("✨ Результат детекции")
                    st.image(result_image, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Статистика обнаруженных объектов
                    if results['detections']:
                        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                        st.subheader("📊 Статистика обнаружений")
                        
                        # Создаем таблицу результатов
                        detection_data = []
                        for detection in results['detections']:
                            detection_data.append({
                                "Знак": detection['class_name'],
                                "Уверенность": f"{detection['confidence']:.2%}",
                                "Координаты": f"({detection['bbox'][0]}, {detection['bbox'][1]})"
                            })
                        
                        st.dataframe(detection_data, use_container_width=True)
                        
                        # График распределения классов
                        chart = create_statistics_chart(results['detections'])
                        st.plotly_chart(chart, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Предупреждения для водителя
                        st.subheader("⚠️ Рекомендации водителю")
                        recommendations = generate_driver_recommendations(results['detections'])
                        for rec in recommendations:
                            if rec['type'] == 'warning':
                                st.warning(rec['message'])
                            elif rec['type'] == 'info':
                                st.info(rec['message'])
                            elif rec['type'] == 'success':
                                st.success(rec['message'])
                    
                    else:
                        st.warning("🔍 На изображении не обнаружено дорожных знаков")
        
        else:
            st.info("👆 Загрузите изображение для начала анализа")
    
    # Футер с информацией
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Точность модели", "95.9%", "mAP50")
    
    with col2:
        st.metric("⚡ Скорость обработки", "~3ms", "на изображение")
    
    with col3:
        st.metric("🔢 Классов знаков", "15", "типов")

def generate_driver_recommendations(detections):
    """Генерируем рекомендации для водителя на основе обнаруженных знаков"""
    recommendations = []
    
    for detection in detections:
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        if "Speed Limit" in class_name and confidence > 0.8:
            speed = class_name.split()[-1]
            recommendations.append({
                'type': 'warning',
                'message': f"🚗 Ограничение скорости: {speed} км/ч"
            })
        
        elif "Stop" in class_name and confidence > 0.8:
            recommendations.append({
                'type': 'warning', 
                'message': "🛑 Обязательная остановка!"
            })
        
        elif "Red Light" in class_name and confidence > 0.8:
            recommendations.append({
                'type': 'warning',
                'message': "🔴 Красный свет - остановитесь!"
            })
        
        elif "Green Light" in class_name and confidence > 0.8:
            recommendations.append({
                'type': 'success',
                'message': "🟢 Зеленый свет - можно продолжать движение"
            })
    
    if not recommendations:
        recommendations.append({
            'type': 'info',
            'message': "ℹ️ Дорожные знаки обнаружены, но рекомендации не требуются"
        })
    
    return recommendations

if __name__ == "__main__":
    main()