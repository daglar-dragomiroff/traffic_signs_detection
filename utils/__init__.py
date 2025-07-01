"""
Модуль утилит для проекта Traffic Signs Detection
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# ========================================
# Инструкции по развертыванию
# ========================================

"""
ПОШАГОВАЯ ИНСТРУКЦИЯ ПО РАЗВЕРТЫВАНИЮ:

1. СОЗДАЙТЕ СТРУКТУРУ ПАПОК:
   mkdir traffic_signs_detection
   cd traffic_signs_detection
   mkdir models utils assets docs
   mkdir assets/demo_images assets/icons
   touch utils/__init__.py

2. СКОПИРУЙТЕ ФАЙЛЫ ИЗ COLAB:
   - best.pt -> models/
   - traffic_signs.yaml -> models/

3. СОЗДАЙТЕ ФАЙЛЫ С КОДОМ:
   - app.py (главное приложение)
   - config.py (настройки)
   - utils/detection.py (детекция)
   - utils/visualization.py (визуализация)
   - requirements.txt (зависимости)
   - README.md (документация)

4. УСТАНОВИТЕ ЗАВИСИМОСТИ:
   pip install -r requirements.txt

5. ЗАПУСТИТЕ ПРИЛОЖЕНИЕ:
   streamlit run app.py

6. ЗАПИСАТЬ ДЕМО-ВИДЕО:
   - Откройте приложение в браузере
   - Загрузите тестовые изображения
   - Покажите работу детекции
   - Запишите экран (~2-3 минуты)

7. ПОДГОТОВЬТЕ К СДАЧЕ:
   - Репозиторий на GitHub
   - Демо-видео в папке docs/
   - README с инструкциями
   - Все файлы проекта
"""