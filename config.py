from pathlib import Path

# Пути к файлам
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
CONFIG_PATH = PROJECT_ROOT / "models" / "traffic_signs.yaml"
DEMO_IMAGES_PATH = PROJECT_ROOT / "assets" / "demo_images"

# Классы дорожных знаков (как в обучении)
CLASS_NAMES = [
    'Green Light',
    'Red Light', 
    'Speed Limit 10',
    'Speed Limit 100',
    'Speed Limit 110',
    'Speed Limit 120',
    'Speed Limit 20',
    'Speed Limit 30',
    'Speed Limit 40',
    'Speed Limit 50',
    'Speed Limit 60',
    'Speed Limit 70',
    'Speed Limit 80',
    'Speed Limit 90',
    'Stop'
]

# Настройки по умолчанию
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU = 0.4
DEFAULT_IMAGE_SIZE = 640

# Цвета для классов (BGR формат для OpenCV)
CLASS_COLORS = {
    'Green Light': (0, 255, 0),
    'Red Light': (0, 0, 255),
    'Stop': (0, 0, 255),
    'Speed Limit 10': (255, 255, 0),
    'Speed Limit 20': (255, 255, 0),
    'Speed Limit 30': (255, 255, 0),
    'Speed Limit 40': (255, 255, 0),
    'Speed Limit 50': (255, 255, 0),
    'Speed Limit 60': (255, 255, 0),
    'Speed Limit 70': (255, 255, 0),
    'Speed Limit 80': (255, 255, 0),
    'Speed Limit 90': (255, 255, 0),
    'Speed Limit 100': (255, 255, 0),
    'Speed Limit 110': (255, 255, 0),
    'Speed Limit 120': (255, 255, 0),
}