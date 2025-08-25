"""
Управление конфигурацией приложения
"""

import configparser
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .file_handler import FileHandler

# Константа DEFAULT_CONFIG
DEFAULT_CONFIG = """[GENERAL]
default_operator_name = Иванов И.И.
order_number_digits = 6
debug_mode = false

[PATHS]
gerber_folder = ./gerber_files
scan_folder = ./scans
output_folder = ./results
create_order_subfolder = true

[SCANNING]
scan_instruction = Установите разрешение 1200 dpi, проверьте ориентацию, совместите трафарет по маркерам на столе. Нажмите Enter после завершения сканирования.
file_check_interval = 3
default_dpi = 600
dpi_priority = metadata
supported_image_formats = .png,.jpg,.jpeg,.tiff,.tif,.bmp

[GERBER_SEARCH]
multiple_files_rule = alphabetic_first

[IMAGE_PREPROCESSING]
size_tolerance_percent = 5.0
partial_preprocessing_action = ignore
min_contour_size = 10
binarization_type = otsu
threshold_value = 127
adaptive_block_size = 11
adaptive_c = 2

[IMAGE_COMPARISON]
consider_reflection = false
rotation_angles = 0,90,180,270
min_contour_coefficient = 2.0
ransac_reprojection_threshold = 3.0
max_iterations = 2000
confidence = 0.99
refine_iterations = 10
low_correlation_threshold = 0.2
medium_correlation_threshold = 0.4
high_correlation_threshold = 0.8

[VISUALIZATION]
reference_color_r = 255
reference_color_g = 0
reference_color_b = 0
scan_color_r = 0
scan_color_g = 255
scan_color_b = 255
intersection_color_r = 255
intersection_color_g = 255
intersection_color_b = 255
info_font_size = 20
info_text_color_r = 255
info_text_color_g = 255
info_text_color_b = 255
info_background_color_r = 0
info_background_color_g = 0
info_background_color_b = 0

[OUTPUT]
save_intermediate_images = false
save_final_image = true
existing_files_action = increment
gerber_image_filename = {order_number}_1_gerber.png
original_scan_filename = {order_number}_2_scan.png
processed_scan_filename = {order_number}_3_scan_prep.png
comparison_result_filename = {order_number}_4_compared.png
detailed_log_filename = {order_number}_5_log_detailed.txt
short_log_filename = {order_number}_6_log_short.txt

[LOGGING]
log_level = short
datetime_format = %Y-%m-%d %H:%M:%S
include_debug_info = false

[PYTHON_REQUIREMENTS]
min_python_version = 3.8
required_modules = cv2,numpy,PIL,yaml,scipy
install_command = pip install opencv-python numpy Pillow PyYAML scipy

[SYSTEM]
text_encoding = utf-8
max_scan_file_size = 100
scan_wait_timeout = 300
cleanup_temp_files = true

[ADVANCED]
use_multithreading = false
thread_count = 0
png_compression_level = 6
jpeg_quality = 95
max_processing_resolution = 4096
enable_profiling = false
"""


class ConfigManager:
    """Централизованное управление конфигурацией приложения."""

    def __init__(self, config_file: str = 'config.ini', debug_mode: bool = False, order_number: Optional[str] = None) -> None:
        self.config_file = Path(config_file)
        self.debug_mode = debug_mode
        self.preset_order_number = order_number
        self._config = configparser.ConfigParser()
        self._load_config()

    def _load_config(self) -> None:
        """Загружает конфигурацию с приоритетами: файл -> параметры командной строки."""
        if not self.config_file.exists():
            self._create_default_config()

        self._config.read(self.config_file, encoding='utf-8')

        # Переопределение режима отладки из параметров запуска
        if self.debug_mode:
            self._config.set('GENERAL', 'debug_mode', 'true')

        # Общие настройки
        self.default_operator_name = self._config.get(
            'GENERAL', 'default_operator_name', fallback='Иванов И.И.')
        self.order_number_digits = self._config.getint(
            'GENERAL', 'order_number_digits', fallback=6)
        self.debug_mode = self._config.getboolean(
            'GENERAL', 'debug_mode', fallback=False)

        # Пути
        self.gerber_folder = self._get_path('PATHS', 'gerber_folder')
        self.scan_folder = self._get_path('PATHS', 'scan_folder')
        self.output_folder = self._get_path('PATHS', 'output_folder')
        self.create_order_subfolder = self._config.getboolean(
            'PATHS', 'create_order_subfolder', fallback=True)

        # Сканирование
        self.scan_instruction = self._config.get(
            'SCANNING', 'scan_instruction', fallback='')
        self.file_check_interval = self._config.getfloat(
            'SCANNING', 'file_check_interval', fallback=3.0)
        self.default_dpi = self._config.getint(
            'SCANNING', 'default_dpi', fallback=600)
        self.dpi_priority = self._config.get(
            'SCANNING', 'dpi_priority', fallback='metadata')
        self.supported_image_formats = [fmt.strip() for fmt in self._config.get(
            'SCANNING', 'supported_image_formats', fallback='.png,.jpg,.jpeg,.tiff,.tif,.bmp').split(',')]

        # Поиск Gerber-файлов
        self.multiple_files_rule = self._config.get(
            'GERBER_SEARCH', 'multiple_files_rule', fallback='alphabetic_first')

        # Отступы в мм
        self.gerber_margin_mm = self._config.getfloat(
            'GERBER_PROCESSING', 'margin_mm', fallback=2.0)

        # Предобработка изображений
        self.size_tolerance_percent = self._config.getfloat(
            'IMAGE_PREPROCESSING', 'size_tolerance_percent', fallback=5.0)
        self.partial_preprocessing_action = self._config.get(
            'IMAGE_PREPROCESSING', 'partial_preprocessing_action', fallback='ignore')

        # Сравнение изображений - добавляем недостающие атрибуты
        self.consider_reflection = self._config.getboolean(
            'IMAGE_COMPARISON', 'consider_reflection', fallback=False)
        self.rotation_angles = [int(angle.strip()) for angle in self._config.get(
            'IMAGE_COMPARISON', 'rotation_angles', fallback='0,90,180,270').split(',')]
        self.min_contour_coefficient = self._config.getfloat(
            'IMAGE_COMPARISON', 'min_contour_coefficient', fallback=2.0)

        # Критически важные параметры RANSAC для AlignmentEngine
        self.ransac_reprojection_threshold = self._config.getfloat(
            'IMAGE_COMPARISON', 'ransac_reprojection_threshold', fallback=3.0)
        self.max_iterations = self._config.getint(
            'IMAGE_COMPARISON', 'max_iterations', fallback=2000)
        self.confidence = self._config.getfloat(
            'IMAGE_COMPARISON', 'confidence', fallback=0.99)
        self.refine_iterations = self._config.getint(
            'IMAGE_COMPARISON', 'refine_iterations', fallback=10)

        # Пороги корреляции
        self.low_correlation_threshold = self._config.getfloat(
            'IMAGE_COMPARISON', 'low_correlation_threshold', fallback=0.2)
        self.medium_correlation_threshold = self._config.getfloat(
            'IMAGE_COMPARISON', 'medium_correlation_threshold', fallback=0.4)
        self.high_correlation_threshold = self._config.getfloat(
            'IMAGE_COMPARISON', 'high_correlation_threshold', fallback=0.8)

        # Визуализация
        self.reference_color = (
            self._config.getint(
                'VISUALIZATION', 'reference_color_r', fallback=255),
            self._config.getint(
                'VISUALIZATION', 'reference_color_g', fallback=0),
            self._config.getint(
                'VISUALIZATION', 'reference_color_b', fallback=0)
        )
        self.scan_color = (
            self._config.getint('VISUALIZATION', 'scan_color_r', fallback=0),
            self._config.getint('VISUALIZATION', 'scan_color_g', fallback=255),
            self._config.getint('VISUALIZATION', 'scan_color_b', fallback=255)
        )
        self.intersection_color = (
            self._config.getint(
                'VISUALIZATION', 'intersection_color_r', fallback=255),
            self._config.getint(
                'VISUALIZATION', 'intersection_color_g', fallback=255),
            self._config.getint(
                'VISUALIZATION', 'intersection_color_b', fallback=255)
        )

        self.info_font_size = self._config.getint(
            'VISUALIZATION', 'info_font_size', fallback=20)

        self.info_text_color = (
            self._config.getint(
                'VISUALIZATION', 'info_text_color_r', fallback=255),
            self._config.getint(
                'VISUALIZATION', 'info_text_color_g', fallback=255),
            self._config.getint(
                'VISUALIZATION', 'info_text_color_b', fallback=255)
        )
        self.info_background_color = (
            self._config.getint(
                'VISUALIZATION', 'info_background_color_r', fallback=0),
            self._config.getint(
                'VISUALIZATION', 'info_background_color_g', fallback=0),
            self._config.getint(
                'VISUALIZATION', 'info_background_color_b', fallback=0)
        )

        # Вывод результатов
        self.save_intermediate_images = self._config.getboolean(
            'OUTPUT', 'save_intermediate_images', fallback=False) or self.debug_mode
        self.save_final_image = self._config.getboolean(
            'OUTPUT', 'save_final_image', fallback=True)
        self.existing_files_action = self._config.get(
            'OUTPUT', 'existing_files_action', fallback='increment')

        # Шаблоны имен файлов
        self.gerber_image_filename = self._config.get(
            'OUTPUT', 'gerber_image_filename', fallback='{order_number}_1_gerber.png')
        self.original_scan_filename = self._config.get(
            'OUTPUT', 'original_scan_filename', fallback='{order_number}_2_scan.png')
        self.processed_scan_filename = self._config.get(
            'OUTPUT', 'processed_scan_filename', fallback='{order_number}_3_scan_prep.png')
        self.comparison_result_filename = self._config.get(
            'OUTPUT', 'comparison_result_filename', fallback='{order_number}_4_compared.png')
        self.detailed_log_filename = self._config.get(
            'OUTPUT', 'detailed_log_filename', fallback='{order_number}_5_log_detailed.txt')
        self.short_log_filename = self._config.get(
            'OUTPUT', 'short_log_filename', fallback='{order_number}_6_log_short.txt')

        # Логирование
        self.log_level = self._config.get(
            'LOGGING', 'log_level', fallback='short')
        self.datetime_format = self._config.get(
            'LOGGING', 'datetime_format', fallback='%Y-%m-%d %H:%M:%S')
        self.short_log_template = self._config.get(
            'LOGGING', 'short_log_template', fallback='Дата и время: {datetime}\nОператор: {operator_name}\nНомер заказа: {order_number}\nРезультат корреляции: {correlation_result:.3f}\nСтатус совмещения: {alignment_status}')
        self.include_debug_info = self._config.getboolean(
            'LOGGING', 'include_debug_info', fallback=False) or self.debug_mode

        # Системные параметры
        self.max_scan_file_size = self._config.getint(
            'SYSTEM', 'max_scan_file_size', fallback=100)
        self.scan_wait_timeout = self._config.getint(
            'SYSTEM', 'scan_wait_timeout', fallback=300)

        # Требования Python
        self.min_python_version = self._config.get(
            'PYTHON_REQUIREMENTS', 'min_python_version', fallback='3.8')
        self.required_modules = [mod.strip() for mod in self._config.get(
            'PYTHON_REQUIREMENTS', 'required_modules', fallback='').split(',') if mod.strip()]
        self.install_command = self._config.get(
            'PYTHON_REQUIREMENTS', 'install_command', fallback='pip install opencv-python numpy Pillow matplotlib scipy')

        self._validate_config()

    def _get_path(self, section: str, option: str, create_dir: bool = False) -> Path:
        """Получает путь из конфигурации и при необходимости создает директорию."""
        path = Path(self._config.get(section, option))
        if create_dir:
            FileHandler.ensure_dir(path, as_dir=True)
        return path

    def _validate_config(self) -> None:
        """Валидация параметров конфигурации."""
        if self.order_number_digits <= 0:
            raise ValueError(
                "Количество цифр в номере заказа должно быть положительным")
        if self.file_check_interval <= 0:
            raise ValueError(
                "Интервал проверки файлов должен быть положительным")
        if not 0 <= self.low_correlation_threshold <= self.medium_correlation_threshold <= self.high_correlation_threshold <= 1:
            raise ValueError(
                "Пороги корреляции должны быть в диапазоне [0,1] и возрастающими")

    def _create_default_config(self) -> None:
        """Создает файл конфигурации с базовыми настройками."""
        FileHandler.ensure_dir(self.config_file, as_dir=False)
        self.config_file.write_text(DEFAULT_CONFIG, encoding='utf-8')

    def check_python_environment(self) -> List[str]:
        """Проверяет версию Python и наличие необходимых модулей."""
        issues = []

        # Маппинг имен модулей для проверки -> имен пакетов для установки
        module_mapping = {
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'yaml': 'PyYAML'
        }

        # Проверка версии Python
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if sys.version_info < tuple(map(int, self.min_python_version.split('.'))):
            issues.append(
                f"Требуется Python {self.min_python_version}+, установлен {current_version}")

        # Проверка модулей
        missing_modules = []
        for module in self.required_modules:
            module_name = module.split('.')[0]
            if importlib.util.find_spec(module_name) is None:
                package_name = module_mapping.get(module_name, module_name)
                missing_modules.append(package_name)

        if missing_modules:
            issues.append(f"Отсутствуют модули: {', '.join(missing_modules)}")
            issues.append(f"Команда для установки: {self.install_command}")

        return issues

    def check_directories(self) -> Dict[str, bool]:
        """Проверяет существование необходимых директорий."""
        directories = {
            'gerber_folder': self.gerber_folder.exists(),
            'scan_folder': self.scan_folder.exists(),
            'output_folder': self.output_folder.exists()
        }
        return directories

    def create_missing_directories(self) -> List[Path]:
        """Создает отсутствующие директории."""
        created = []
        for path in [self.gerber_folder, self.scan_folder, self.output_folder]:
            if not path.exists():
                FileHandler.ensure_dir(path, as_dir=True)
                created.append(path)
        return created

    def create_order_workspace(self, order_number: str) -> Path:
        """Создает рабочую папку для заказа."""
        if self.create_order_subfolder:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            workspace = self.output_folder / f"{order_number}_{timestamp}"
        else:
            workspace = self.output_folder

        FileHandler.ensure_dir(workspace, as_dir=True)
        return workspace

    def generate_example_order_number(self) -> str:
        """Генерирует пример номера заказа для подсказки пользователю."""
        import random
        return ''.join([str(random.randint(0, 9)) for _ in range(self.order_number_digits)])

    def get_filename(self, template: str, order_number: str, workspace: Path) -> Path:
        """Генерирует имя файла на основе шаблона и обрабатывает конфликты имен."""
        filename = template.format(order_number=order_number)
        filepath = workspace / filename

        if self.existing_files_action == 'increment' and filepath.exists():
            counter = 1
            name, ext = filepath.stem, filepath.suffix
            while True:
                new_name = f"{name}_{counter}{ext}"
                new_filepath = workspace / new_name
                if not new_filepath.exists():
                    return new_filepath
                counter += 1

        return filepath
