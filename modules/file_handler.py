"""
Утилиты для работы с файлами, изображениями и данными
"""

import os
import yaml
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
from PIL import Image

# Импорт dataclasses из того же пакета
from .data_models import EnsureDirResult


class FileHandler:
    """Утилиты для работы с файлами, изображениями и данными."""

    @staticmethod
    def ensure_dir(path: Union[Path, str], as_dir: Optional[bool] = None) -> EnsureDirResult:
        """
        Гарантирует существование директории, связанной с указанным путём.

        Аргументы:
            path: Путь к директории или файлу
            as_dir: Явное указание типа пути (True=директория, False=файл, None=автоопределение)

        Возвращает:
            EnsureDirResult: результат создания или обнаружения директории
        """
        p = Path(path)
        if as_dir is True:
            dir_path = p
        elif as_dir is False:
            dir_path = p.parent
        else:
            # Эвристика: путь с расширением считается файлом
            s = str(p)
            dir_path = p if s.endswith(
                ("/", "\\")) or p.suffix == "" else p.parent

        existed = dir_path.exists()
        if not existed:
            dir_path.mkdir(parents=True, exist_ok=True)
        return EnsureDirResult(created=not existed, dir_path=dir_path)

    @staticmethod
    def read_text(path: Union[Path, str], encoding: str = 'utf-8') -> str:
        """Читает текстовый файл с обработкой ошибок кодировки."""
        return Path(path).read_text(encoding=encoding, errors='ignore')

    # ДОБАВЛЕНО: Новый универсальный метод для записи текста в файл
    @staticmethod
    def write_text(path: Union[Path, str], content: str, encoding: str = 'utf-8') -> None:
        """Записывает текстовые данные в файл, создавая директорию при необходимости."""
        FileHandler.ensure_dir(path, as_dir=False)
        Path(path).write_text(content, encoding=encoding)

    @staticmethod
    def read_image(path: Union[Path, str]) -> np.ndarray:
        """Загружает изображение и конвертирует в оттенки серого."""
        img = cv2.imread(os.fspath(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {path}")

        if img.ndim == 3:
            channels = img.shape[2]
            if channels == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def write_image(path: Union[Path, str], img: np.ndarray, quality: int = 95) -> None:
        """Сохраняет изображение с оптимизацией качества."""
        FileHandler.ensure_dir(path, as_dir=False)

        # Параметры сжатия в зависимости от формата
        ext = Path(path).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, 6]  # Умеренное сжатие
        else:
            params = []

        success = cv2.imwrite(os.fspath(path), img, params)
        if not success:
            raise IOError(f"Ошибка сохранения изображения: {path}")

    @staticmethod
    def get_image_info(path: Union[Path, str]) -> Dict:
        """Получает информацию об изображении включая DPI."""
        try:
            with Image.open(path) as img:
                width, height = img.size
                dpi = img.info.get('dpi', (72, 72))  # По умолчанию 72 DPI
                if isinstance(dpi, tuple):
                    dpi = int(dpi[0])  # Берем горизонтальное разрешение
                else:
                    dpi = int(dpi)

                return {
                    'width_pixels': width,
                    'height_pixels': height,
                    'dpi': dpi,
                    'width_mm': width * 25.4 / dpi,
                    'height_mm': height * 25.4 / dpi,
                    'has_dpi_metadata': 'dpi' in img.info
                }
        except Exception as e:
            # Fallback на OpenCV если PIL не может обработать
            img = cv2.imread(os.fspath(path))
            if img is None:
                raise ValueError(
                    f"Не удалось получить информацию об изображении: {path}") from e

            height, width = img.shape[:2]
            return {
                'width_pixels': width,
                'height_pixels': height,
                'dpi': 72,  # Значение по умолчанию
                'width_mm': width * 25.4 / 72,
                'height_mm': height * 25.4 / 72,
                'has_dpi_metadata': False
            }

    @staticmethod
    def find_gerber_files(search_dir: Path, order_number: str, rule: str = 'alphabetic_first') -> List[Path]:
        """Ищет Gerber-файлы с указанным номером заказа."""
        if not search_dir.exists():
            return []

        # Паттерн для поиска файлов, содержащих номер заказа
        pattern = f"*{order_number}*.gbr"
        found_files = list(search_dir.glob(pattern))

        if not found_files:
            return []

        # Применяем правило сортировки
        if rule == 'alphabetic_first':
            found_files.sort(key=lambda x: x.name)
        elif rule == 'newest':
            found_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        elif rule == 'oldest':
            found_files.sort(key=lambda x: x.stat().st_mtime)

        return found_files

    @staticmethod
    def validate_image_file(path: Path, max_size_mb: int = 0) -> Tuple[bool, str]:
        """Валидирует файл изображения."""
        if not path.exists():
            return False, "Файл не существует"

        # Проверка размера файла
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if max_size_mb > 0 and file_size_mb > max_size_mb:
            return False, f"Размер файла превышает {max_size_mb} МБ"

        # Проверка, что это действительно изображение
        try:
            with Image.open(path) as img:
                img.verify()  # Проверяет целостность файла
            return True, "OK"
        except Exception as e:
            return False, f"Некорректный файл изображения: {str(e)}"

    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Форматирует размер файла в человекочитаемом виде."""
        if size_bytes < 1024:
            return f"{size_bytes} байт"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.1f} КБ"
        elif size_bytes < 1024**3:
            return f"{size_bytes / 1024**2:.1f} МБ"
        else:
            return f"{size_bytes / 1024**3:.1f} ГБ"

    @staticmethod
    def read_yaml(path: Union[Path, str]) -> Dict:
        """Читает YAML файл с обработкой ошибок."""
        p = Path(path)
        if not p.exists():
            return {}
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Ошибка парсинга YAML: {p}: {e}") from e

    @staticmethod
    def write_yaml(path: Union[Path, str], data: Dict) -> None:
        """Записывает данные в YAML файл."""
        FileHandler.ensure_dir(path, as_dir=False)
        with open(Path(path), 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False,
                           default_flow_style=False, indent=2)

    @staticmethod
    def clean_filename(filename: str) -> str:
        """Очищает имя файла от недопустимых символов."""
        # Заменяем недопустимые символы на подчеркивание
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename

    @staticmethod
    def get_image_dpi(file_path: Path, default_dpi: int, dpi_priority: str) -> int:
        """Получение DPI изображения."""
        try:
            img = Image.open(file_path)
            dpi = img.info.get('dpi', (default_dpi, default_dpi))[0]
            return int(dpi) if dpi_priority == 'metadata' else default_dpi
        except Exception:
            return default_dpi

    @staticmethod
    def get_image_size(file_path: Path) -> Tuple[int, int]:
        """Получение размеров изображения в пикселях."""
        try:
            img = cv2.imdecode(np.fromfile(
                file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Не удалось определить размер изображения")
            return img.shape[1], img.shape[0]  # width, height
        except Exception:
            return 0, 0

    @staticmethod
    def pixels_to_mm(size_pixels: Tuple[int, int], dpi: int) -> Tuple[float, float]:
        """Перевод размеров из пикселей в миллиметры."""
        if dpi <= 0:
            return 0.0, 0.0
        mm_per_pixel = 25.4 / dpi
        return size_pixels[0] * mm_per_pixel, size_pixels[1] * mm_per_pixel

    @staticmethod
    def mm2_to_pixels(area_mm2: float, dpi: int) -> float:
        """
        Преобразует площадь из мм² в пиксели.

        Args:
            area_mm2: Площадь в мм²
            dpi: Разрешение в DPI

        Returns:
            float: Площадь в пикселях
        """
        if dpi <= 0:
            return max(area_mm2, 10.0)  # Минимум 10 пикселей

        if area_mm2 <= 0:
            # Если площадь нулевая или отрицательная, используем минимальное значение
            area_mm2 = 0.1  # 0.1 мм² как минимальная площадь

        # Преобразование: 1 дюйм = 25.4 мм
        # Площадь в пикселях = площадь в мм² * (dpi / 25.4)^2
        pixels_per_mm = dpi / 25.4
        result = area_mm2 * (pixels_per_mm ** 2)

        # Гарантируем минимальный размер контура
        return max(result, 10.0)  # Не менее 10 пикселей

    @staticmethod
    def wait_for_new_file(scan_folder: Path, supported_formats: List[str],
                          check_interval: float, timeout: float,
                          callback: Optional[callable] = None) -> Optional[Path]:
        """
        Ожидает появление нового файла в указанной папке.

        Args:
            scan_folder: Папка для мониторинга
            supported_formats: Поддерживаемые форматы файлов
            check_interval: Интервал проверки в секундах
            timeout: Таймаут ожидания в секундах
            callback: Функция обратного вызова для анимации (принимает timeout, elapsed)

        Returns:
            Path: Путь к новому файлу или None при таймауте
        """
        start_time = time.time()
        existing_files = FileHandler.get_existing_files(
            scan_folder, supported_formats)

        while True:
            current_time = time.time()
            elapsed = current_time - start_time

            # Вызов callback для анимации
            if callback:
                callback(timeout, elapsed)

            # Проверка таймаута
            if timeout > 0 and elapsed > timeout:
                return None

            # Проверяем новые файлы
            new_files = FileHandler.find_new_files(
                scan_folder, supported_formats, existing_files)
            if new_files:
                return FileHandler.get_newest_file(new_files)

            time.sleep(check_interval)

    @staticmethod
    def get_existing_files(scan_folder: Path, supported_formats: List[str]) -> Set[Path]:
        """Возвращает множество существующих файлов."""
        existing_files = set()
        for ext in supported_formats:
            pattern = f"*{ext}"
            existing_files.update(
                f for f in scan_folder.glob(pattern) if f.is_file()
            )
        return existing_files

    @staticmethod
    def find_new_files(scan_folder: Path, supported_formats: List[str],
                       existing_files: Set[Path]) -> List[Path]:
        """Находит новые файлы, которых не было в existing_files."""
        new_files = []
        for ext in supported_formats:
            pattern = f"*{ext}"
            for file_path in scan_folder.glob(pattern):
                if file_path.is_file() and file_path not in existing_files:
                    if FileHandler.is_file_fully_written(file_path):
                        new_files.append(file_path)
        return new_files

    @staticmethod
    def get_newest_file(files: List[Path]) -> Path:
        """Возвращает самый новый файл по дате изменения."""
        return max(files, key=lambda x: x.stat().st_mtime)

    @staticmethod
    def is_file_fully_written(file_path: Path) -> bool:
        """Проверяет, что файл полностью записан."""
        try:
            initial_size = file_path.stat().st_size
            time.sleep(0.1)
            return initial_size == file_path.stat().st_size
        except (OSError, FileNotFoundError):
            return False

    @staticmethod
    def get_last_existing_file(scan_folder: Path, supported_formats: List[str]) -> Optional[Path]:
        """Возвращает последний существующий файл в папке."""
        all_files = []
        for ext in supported_formats:
            pattern = f"*{ext}"
            all_files.extend(scan_folder.glob(pattern))

        if not all_files:
            return None

        # Фильтруем только файлы и берем самый новый
        files = [f for f in all_files if f.is_file()]
        if not files:
            return None

        return max(files, key=lambda x: x.stat().st_mtime)
