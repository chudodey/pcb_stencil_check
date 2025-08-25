import math
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Union
import cv2
import numpy as np


class ImageProcessor:
    """
    Предварительная обработка изображений PCB.

    Параметры:
        dpi (int): Разрешение изображения для расчетов в мм (по умолчанию 600)
        margin_mm (float): Отступ при обрезке в мм (по умолчанию 2.0)
        binary_threshold (int): Порог бинаризации (0 для Отсу) (по умолчанию 0)
        crop_min_area (float): Мин. площадь контура для учета (по умолчанию 50.0)
        crop_max_area_ratio (float): Макс. отношение площади контура к изображению (по умолчанию 0.5)

    Пример использования:
        processor = ImageProcessor(dpi=600)
        result = processor.process(scan_image)
        cv2.imwrite('processed.png', result['result_image'])
    """

    def __init__(self,
                 dpi: int = 600,
                 margin_mm: float = 2.0,
                 binary_threshold: int = 0,
                 crop_min_area: float = 50.0,
                 crop_max_area_ratio: float = 0.5):
        self.params = {
            'dpi': dpi,
            'margin_mm': margin_mm,
            'binary_threshold': binary_threshold,
            'crop_min_area': crop_min_area,
            'crop_max_area_ratio': crop_max_area_ratio
        }

    def process(self, image: np.ndarray, is_grayscale: bool = False) -> Dict:
        """
        Обработка изображения через полный конвейер.

        Аргументы:
            image (np.ndarray): Входное изображение
            is_grayscale (bool): Если True, пропускает преобразование в градации серого

        Возвращает:
            Словарь с результатами:
            - result_image (np.ndarray): Обработанное бинарное изображение
            - source_image (np.ndarray): Исходное изображение
            - metrics (dict): Метрики обработки
            - params (dict): Параметры обработки
            - debug_info (dict): Промежуточные данные
        """
        gray = image if is_grayscale else self._to_grayscale(image)
        binary = self._binarize(gray)
        cropped = self._crop_to_content(binary)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c)
                 for c in contours if cv2.contourArea(c) > 10]

        return {
            'result_image': cropped,
            'source_image': image,
            'metrics': {
                'original_size': f"{image.shape[1]}x{image.shape[0]}",
                'processed_size': f"{cropped.shape[1]}x{cropped.shape[0]}",
                'contour_count': len(contours),
                'mean_contour_area': np.mean(areas) if areas else 0,
                'crop_ratio': (cropped.shape[0] * cropped.shape[1]) / (image.shape[0] * image.shape[1])
            },
            'params': self.params,
            'debug_info': {
                'binary_image': binary,
                'all_contours': contours
            }
        }

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Преобразование в градации серого."""
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """Бинаризация изображения."""
        thresh = self.params['binary_threshold']
        method = cv2.THRESH_BINARY | (cv2.THRESH_OTSU if thresh <= 0 else 0)
        _, binary = cv2.threshold(image, max(0, thresh), 255, method)
        return binary

    def _crop_to_content(self, binary: np.ndarray) -> np.ndarray:
        """Обрезка изображения по значимым контурам."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary

        padding_px = int(self.params['margin_mm']
                         * (self.params['dpi'] / 25.4))
        img_area = binary.shape[0] * binary.shape[1]
        max_area = img_area * self.params['crop_max_area_ratio']

        valid = [c for c in contours
                 if self.params['crop_min_area'] < cv2.contourArea(c) < max_area]
        use_contours = valid if valid else contours

        x, y, w, h = cv2.boundingRect(np.vstack(use_contours))
        x1 = max(0, x - padding_px)
        y1 = max(0, y - padding_px)
        x2 = min(binary.shape[1], x + w + padding_px)
        y2 = min(binary.shape[0], y + h + padding_px)

        return binary[y1:y2, x1:x2]

    def validate_preprocessing(self, processed_image: np.ndarray,
                               gerber_metrics: Dict) -> Dict:
        """Валидация результатов предобработки относительно Gerber."""
        # Проверка количества контуров
        contours, _ = cv2.findContours(
            processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        validation_result = {
            'contour_count': len(contours),
            'contour_count_ok': len(contours) >= gerber_metrics.get('contour_count', 0) * 0.5,
            'size_match_ok': False,
            'rotation_detected': False
        }

        return validation_result
