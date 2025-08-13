"""
1. GerberProcessor - парсинг Gerber-файлов и преобразование в изображение

Все классы реализуют единый интерфейс:
- Параметры обработки задаются при инициализации
- Основной метод process()/align() возвращает словарь с результатами
- Стандартизированная структура выходных данных
"""

import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import cv2
import numpy as np


@dataclass
class Point:
    """Точка с координатами в миллиметрах."""
    x: float
    y: float


@dataclass
class Aperture:
    """Апертура из Gerber-файла."""
    code: int
    type: str  # Тип: 'C' (круг), 'R' (прямоугольник), 'O' (обвод), 'P' (полигон)
    parameters: List[float]  # Параметры апертуры


class GerberProcessor:
    """
    Парсер Gerber-файлов с преобразованием в растровое изображение.

    Параметры:
        dpi (int): Разрешение выходного изображения (по умолчанию 600)
        margin_mm (float): Отступ от краев в мм (по умолчанию 2.0)

    Пример использования:
        processor = GerberProcessor(dpi=1200)
        result = processor.parse(gerber_content)
        cv2.imwrite('board.png', result['result_image'])
    """

    def __init__(self, dpi: int = 600, margin_mm: float = 2.0):
        self.params = {
            'dpi': dpi,
            'margin_mm': margin_mm,
            'units': 'MM'  # Единицы измерения по умолчанию
        }
        self._reset_state()

    def _reset_state(self) -> None:
        """Сброс состояния парсера перед обработкой нового файла."""
        self.contours: List[List[Point]] = []
        self.apertures: Dict[int, Aperture] = {}
        self.current_path: List[Point] = []
        self.current_pos = Point(0.0, 0.0)
        self.current_aperture: Optional[int] = None
        self.format_spec: Dict[str, Tuple[int, int]] = {'x': (3, 3), 'y': (3, 3)}

    def parse(self, content: str) -> Dict:
        """
        Парсинг содержимого Gerber-файла.

        Аргументы:
            content (str): Текст Gerber-файла

        Возвращает:
            Словарь с результатами:
            - result_image (np.ndarray): Растровое изображение платы
            - metrics (dict): Характеристики платы
            - params (dict): Параметры обработки
            - contours (list): Список контуров
            - apertures (dict): Определения апертур
            - debug_info (dict): Дополнительная информация
        """
        self._reset_state()
        
        # Обработка строк файла
        for line in (l.strip() for l in content.split('\n') if l.strip()):
            self._parse_line(line)

        # >>> РЕШЕНИЕ: Добавьте эту строку, чтобы сохранить последний контур
        self._finalize_path()

        # Расчет границ платы
        xmin, ymin, xmax, ymax = self._calculate_bounds()
        width_mm = max(0.0, xmax - xmin)
        height_mm = max(0.0, ymax - ymin)

        return {
            'result_image': self._rasterize(),
            'metrics': {
                'board_width_mm': round(width_mm, 3),
                'board_height_mm': round(height_mm, 3),
                'contour_count': len(self.contours),
                'aperture_count': len(self.apertures)
            },
            'params': self.params,
            'contours': self.contours,
            'apertures': self.apertures,
            'debug_info': {
                'bounds_mm': (xmin, ymin, xmax, ymax)
            }
        }

    def _rasterize(self) -> np.ndarray:
        """Преобразование контуров в растровое изображение."""
        if not self.contours:
            return np.zeros((100, 100), dtype=np.uint8)

        bounds = self._calculate_bounds()
        rasterizer = _GerberRasterizer(self.contours, bounds)
        return rasterizer.rasterize(self.params['dpi'], self.params['margin_mm'])

    def _calculate_bounds(self) -> Tuple[float, float, float, float]:
        """Вычисление границ платы по контурам."""
        if not self.contours:
            return (0.0, 0.0, 0.0, 0.0)
        
        all_points = [p for contour in self.contours for p in contour]
        xs = [p.x for p in all_points]
        ys = [p.y for p in all_points]
        return (min(xs), min(ys), max(xs), max(ys))

    def _parse_line(self, line: str) -> None:
        """Разбор одной строки Gerber-файла."""
        if line.startswith('G04'):
            return  # Пропуск комментариев

        if line.startswith('%') and line.endswith('%'):
            self._parse_extended_command(line[1:-1])
            return

        if self._is_aperture_select(line):
            d_index = line.index('D')
            num = ''.join(ch for ch in line[d_index + 1:] if ch.isdigit())
            if num:
                self.current_aperture = int(num)
            return

        if line in ('G01', 'G02', 'G03'):
            return  # Режим интерполяции пока не используется

        if any(c in line for c in ('X', 'Y', 'D')):
            self._parse_coordinate_command(line)

    def _parse_extended_command(self, cmd: str) -> None:
        """Разбор расширенной команды (между %%)."""
        if cmd.startswith('FS'):
            self._parse_format_spec(cmd)
        elif cmd.startswith('MO'):
            self.units = 'MM' if 'MM' in cmd else 'IN'
        elif cmd.startswith('ADD'):
            self._parse_aperture_def(cmd)
        elif cmd.startswith('LP'):
            pass  # Полярность пока не используется

    def _parse_format_spec(self, cmd: str) -> None:
        """Разбор спецификации формата (FS)."""
        match = re.search(r'X(\d)(\d)Y(\d)(\d)', cmd)
        if match:
            self.format_spec = {
                'x': (int(match.group(1)), int(match.group(2))),
                'y': (int(match.group(3)), int(match.group(4))),
            }

    def _parse_aperture_def(self, cmd: str) -> None:
        """Разбор определения апертуры (ADD)."""
        match = re.match(r'ADD(\d+)([CROP])(.*)', cmd)
        if match:
            params = [float(p) for p in re.findall(r'[\d.]+', match.group(3))]
            code = int(match.group(1))
            self.apertures[code] = Aperture(code=code, type=match.group(2), parameters=params)

    def _parse_coordinate_command(self, line: str) -> None:
        """Разбор команд с координатами."""
        x = self._extract_coord(line, 'X')
        y = self._extract_coord(line, 'Y')
        d = self._extract_d_code(line)

        self.current_pos = Point(
            x if x is not None else self.current_pos.x,
            y if y is not None else self.current_pos.y
        )

        if d == 1:  # Режим рисования (D01)
            self._add_path_point()
        elif d == 2:  # Режим перемещения (D02)
            self._finalize_path()
        elif d == 3:  # Вспышка апертуры (D03)
            self._finalize_path()
            self.contours.append([self.current_pos])
        elif x is not None or y is not None:  # ✅ ДОБАВИТЬ ЭТУ СТРОКУ!
            # Если есть координаты без D-кода, продолжаем рисование
            self._add_path_point()

    def _extract_coord(self, line: str, axis: str) -> Optional[float]:
        """Извлечение координаты из строки."""
        if axis not in line:
            return None
        
        m = re.search(fr'{axis}([+-]?\d+)', line)
        if not m:
            return None
        
        raw = int(m.group(1))
        decimals = self.format_spec[axis.lower()][1]
        return raw / (10 ** decimals)

    def _extract_d_code(self, line: str) -> Optional[int]:
        """Извлечение D-кода из строки."""
        m = re.search(r'D0?([123])', line)
        return int(m.group(1)) if m else None

    def _add_path_point(self) -> None:
        """Добавление точки в текущий контур."""
        if not self.current_path or self.current_pos != self.current_path[-1]:
            self.current_path.append(self.current_pos)

    def _finalize_path(self) -> None:
        """Завершение текущего контура."""
        if self.current_path:
            self.contours.append(self.current_path)
            self.current_path = []

    def _is_aperture_select(self, line: str) -> bool:
        """Проверка, является ли строка выбором апертуры."""
        if ('D' not in line) or any(c in line for c in ('X', 'Y')):
            return False
        
        stripped = line.replace('G54D', '').replace('D', '').strip('*')
        return stripped.isdigit()


class _GerberRasterizer:
    """Внутренний класс для преобразования контуров Gerber в изображение."""

    def __init__(self, contours: List[List[Point]], bounds: Tuple[float, float, float, float]):
        self.contours = contours
        self.bounds = bounds  # (xmin, ymin, xmax, ymax)

    def rasterize(self, dpi: int, margin_mm: float) -> np.ndarray:
        """Рендеринг контуров в бинарное изображение."""
        width_px, height_px = self._calculate_dimensions(dpi, margin_mm)
        image = np.zeros((height_px, width_px), dtype=np.uint8)

        for contour in self.contours:
            self._draw_contour(image, contour, dpi, margin_mm)

        return image

    def _calculate_dimensions(self, dpi: int, margin_mm: float) -> Tuple[int, int]:
        """Вычисление размеров изображения в пикселях."""
        xmin, ymin, xmax, ymax = self.bounds
        width_mm = max(0.0, xmax - xmin) + 2 * margin_mm
        height_mm = max(0.0, ymax - ymin) + 2 * margin_mm
        mm_to_px = dpi / 25.4
        return int(width_mm * mm_to_px), int(height_mm * mm_to_px)

    def _draw_contour(self, image: np.ndarray, contour: List[Point], dpi: int, margin_mm: float) -> None:
        """Отрисовка одного контура на изображении."""
        if not contour:
            return

        xmin, ymin, _, _ = self.bounds
        mm_to_px = dpi / 25.4

        pts: List[List[int]] = []
        for p in contour:
            px = int((p.x - xmin + margin_mm) * mm_to_px)
            py = image.shape[0] - int((p.y - ymin + margin_mm) * mm_to_px)
            pts.append([px, py])

        if len(pts) == 1:
            cv2.circle(image, tuple(pts[0]), 1, 255, -1)
        elif len(pts) == 2:
            cv2.line(image, tuple(pts[0]), tuple(pts[1]), 255, 1)
        else:
            cv2.fillPoly(image, [np.array(pts, dtype=np.int32)], 255)