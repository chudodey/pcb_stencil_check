"""
Структуры данных для всего проекта
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np


@dataclass
class EnsureDirResult:
    """Результат проверки/создания директории"""
    created: bool
    dir_path: Path


@dataclass
class ScanResult:
    """Результат сканирования и сравнения"""
    file: str
    time: str
    rotation: str
    shift_x: str
    shift_y: str
    match: str
    correlation: float
    status: str = "success"


@dataclass
class OrderData:
    """Данные для обработки одного заказа"""
    order_number: str
    gerber_path: Path
    gerber_image: Optional[np.ndarray] = None
    gerber_metrics: Optional[dict] = None
    gerber_contours: Optional[List[np.ndarray]] = None
    gerber_bounds_mm: Optional[tuple] = None
    scan_image_processed: Optional[np.ndarray] = None
    scan_path: Optional[Path] = None

    def __post_init__(self):
        # Гарантируем, что gerber_path всегда объект Path
        if isinstance(self.gerber_path, str):
            self.gerber_path = Path(self.gerber_path)

    scan_dpi: Optional[int] = None
    min_contour_area_mm2: Optional[float] = None
    max_contour_area_mm2: Optional[float] = None


@dataclass
class OrderLog:
    """Лог обработки заказа для сохранения в базу данных"""
    order_id: str
    gerber_file: str
    operator: str
    start_time: str
    end_time: str = ""
    scans: List[ScanResult] = field(default_factory=list)
    board_size: str = ""
    polygon_count: int = 0
    processing_errors: List[str] = field(default_factory=list)


@dataclass
class SessionData:
    """Данные рабочей сессии оператора"""
    start_time: str
    end_time: str = ""
    operator: str = ""
    orders: List[str] = field(default_factory=list)
    total_scans: int = 0
    successful_scans: int = 0


def calculate_alignment_status(correlation: float,
                               high_threshold: float,
                               medium_threshold: float) -> Tuple[str, str]:
    """
    Определяет статус совмещения на основе корреляции.

    Returns:
        Tuple[str, str]: (статус_ключ, статус_текст)
    """
    if correlation >= high_threshold:
        return "success", "УСПЕШНО"
    elif correlation >= medium_threshold:
        return "warning", "С ПРЕДУПРЕЖДЕНИЕМ"
    else:
        return "failed", "НЕУДАЧНО"
