"""
Единый модуль утилит и пользовательского интерфейса.

Содержит:
- ConfigManager — управление конфигурацией приложения
- SessionLogger — логирование сессий и результатов обработки
- UserInterface — командный интерфейс для взаимодействия с оператором
- FileHandler — работа с файлами, изображениями и YAML
- ErrorHandler — централизованная обработка ошибок

Автор: Арабов Далер Искандарович
"""

import configparser
import yaml
import numpy as np
import cv2
import re
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Union, Optional
from dataclasses import dataclass, field, asdict

# =========================
# Константы
# =========================
DEFAULT_CONFIG = """[paths]
source_dir = ./test_data/gerber
scan_dir = ./test_data/scans
output_dir = ./results
log_dir = ./logs
search_in_subdirs = false

[processing]
dpi = 1200
margin_mm = 1.0
binary_threshold = 200
crop_min_area = 10
crop_max_area_ratio = 0.1
alignment_accuracy = 0.95
order_processing = overwrite

[logging]
enable_global_log = true
default_operator = Иванов И.И.
"""

# =========================
# Конфигурация
# =========================
@dataclass(frozen=True)
class EnsureDirResult:
    created: bool
    dir_path: Path

class ConfigManager:
    """Централизованное управление конфигурацией приложения."""

    def __init__(self, config_file: str = 'config.txt') -> None:
        self.config_file = Path(config_file)
        self._load_config()

    def _load_config(self) -> None:
        if not self.config_file.exists():
            self._create_default_config()

        config = configparser.ConfigParser()
        config.read(self.config_file, encoding='utf-8')

        # Пути
        self.source_dir = self._get_path(config, 'paths', 'source_dir')
        self.scan_dir = self._get_path(config, 'paths', 'scan_dir')
        self.output_dir = self._get_path(config, 'paths', 'output_dir', create_dir=True)
        self.log_dir = self._get_path(config, 'paths', 'log_dir', create_dir=True)
        self.search_in_subdirs = config.getboolean('paths', 'search_in_subdirs', fallback=False)

        # Параметры обработки
        self.dpi = config.getint('processing', 'dpi', fallback=1200)
        self.margin_mm = config.getfloat('processing', 'margin_mm', fallback=1.0)
        self.binary_threshold = config.getint('processing', 'binary_threshold', fallback=200)
        self.crop_min_area = config.getint('processing', 'crop_min_area', fallback=10)
        self.crop_max_area_ratio = config.getfloat('processing', 'crop_max_area_ratio', fallback=0.1)
        self.alignment_accuracy = config.getfloat('processing', 'alignment_accuracy', fallback=0.95)
        self.order_processing = config.get('processing', 'order_processing', fallback='overwrite')

        # Логирование
        self.enable_global_log = config.getboolean('logging', 'enable_global_log', fallback=True)
        self.default_operator = config.get('logging', 'default_operator', fallback='Неизвестный оператор')

        self._validate_config()

    def _get_path(self, config: configparser.ConfigParser, section: str, option: str, create_dir: bool = False) -> Path:
        path = Path(config.get(section, option))
        if create_dir:
            FileHandler.ensure_dir(path, as_dir=True)
        return path

    def _validate_config(self) -> None:
        if self.dpi <= 0:
            raise ValueError("DPI должно быть положительным")
        if not 0 <= self.alignment_accuracy <= 1:
            raise ValueError("Точность совмещения должна быть от 0 до 1")
        if not self.source_dir.exists():
            raise ValueError(f"Папка с исходными файлами не найдена: {self.source_dir}")

    def _create_default_config(self) -> None:
        FileHandler.ensure_dir(self.config_file, as_dir=False)  # создаст родительскую директорию
        self.config_file.write_text(DEFAULT_CONFIG, encoding='utf-8')

    def create_order_workspace(self, order_number: str) -> Path:
        workspace = self.output_dir / order_number
        FileHandler.ensure_dir(workspace, as_dir=True)
        return workspace

    def get_order_processing_mode(self) -> str:
        return self.order_processing

# =========================
# Логирование
# =========================
@dataclass
class ScanResult:
    file: str
    time: str
    rotation: str
    shift_x: str
    shift_y: str
    match: str
    status: str = "success"

@dataclass
class OrderLog:
    order_id: str
    gerber_file: str
    scans: List[ScanResult] = field(default_factory=list)

@dataclass
class SessionData:
    start_time: str
    end_time: str = ""
    operator: str = ""
    orders: List[str] = field(default_factory=list)

class SessionLogger:
    """Логирование работы оператора."""

    def __init__(self, config_manager: ConfigManager):
        self._config = config_manager
        self._session: Optional[SessionData] = None
        self._current_operator: Optional[str] = None

    def start_session(self) -> None:
        self._session = SessionData(
            start_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            operator=self._current_operator or self._config.default_operator
        )

    def set_operator(self, name: str) -> None:
        self._current_operator = name
        if self._session:
            self._session.operator = name

    def log_scan_result(self, order_id: str, scan_path: Union[str, Path], alignment_result: Dict[str, float]) -> None:
        if not self._session:
            return
        if order_id not in self._session.orders:
            self._session.orders.append(order_id)
        order_log = self._load_order_log(order_id)
        spath = Path(scan_path)
        order_log.scans.append(ScanResult(
            file=spath.name,
            time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            rotation=f"{alignment_result['rotation']:.2f}°",
            shift_x=f"{alignment_result['shift_x']:.3f}mm",
            shift_y=f"{alignment_result['shift_y']:.3f}mm",
            match=f"{alignment_result['match_percentage']:.1f}%"
        ))
        self._save_order_log(order_id, order_log)

    def end_session(self) -> None:
        if not self._session:
            return
        self._session.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file = self._config.log_dir / f"session_{self._session.start_time.replace(':', '-')}.yaml"
        FileHandler.write_yaml(log_file, asdict(self._session))

    def _load_order_log(self, order_id: str) -> OrderLog:
        log_file = self._config.output_dir / order_id / "order_log.yaml"
        data = FileHandler.read_yaml(log_file)
        if data:
            return OrderLog(
                order_id=data.get('order_id', order_id),
                gerber_file=data.get('gerber_file', f"{order_id}.gbr"),
                scans=[ScanResult(**scan) for scan in data.get('scans', [])]
            )
        return OrderLog(order_id=order_id, gerber_file=f"{order_id}.gbr")

    def _save_order_log(self, order_id: str, log_data: OrderLog) -> None:
        log_path = self._config.output_dir / order_id / "order_log.yaml"
        FileHandler.write_yaml(log_path, asdict(log_data))

class UserInterface:
    """Текстовый пользовательский интерфейс."""

    def __init__(self):
        self._templates = {
            'header': "=" * 60 + "\n  Система контроля качества трафаретов v1.0\n  Автор: Арабов Далер Искандарович\n" + "=" * 60,
            'scan_instructions': "\n📷 ИНСТРУКЦИЯ ПО СКАНИРОВАНИЮ:\n1. Поместите трафарет в сканер\n2. Убедитесь в правильном позиционировании\n3. Выполните сканирование\n4. Нажмите Enter после завершения",
            'result_divider': "─" * 50
        }

        self._menus = {
            'file_not_found': ("Файл не обнаружен", {"1": "Повторить ввод номера заказа", "2": "Выйти"}),
            'parsing_error': ("Ошибка парсинга", {"1": "Повторить парсинг", "2": "Новый заказ", "3": "Выйти"}),
            'next_action': ("Выберите действие", {"1": "Повторить сканирование", "2": "Перезапустить Gerber", "3": "Новый заказ", "4": "Выйти"})
        }

        # Новое: универсальные префиксы и флаг отладки
        self._prefixes = {
            "info": "",
            "success": "✅ ",
            "warning": "⚠️ ",
            "error": "❌ ",
            "debug": "[DEBUG] ",
        }
        self._debug_enabled = False

    # ----- Новые универсальные сообщения -----

    def set_debug(self, enabled: bool) -> None:
        """Включить/выключить вывод отладочных сообщений."""
        self._debug_enabled = enabled

    def show_message(self, text: str, kind: str = "info") -> None:
        prefix = self._prefixes.get(kind, "")
        print(f"{prefix}{text}")

    def show_success(self, text: str) -> None:
        self.show_message(text, kind="success")

    def show_warning(self, text: str) -> None:
        self.show_message(text, kind="warning")

    def show_error(self, text: str) -> None:
        self.show_message(text, kind="error")

    def show_debug(self, text: str) -> None:
        if self._debug_enabled:
            self.show_message(text, kind="debug")

    # ----- Существующая функциональность -----

    def _show(self, message: str, prefix: str = "") -> None:
        print(f"{prefix}{message}")

    def show_header(self):
        self._show(self._templates['header'])

    def show_scanning_instructions(self):
        self._show(self._templates['scan_instructions'])

    def show_parsing_stats(self, stats: Dict):
        lines = [
            "\n✓ Gerber-файл успешно обработан:",
            f"  Размер платы: {stats['board_width']}×{stats['board_height']} мм",
            f"  Найдено полигонов: {stats['polygon_count']}",
            f"  Время обработки: {stats['processing_time']:.2f} сек"
        ]
        print("\n".join(lines))

    def show_alignment_results(self, result: Dict):
        quality_map = [
            (98.0, "ОТЛИЧНОЕ 🟢", "✅"),
            (95.0, "ХОРОШЕЕ 🟡", "✅"),
            (90.0, "УДОВЛЕТВОРИТЕЛЬНОЕ 🟠", "⚠️"),
            (0.0, "НИЗКОЕ 🔴", "❌")
        ]
        match_pct = result['match_percentage']
        quality = next((q for t, q, _ in quality_map if match_pct >= t), "НЕИЗВЕСТНО")
        status = next((s for t, _, s in quality_map if match_pct >= t), "❓")
        lines = [
            self._templates['result_divider'],
            "📊 РЕЗУЛЬТАТЫ СОВМЕЩЕНИЯ",
            self._templates['result_divider'],
            f"🎯 Совпадение:     {match_pct:.1f}%",
            f"📊 Корреляция:     {result.get('correlation_peak', 0):.3f}",
            f"📐 Поворот:        {result['rotation']:+.2f}°",
            f"↔️  Сдвиг X:        {result['shift_x']:+.3f} мм",
            f"↕️  Сдвиг Y:        {result['shift_y']:+.3f} мм",
            f"📈 Качество:       {quality}",
            f"📋 Статус:         {status} Трафарет {'прошел' if match_pct >= 90.0 else 'не прошел'} контроль",
            self._templates['result_divider']
        ]
        print("\n".join(lines))

    def get_operator_name(self, default: str) -> str:
        pattern = re.compile(r"^[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-\.\s]{1,}$")
        while True:
            name = input(f'Введите ваше ФИО (или Enter для "{default}"):\n> ').strip()
            if name == "":
                return default
            if pattern.fullmatch(name):
                # Нормализуем лишние пробелы
                return re.sub(r"\s+", " ", name)
            self._show("ФИО может содержать буквы, пробелы, дефис и точку", "❌ ")

    def get_order_number(self) -> Optional[str]:
        while True:
            value = input("\nВведите 6-значный номер заказа или 'exit':\n> ").strip()
            if value.lower() == 'exit':
                return None
            if re.fullmatch(r'\d{6}', value):
                return value
            self._show("Номер должен содержать ровно 6 цифр", "❌ ")

    def wait_for_scan_completion(self):
        input("Нажмите Enter после сканирования...\n> ")

    def show_menu(self, menu_key: str, error_message: str = "") -> int:
        if error_message:
            self._show(error_message, "❌ ")
        title, options = self._menus[menu_key]
        print(f"\n{title}:")
        for k, v in options.items():
            print(f"{k} — {v}")
        while True:
            choice = input("> ").strip()
            if choice in options:
                return int(choice)
            self._show(f"Введите {', '.join(options.keys())}", "❌ ")

    def select_scan_file(self, files: List[str], new_files: bool = True) -> Optional[str]:
        if not files:
            self._show("Нет доступных файлов", "❌ ")
            return None
        file_type = "новых" if new_files else "доступных"
        print(f"\n📁 {file_type.capitalize()} файлов ({len(files)}):")
        for i, path in enumerate(files, 1):
            print(f"  {i}. {Path(path).name}")
        print("\nВыберите файл:\nEnter — последний, R — повтор, X — отмена")
        while True:
            choice = input("> ").strip().upper()
            if choice == "":
                return files[-1]
            if choice in {"R", "X"}:
                return None
            if choice.isdigit() and 1 <= int(choice) <= len(files):
                return files[int(choice) - 1]
            self._show(f"Введите 1-{len(files)}, Enter, R или X", "❌ ")

# =========================
# Работа с файлами
# =========================
class FileHandler:
    @staticmethod
    def ensure_dir(path: Union[Path, str], as_dir: Optional[bool] = None) -> EnsureDirResult:
        """
        Гарантирует существование директории, связанной с указанным путём.

        Аргументы:
            path (Union[Path, str]): Путь к директории или файлу.
            as_dir (Optional[bool]): Явное указание, что path — директория:
                - True → path интерпретируется как директория, создаётся напрямую.
                - False → path интерпретируется как файл, создаётся его родительская директория.
                - None → включается эвристика:
                    • если путь заканчивается на /, \ или не имеет суффикса → считается директорией
                    • иначе — считается файлом

        Возвращает:
            EnsureDirResult: результат создания или обнаружения директории.
        """
        p = Path(path)
        if as_dir is True:
            dir_path = p
        elif as_dir is False:
            dir_path = p.parent
        else:
            s = str(p)
            dir_path = p if s.endswith(("/", "\\")) or p.suffix == "" else p.parent
        existed = dir_path.exists()
        if not existed:
            dir_path.mkdir(parents=True, exist_ok=True)
        return EnsureDirResult(created=not existed, dir_path=dir_path)

    @staticmethod
    def read_text(path: Union[Path, str]) -> str:
        return Path(path).read_text(encoding='utf-8', errors='ignore')

    @staticmethod
    def read_image(path: Union[Path, str]) -> np.ndarray:
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
    def write_image(path: Union[Path, str], img: np.ndarray) -> None:
        FileHandler.ensure_dir(path, as_dir=False)
        if not cv2.imwrite(os.fspath(path), img):
            raise IOError(f"Ошибка сохранения изображения: {path}")

    @staticmethod
    def read_yaml(path: Union[Path, str]) -> Dict:
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
        FileHandler.ensure_dir(path, as_dir=False)
        with open(Path(path), 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

# =========================
# Обработка ошибок
# =========================
class ErrorHandler:
    @staticmethod
    def handle(exception: Exception, message: str, critical: bool = False) -> None:
        print(f"❌ {message}: {exception}")
        if critical:
            raise exception