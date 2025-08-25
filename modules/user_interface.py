"""
Текстовый пользовательский интерфейс для системы контроля качества трафаретов.

Модуль предоставляет класс UserInterface для взаимодействия с оператором
через консольный интерфейс с поддержкой различных типов сообщений,
валидации ввода и отображения результатов обработки.
"""

import re
import time
import glob
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
from screeninfo import get_monitors


# Импорт из того же пакета
from .file_handler import FileHandler
from .config_manager import ConfigManager


class UserInterface:
    """
    Текстовый пользовательский интерфейс для взаимодействия с оператором.

    Обеспечивает:
    - Отображение различных типов сообщений (информация, предупреждения, ошибки)
    - Ввод данных с валидацией
    - Отображение результатов обработки
    - Интерактивное взаимодействие с пользователем

    Attributes:
        _config (ConfigManager): Менеджер конфигурации
        _templates (Dict[str, str]): Шаблоны для отображения
        _prefixes (Dict[str, str]): Префиксы для различных типов сообщений
    """

    # Константы для типов сообщений
    MSG_INFO = "info"
    MSG_SUCCESS = "success"
    MSG_WARNING = "warning"
    MSG_ERROR = "error"
    MSG_DEBUG = "debug"

    def __init__(self, config_manager: ConfigManager):
        """
        Инициализация пользовательского интерфейса.

        Args:
            config_manager (ConfigManager): Менеджер конфигурации приложения
        """
        self._config = config_manager
        self._templates = {
            'header': "=" * 60 + "\n  Система контроля качества трафаретов v1.0\n  Автор: Арабов Далер Искандарович\n" + "=" * 60,
            'result_divider': "─" * 50
        }

        self._prefixes = {
            self.MSG_INFO: "",
            self.MSG_SUCCESS: "✅ ",
            self.MSG_WARNING: "⚠️ ",
            self.MSG_ERROR: "❌ ",
            self.MSG_DEBUG: "[DEBUG] ",
        }

    # region Универсальные методы вывода сообщений

    def show_message(self, text: str, kind: str = MSG_INFO) -> None:
        """
        Выводит сообщение с соответствующим префиксом.

        Args:
            text (str): Текст сообщения
            kind (str): Тип сообщения (info, success, warning, error, debug)
        """
        prefix = self._prefixes.get(kind, "")
        print(f"{prefix}{text}")

    def show_success(self, text: str) -> None:
        """Выводит сообщение об успехе."""
        self.show_message(text, kind=self.MSG_SUCCESS)

    def show_warning(self, text: str) -> None:
        """Выводит предупреждающее сообщение."""
        self.show_message(text, kind=self.MSG_WARNING)

    def show_error(self, text: str) -> None:
        """Выводит сообщение об ошибке."""
        self.show_message(text, kind=self.MSG_ERROR)

    def show_debug(self, text: str) -> None:
        """Выводит отладочное сообщение (только в режиме отладки)."""
        if self._config.debug_mode:
            self.show_message(text, kind=self.MSG_DEBUG)

    # endregion

    # region Методы отображения информации

    def show_header(self) -> None:
        """Отображает заголовок программы."""
        self.show_message(self._templates['header'])
        if self._config.debug_mode:
            self.show_debug("Режим отладки включен")

    def show_environment_check(self, issues: List[str]) -> bool:
        """
        Показывает результаты проверки окружения.

        Args:
            issues (List[str]): Список проблем с окружением

        Returns:
            bool: False при критических ошибках, True если проблем нет
        """
        if not issues:
            self.show_success("Проверка окружения пройдена успешно")
            return True

        self.show_error("Обнаружены проблемы с окружением:")
        for issue in issues:
            print(f"  • {issue}")

        return False

    def show_gerber_search_result(self, files: List[Path], order_number: str) -> None:
        """
        Отображает результаты поиска Gerber-файлов.

        Args:
            files (List[Path]): Найденные файлы
            order_number (str): Номер заказа
        """

        if not files:
            self.show_error(f"Gerber-файл с номером {order_number} не найден")
            return

        if len(files) == 1:
            self.show_success(f"Найден Gerber-файл: {files[0].name}")
        else:
            rule_text = {
                'alphabetic_first': 'первый по алфавиту',
                'newest': 'последний по дате изменения',
                'oldest': 'первый по дате изменения'
            }.get(self._config.multiple_files_rule, 'первый найденный')

            self.show_warning(
                f"Найдено {len(files)} файлов с номером {order_number}:")
            for f in files:
                print(f"  • {f.name}")
            self.show_message(f"Выбран {rule_text}: {files[0].name}")

    def show_parsing_stats(self, stats: Dict) -> None:
        """
        Отображает результаты парсинга Gerber-файла.
        Показывает только определенные метрики, если их нет - выводит N/A.

        Args:
            stats (Dict): Статистика парсинга
        """
        self.show_success("Gerber-файл успешно обработан:")

        # Фиксированный набор метрик с описаниями
        metric_descriptions = {
            'board_width_mm': "📐 Ширина платы",
            'board_height_mm': "📐 Высота платы",
            'contour_count': "🔢 Количество контуров",
            'aperture_count': "🔧 Количество апертур",
            'min_contour_area': "📏 Мин. площадь контура",
            'max_contour_area': "📏 Макс. площадь контура",
            'total_area_mm2': "📊 Суммарная площадь",
        }

        # Форматы вывода для каждой метрики
        metric_formats = {
            'board_width_mm': lambda x: f"{x} мм",
            'board_height_mm': lambda x: f"{x} мм",
            'min_contour_area': lambda x: f"{x} мм²",
            'max_contour_area': lambda x: f"{x} мм²",
            'total_area_mm2': lambda x: f"{x} мм²"
        }

        # Выводим все 7 метрик в заданном порядке
        for metric_key, description in metric_descriptions.items():
            value = stats.get(metric_key, 'N/A')

            # Форматируем значение если нужно
            if value != 'N/A' and metric_key in metric_formats:
                formatted_value = metric_formats[metric_key](value)
            else:
                formatted_value = str(value)

            print(f"  {description}: {formatted_value}")

        # Добавляем время обработки отдельно, если оно есть
        processing_time = stats.get('processing_time')
        if processing_time is not None:
            print(f"  ⏱️ Время обработки: {processing_time:.2f} сек")

    def show_scanning_instructions(self) -> None:
        """Отображает инструкции по сканированию."""
        print("\n" + "📷 ИНСТРУКЦИЯ ПО СКАНИРОВАНИЮ:")
        print(self._config.scan_instruction)

    def show_scan_info(self, scan_path: Path, dpi: int, size_pixels: Tuple[int, int],
                       size_mm: Tuple[float, float], file_size: int) -> None:
        """
        Отображает информацию о скане.

        Args:
            scan_path (Path): Путь к файлу скана
            dpi (int): Разрешение скана
            size_pixels (Tuple[int, int]): Размер в пикселях
            size_mm (Tuple[float, float]): Размер в миллиметрах
            file_size (int): Размер файла в байтах
        """
        size_str = FileHandler.format_file_size(file_size)
        dpi_source = "метаданные" if self._config.dpi_priority == 'metadata' else "конфигурация"

        print(f"\n📄 ИНФОРМАЦИЯ О СКАНЕ:")
        print(f"  📁 Файл: {scan_path.name}")
        print(f"  🔍 DPI: {dpi} (источник: {dpi_source})")
        print(f"  📐 Размер: {size_pixels[0]}×{size_pixels[1]} пикселей")
        print(f"  📏 Размер: {size_mm[0]:.1f}×{size_mm[1]:.1f} мм")
        print(f"  💾 Размер файла: {size_str}")

    def show_preprocessing_result(self, success: bool, details: Dict) -> None:
        """
        Отображает результаты предобработки.

        Args:
            success (bool): Успешность обработки
            details (Dict): Детали обработки
        """
        if success:
            self.show_success("Предобработка завершена успешно")

            # Показываем только доступные метрики
            if 'original_size' in details:
                print(
                    f"  📷 Исходный размер: {details['original_size']} пикселей")
            if 'processed_size' in details:
                print(
                    f"  ✂️  Обрезанный размер: {details['processed_size']} пикселей")
            if 'contour_count' in details:
                print(f"  🔢 Найдено контуров: {details['contour_count']}")
            if 'mean_contour_area' in details:
                print(
                    f"  📊 Средняя площадь контура: {details['mean_contour_area']:.1f} px²")
            if 'crop_ratio' in details:
                crop_pct = (1 - details['crop_ratio']) * 100
                print(f"  📊 Процент обрезки: {crop_pct:.1f}%")

        else:
            self.show_error("Предобработка не удалась")
            if 'error' in details:
                print(f"  Причина: {details['error']}")

    def show_reference_generation(self, size_pixels: Tuple[int, int],
                                  size_mm: Tuple[float, float], dpi: int) -> None:
        """
        Отображает информацию о генерации эталонного изображения.

        Args:
            size_pixels (Tuple[int, int]): Размер в пикселях
            size_mm (Tuple[float, float]): Размер в миллиметрах
            dpi (int): Разрешение изображения
        """
        self.show_success("Создано бинарное изображение эталона:")
        print(f"  📐 Размер: {size_pixels[0]}×{size_pixels[1]} пикселей")
        print(f"  📏 Размер: {size_mm[0]:.1f}×{size_mm[1]:.1f} мм")
        print(f"  🔍 Разрешение: {dpi} DPI")

    def show_alignment_results(self, result: Dict) -> None:
        """
        Отображает результаты совмещения изображений.

        Args:
            result (Dict): Результаты совмещения
        """
        correlation = result.get('correlation', 0.0)
        orientation = result.get('orientation', 'N/A')

        # Используем единую функцию для определения статуса
        from .data_models import calculate_alignment_status
        status_key, status_text = calculate_alignment_status(
            correlation,
            self._config.high_correlation_threshold,
            self._config.medium_correlation_threshold
        )

        # Добавляем emoji для визуального отличия
        status_emoji = {
            "success": "✅",
            "warning": "🟡",
            "failed": "❌"
        }.get(status_key, "")

        print("\n" + self._templates['result_divider'])
        print("📊 РЕЗУЛЬТАТЫ СОВМЕЩЕНИЯ")
        print(self._templates['result_divider'])
        print(f"🎯 Корреляция:         {correlation:.3f}")
        print(f"📐 Ориентация скана:   {orientation}")
        print(f"📋 Статус совмещения:  {status_text} {status_emoji}")

        # Отладочная информация
        if self._config.include_debug_info and 'rotation' in result:
            print(f"📐 Поворот:        {result['rotation']:+.2f}°")
            print(f"↔️ Сдвиг X:        {result['shift_x']:+.3f} мм")
            print(f"↕️ Сдвиг Y:        {result['shift_y']:+.3f} мм")

        print(self._templates['result_divider'])

    def show_files_saved(self, saved_files: List[Path]) -> None:
        """
        Отображает информацию о сохраненных файлах.

        Args:
            saved_files (List[Path]): Список сохраненных файлов
        """
        if not saved_files:
            return

        self.show_success("Сохранены файлы результатов:")
        for file_path in saved_files:
            size_str = FileHandler.format_file_size(file_path.stat().st_size)
            print(f"  📁 {file_path.name} ({size_str})")

    # endregion

    # region Методы ввода данных

    def get_operator_name(self) -> str:
        """
        Запрашивает ФИО оператора с валидацией.

        Returns:
            str: Валидное ФИО оператора
        """
        default = self._config.default_operator_name
        pattern = re.compile(r"^[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-\.\s]{1,50}$")

        while True:
            name = input(
                f'\nВведите ваше ФИО (или Enter для "{default}"):\n> ').strip()
            if name == "":
                return default
            if pattern.fullmatch(name):
                return re.sub(r"\s+", " ", name)  # нормализуем пробелы
            self.show_error(
                "ФИО должно содержать 2-50 символов: буквы, пробелы, дефис, точка")

    def get_order_number(self) -> Optional[str]:
        """
        Запрашивает номер заказа с валидацией.

        Returns:
            Optional[str]: Номер заказа или None если выход
        """
        if self._config.preset_order_number:
            self.show_success(
                f"Использован предустановленный номер заказа: {self._config.preset_order_number}")
            return self._config.preset_order_number

        example = self._config.generate_example_order_number()
        digits = self._config.order_number_digits

        while True:
            prompt = f"\nВведите {digits}-значный номер заказа (пример: {example}) или 'exit' для выхода:\n> "
            value = input(prompt).strip()

            if value.lower() == 'exit':
                return None

            if re.fullmatch(rf'\d{{{digits}}}', value):
                return value

            self.show_error(f"Номер должен содержать ровно {digits} цифр")

    # endregion

    # region Методы ожидания и проверки

    def wait_for_scan_file(self) -> Path:
        """
        Ожидает появления файла скана в папке сканирования.

        Returns:
            Path: Путь к найденному файлу скана

        Raises:
            TimeoutError: Если превышено время ожидания
        """
        self.show_message(
            f"Ожидание файла скана в папке: {self._config.scan_folder}")
        self.show_message("Поддерживаемые форматы: " +
                          ", ".join(self._config.supported_image_formats))

        start_time = time.time()
        timeout = self._config.scan_wait_timeout

        while True:
            if timeout > 0 and (time.time() - start_time) > timeout:
                raise TimeoutError(
                    f"Превышено время ожидания файла скана ({timeout} сек)")

            # Ищем новые файлы изображений
            scan_files = []
            for ext in self._config.supported_image_formats:
                pattern = str(self._config.scan_folder / f"*{ext}")
                scan_files.extend(glob.glob(pattern, recursive=False))

            if scan_files:
                # Берем самый новый файл
                newest_file = max(scan_files, key=os.path.getmtime)
                scan_path = Path(newest_file)
                self.show_success(f"Обнаружен файл скана: {scan_path.name}")
                return scan_path

            time.sleep(self._config.file_check_interval)

    # endregion

    # region Интерактивные методы (вопросы пользователю)

    def ask_create_directories(self, missing_dirs: List[str]) -> bool:
        """
        Спрашивает пользователя о создании отсутствующих директорий.

        Args:
            missing_dirs (List[str]): Список отсутствующих директорий

        Returns:
            bool: True если нужно создать директории
        """
        self.show_warning("Не найдены следующие директории:")
        for dir_path in missing_dirs:
            print(f"  • {dir_path}")

        while True:
            response = input(
                "\nСоздать отсутствующие директории? (y/n): ").strip().lower()
            if response in ['y', 'yes', 'да', 'д']:
                return True
            elif response in ['n', 'no', 'нет', 'н']:
                return False
            self.show_error("Введите 'y' для создания или 'n' для отмены")

    def ask_preprocessing_action(self) -> int:
        """
        Спрашивает пользователя о действии при частичной предобработке.

        Returns:
            int: Выбранное действие (1-4)
        """
        self.show_warning(
            "Размеры скана не совпадают с эталоном в пределах допуска")
        print("\nВыберите действие:")
        print("1 — Продолжить обработку")
        print("2 — Повторить сканирование")
        print("3 — Выбрать новый заказ")
        print("4 — Выйти")

        while True:
            choice = input("> ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            self.show_error("Введите 1, 2, 3 или 4")

    def ask_alignment_failed_action(self, correlation: float) -> int:
        """
        Спрашивает пользователя о действии при неудачном совмещении.

        Args:
            correlation (float): Уровень корреляции

        Returns:
            int: Выбранное действие (1-3)
        """
        if correlation < self._config.low_correlation_threshold:
            self.show_error("Совмещение не удалось")
        else:
            self.show_warning(
                "Совмещение сомнительно или обнаружен высокий брак")

        print("\nВыберите действие:")
        print("1 — Повторить сканирование")
        print("2 — Ввести новый номер заказа")
        print("3 — Выйти")

        while True:
            choice = input("> ").strip()
            if choice in ['1', '2', '3']:
                return int(choice)
            self.show_error("Введите 1, 2 или 3")

    def ask_parsing_failed_action(self, error_details: Optional[str] = None) -> int:
        """
        Спрашивает пользователя о действии при неудачном парсинге.

        Args:
            error_details (Optional[str]): Детали ошибки

        Returns:
            int: Выбранное действие (1-4)
        """
        self.show_error("Ошибка парсинга Gerber-файла")
        if error_details:
            print(f"Причина: {error_details}")

        print("\nВыберите действие:")
        print("1 — Показать детали ошибки")
        print("2 — Повторить парсинг с тем же файлом")
        print("3 — Ввести новый номер заказа")
        print("4 — Выйти")

        while True:
            choice = input("> ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            self.show_error("Введите 1, 2, 3 или 4")

    def show_main_menu(self) -> int:
        """
        Отображает главное меню программы.

        Returns:
            int: Выбранный пункт меню (1-2)
        """
        print("\n" + "="*40)
        print("ГЛАВНОЕ МЕНЮ")
        print("="*40)
        print("1 — Ввести новый номер заказа")
        print("2 — Выйти из программы")

        while True:
            choice = input("\nВыберите действие: ").strip()
            if choice in ['1', '2']:
                return int(choice)
            self.show_error("Введите 1 или 2")

    def confirm_exit(self) -> bool:
        """
        Подтверждение выхода из программы.

        Returns:
            bool: True если нужно выйти
        """
        while True:
            response = input(
                "\nВы действительно хотите выйти? (y/n): ").strip().lower()
            if response in ['y', 'yes', 'да', 'д']:
                return True
            elif response in ['n', 'no', 'нет', 'н']:
                return False
            self.show_error("Введите 'y' для выхода или 'n' для продолжения")

    # endregion

    def show_scan_waiting_animation(self, timeout: float, elapsed: float) -> None:
        """Показывает анимацию ожидания скана."""
        animation_chars = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        anim_index = int(elapsed * 10) % len(animation_chars)
        anim_char = animation_chars[anim_index]

        remaining = int(timeout - elapsed) if timeout > 0 else "∞"
        print(f"\r{anim_char} Ожидание нового файла... (таймаут: {remaining} сек) ",
              end="", flush=True)

    def ask_scan_timeout_action(self, last_existing_file: Optional[Path]) -> int:
        """
        Спрашивает пользователя о действии при таймауте ожидания скана.

        Returns:
            int: Выбранное действие (1-4)
        """
        self.show_warning("Время ожидания нового файла истекло")

        print("\nВыберите действие:")
        print("1 — Продолжить ожидание нового файла")

        # Пункт 2 показываем только если есть существующие файлы
        if last_existing_file:
            print(
                f"2 — Использовать последний существующий файл: {last_existing_file.name}")

        print("3 — Ввести новый номер заказа")
        print("4 — Выйти из программы")

        while True:
            choice = input("\nВыберите действие (1-4): ").strip()

            # Валидация ввода в зависимости от доступных опций
            valid_choices = ['1', '3', '4']
            if last_existing_file:
                valid_choices.append('2')

            if choice in valid_choices:
                return int(choice)

            self.show_error(
                f"Введите число из доступных опций: {', '.join(valid_choices)}")

    def ask_scan_failed_action(self) -> int:
        """
        Спрашивает пользователя о действии при ошибке сканирования.

        Returns:
            int: Выбранное действие (1-3)
        """
        self.show_error("Ошибка при обработке файла скана")

        print("\nВыберите действие:")
        print("1 — Повторить сканирование")
        print("2 — Ввести новый номер заказа")
        print("3 — Выйти из программы")

        while True:
            choice = input("> ").strip()
            if choice in ['1', '2', '3']:
                return int(choice)
            self.show_error("Введите 1, 2 или 3")

    def show_debug(self, text: str) -> None:
        """Выводит отладочное сообщение (только в режиме отладки)."""
        if self._config.debug_mode:
            self.show_message(text, kind=self.MSG_DEBUG)

    def ask_preprocessing_failed_action(self, error_details: str) -> int:
        """Спрашивает действие при неудачной предобработке."""
        self.show_error(f"Предобработка не удалась: {error_details}")
        print("\nВыберите действие:")
        print("1 — Повторить сканирование")
        print("2 — Выбрать новый заказ")
        print("3 — Выйти")

        while True:
            choice = input("> ").strip()
            if choice in ['1', '2', '3']:
                return int(choice)
            self.show_error("Введите 1, 2 или 3")

    def show_dimension_mismatch_warning(self, gerber_size_mm: Tuple[float, float], scan_size_mm: Tuple[float, float], tolerance_percent: float) -> None:
        """
            Отображает детализированное предупреждение о несоответствии размеров.

            Args:
                gerber_size_mm (Tuple[float, float]): Размеры эталона (ширина, высота) в мм.
                scan_size_mm (Tuple[float, float]): Размеры обрезанного скана (ширина, высота) в мм.
                tolerance_percent (float): Установленный допуск в процентах.
            """
        self.show_warning(
            "Размеры скана и эталона не совпадают в пределах допуска!")
        print(
            f"  - Размеры эталона (Gerber): {gerber_size_mm[0]:.1f} x {gerber_size_mm[1]:.1f} мм")
        print(
            f"  - Размеры скана (обрезан.): {scan_size_mm[0]:.1f} x {scan_size_mm[1]:.1f} мм")

        diff_w = abs(gerber_size_mm[0] - scan_size_mm[0])
        diff_h = abs(gerber_size_mm[1] - scan_size_mm[1])
        print(f"  - Абсолютное отклонение:   {diff_w:.1f} x {diff_h:.1f} мм")

        print(f"  - Установленный допуск:     {tolerance_percent:.1f}%")

    def show_combined_image(self, image: np.ndarray) -> None:
        """Отображает совмещенное изображение в центре с масштабом."""
        monitor = get_monitors()[0]
        screen_width, screen_height = monitor.width, monitor.height
        h, w = image.shape[:2]

        # Масштаб до 80% экрана
        max_width, max_height = int(
            screen_width * 0.8), int(screen_height * 0.8)
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)

        # Центрирование
        x = (screen_width - new_w) // 2
        y = (screen_height - new_h) // 2
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Result', new_w, new_h)
        cv2.moveWindow('Result', x, y)
        cv2.imshow('Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
