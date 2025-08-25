"""
Логирование работы оператора и результатов обработки
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import asdict

from .data_models import SessionData, ScanResult
from .file_handler import FileHandler
from .config_manager import ConfigManager


class SessionLogger:
    """Логирование работы оператора и результатов обработки."""

    def __init__(self, config_manager: ConfigManager):
        self._config = config_manager
        self._session: Optional[SessionData] = None
        self._current_operator: Optional[str] = None
        self._session_log: List[str] = []
        self._current_order_log: List[str] = []

    def start_session(self) -> None:
        """Начинает новую сессию работы."""
        self._session = SessionData(
            start_time=datetime.now().strftime(self._config.datetime_format),
            operator=self._current_operator or self._config.default_operator_name
        )
        self._session_log = []
        self.log("Начало сессии работы")

    def set_operator(self, name: str) -> None:
        """Устанавливает имя оператора для текущей сессии."""
        self._current_operator = name
        if self._session:
            self._session.operator = name
        self.log(f"Оператор: {name}")

    def log(self, message: str) -> None:
        """Добавляет сообщение в лог сессии."""
        timestamp = datetime.now().strftime(self._config.datetime_format)
        log_entry = f"[{timestamp}] {message}"
        self._session_log.append(log_entry)
        if self._config.debug_mode:
            print(f"[LOG] {log_entry}")

    def start_order_processing(self, order_number: str, gerber_file: str) -> None:
        """Начинает обработку нового заказа."""
        if self._session and order_number not in self._session.orders:
            self._session.orders.append(order_number)

        self._current_order_log = []
        self.log(f"Начало обработки заказа {order_number}")
        self.log(f"Gerber-файл: {gerber_file}")

    def log_parsing_results(self, order_number: str, stats: Dict) -> None:
        """Логирует результаты парсинга Gerber-файла."""
        self.log(f"Парсинг заказа {order_number} завершен:")
        self.log(
            f"  Размер платы: {stats.get('board_width', 'N/A')}×{stats.get('board_height', 'N/A')} мм")
        self.log(f"  Количество полигонов: {stats.get('polygon_count', 0)}")
        self.log(
            f"  Минимальный элемент: {stats.get('min_element_size', 'N/A')} мм")
        self.log(
            f"  Максимальный элемент: {stats.get('max_element_size', 'N/A')} мм")
        self.log(f"  Суммарная площадь: {stats.get('total_area', 'N/A')} мм²")

    def log_scan_info(self, scan_path: Path, dpi: int, size_mm: Tuple[float, float], file_size: int) -> None:
        """Логирует информацию о скане."""
        size_str = FileHandler.format_file_size(file_size)
        self.log(f"Получен скан: {scan_path.name}")
        self.log(f"  DPI: {dpi}")
        self.log(f"  Размер: {size_mm[0]:.1f}×{size_mm[1]:.1f} мм")
        self.log(f"  Размер файла: {size_str}")

    def log_preprocessing_result(self, success: bool, details: Dict) -> None:
        """Логирует результаты предобработки."""
        if success:
            self.log("Предобработка завершена успешно")
            if 'crop_percentage' in details:
                self.log(f"  Обрезка: {details['crop_percentage']:.1f}%")
        else:
            self.log("Предобработка не удалась")
            if 'error' in details:
                self.log(f"  Ошибка: {details['error']}")

    # ИЗМЕНЕНО: Добавлен аргумент 'workspace' для передачи готового пути

    def log_alignment_result(self, order_number: str, scan_path: Path, result: Dict, workspace: Path) -> None:
        """Логирует результаты совмещения."""
        if not self._session:
            print("DEBUG: No session, skipping log")  # ← ДОБАВЬТЕ
            return

        # # ← ДОБАВЬТЕ
        # print(f"DEBUG: log_alignment_result called for {order_number}")
        # print(f"DEBUG: result keys = {result.keys()}")  # ← ДОБАВЬТЕ
        # # ← ДОБАВЬТЕ
        # print(f"DEBUG: result['metrics'] = {result.get('metrics', {})}")

        self._session.total_scans += 1
        correlation = result.get('correlation', 0.0)

        # Используем единую функцию для определения статуса
        from .data_models import calculate_alignment_status
        status_key, status_text = calculate_alignment_status(
            correlation,
            self._config.high_correlation_threshold,
            self._config.medium_correlation_threshold
        )

        if status_key == "success":
            self._session.successful_scans += 1

        # Основная информация
        self.log(f"Совмещение заказа {order_number}: {status_text}")
        self.log(f"  Корреляция: {correlation:.3f}")
        self.log(f"  Совпадение: {result.get('match_percentage', 0):.1f}%")

        # Отладочная информация
        if self._config.include_debug_info:
            self.log(f"  Поворот: {result.get('rotation', 0):.2f}°")
            self.log(f"  Сдвиг X: {result.get('shift_x', 0):.3f} мм")
            self.log(f"  Сдвиг Y: {result.get('shift_y', 0):.3f} мм")

        # Создание записи о скане с русским текстом статуса
        scan_result = ScanResult(
            file=scan_path.name,
            time=datetime.now().strftime(self._config.datetime_format),
            rotation=f"{result.get('rotation', 0):.2f}°",
            shift_x=f"{result.get('shift_x', 0):.3f}mm",
            shift_y=f"{result.get('shift_y', 0):.3f}mm",
            match=f"{result.get('match_percentage', 0):.1f}%",
            correlation=correlation,
            status=status_text  # ← Передаем русский текст напрямую
        )

        # Сохранение в лог заказа
        self._save_order_log(order_number, scan_result, result, workspace)

    def _save_order_log(self, order_number: str, scan_result: ScanResult, full_result: Dict, workspace: Path) -> None:
        """Сохраняет лог заказа в файл, используя шаблон из конфигурации."""
        # print(
        #     f"DEBUG: _save_order_log called for order {order_number}")
        # print(f"DEBUG: log_level = {self._config.log_level}")

        if self._config.log_level == 'none':
            # print("DEBUG: log_level is 'none', skipping")
            return

        # print(f"DEBUG: workspace = {workspace}")  # ← ДОБАВЬТЕ
        # print(f"DEBUG: workspace exists = {workspace.exists()}")  # ← ДОБАВЬТЕ

        # Подготовка данных для шаблона
        log_data = {
            'datetime': scan_result.time,
            'operator_name': self._session.operator if self._session else 'Неизвестен',
            'order_number': order_number,
            'correlation_result': scan_result.correlation,
            'alignment_status': scan_result.status,  # Уже содержит русский текст
        }

        # print(f"DEBUG: log_data = {log_data}")  # ← ДОБАВЬТЕ

        # Добавляем дополнительные метрики из full_result если они есть
        log_data.update({
            'rotation_angle': full_result.get('rotation', 0),
            'shift_x_mm': full_result.get('shift_x', 0),
            'shift_y_mm': full_result.get('shift_y', 0),
            'match_percentage': full_result.get('match_percentage', 0),
            'inliers_count': full_result.get('inliers', 0),
            'error_value': full_result.get('error', 0),
            'orientation': full_result.get('orientation', 'N/A'),
            'approach_used': full_result.get('approach_used', 'N/A')
        })

        # Проверяем шаблон
        # ← ДОБАВЬТЕ
        # print(
        #     f"DEBUG: short_log_template = '{self._config.short_log_template}'")

        template = self._config.short_log_template
        if '\\n' in template:
            template = template.replace('\\n', '\n')

        # Формирование контента
        try:
            short_content = template.format(**log_data)
            # print(f"DEBUG: short_content = '{short_content}'")  # ← ДОБАВЬТЕ
        except KeyError as e:
            print(f"ERROR: Missing key in template: {e}")  # ← ДОБАВЬТЕ
            return

        # Записываем краткий лог
        if self._config.log_level in ['short', 'detailed']:
            short_log_path = self._config.get_filename(
                self._config.short_log_filename, order_number, workspace
            )

            # print(f"DEBUG: Writing to {short_log_path}")  # ← ДОБАВЬТЕ
            try:
                FileHandler.write_text(short_log_path, short_content.strip())
                # print("DEBUG: File written successfully")  # ← ДОБАВЬТЕ
            except Exception as e:
                print(f"ERROR: Failed to write file: {e}")  # ← ДОБАВЬТЕ

                # Подробный лог
        if self._config.log_level == 'detailed':
            detailed_log_path = self._config.get_filename(
                self._config.detailed_log_filename, order_number, workspace
            )

            # Базовый контент
            detailed_content = short_content.strip() + "\n\n" + "="*50 + \
                "\nПОДРОБНЫЙ ЛОГ СЕССИИ:\n" + "="*50 + "\n"
            detailed_content += "\n".join(self._session_log)

            # Добавляем детальную информацию о совмещении
            detailed_content += "\n\n" + "="*50 + "\nДЕТАЛИ СОВМЕЩЕНИЯ:\n" + "="*50 + "\n"
            detailed_content += f"Угол поворота: {log_data['rotation_angle']:.2f}°\n"
            detailed_content += f"Смещение по X: {log_data['shift_x_mm']:.3f} мм\n"
            detailed_content += f"Смещение по Y: {log_data['shift_y_mm']:.3f} мм\n"
            detailed_content += f"Процент совпадения: {log_data['match_percentage']:.1f}%\n"
            detailed_content += f"Количество инлайнеров: {log_data['inliers_count']}\n"
            detailed_content += f"Ошибка репроекции: {log_data['error_value']:.3f}\n"
            detailed_content += f"Ориентация: {log_data['orientation']}\n"
            detailed_content += f"Использованный подход: {log_data['approach_used']}\n"

            # Отладочная информация
            if self._config.include_debug_info and 'debug_info' in full_result:
                detailed_content += "\n\n" + "="*50 + \
                    "\nОТЛАДОЧНАЯ ИНФОРМАЦИЯ:\n" + "="*50 + "\n"
                debug_info = full_result['debug_info']
                detailed_content += f"Матрица преобразования: {debug_info.get('matrix', 'N/A')}\n"
                detailed_content += f"Кандидатов проверено: {debug_info.get('candidates_tried', 0)}\n"

            FileHandler.write_text(detailed_log_path, detailed_content)

    def end_session(self) -> None:
        """Завершает текущую сессию."""
        if not self._session:
            return

        self._session.end_time = datetime.now().strftime(self._config.datetime_format)
        self.log("Завершение сессии работы")
        self.log(f"Обработано заказов: {len(self._session.orders)}")
        self.log(f"Выполнено сканирований: {self._session.total_scans}")
        self.log(f"Успешных сканирований: {self._session.successful_scans}")

        # Сохранение сессии в YAML (если нужно для глобальной статистики)
        if self._config.debug_mode:
            session_file = self._config.output_folder / \
                f"session_{self._session.start_time.replace(':', '-').replace(' ', '_')}.yaml"
            FileHandler.write_yaml(session_file, asdict(self._session))

    def log_error(self, order_number: str, error_message: str) -> None:
        """Логирует ошибку обработки заказа."""
        log_entry = f"ОШИБКА в заказе {order_number}: {error_message}"
        self.log(log_entry)
        print(f"❌ {log_entry}")
