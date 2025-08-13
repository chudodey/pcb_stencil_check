#!/usr/bin/env python3
"""
Оптимизированная система контроля качества трафаретов
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
from modules import ConfigManager, GerberParser, ImageProcessor, AlignmentEngine, SessionLogger, UserInterface

class MaskCompareApp:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self._init_components()
        self._setup_paths()
        
    def _init_components(self):
        """Инициализация компонентов системы"""
        self.cfg = ConfigManager()
        self.gerber = GerberParser()
        self.img_processor = ImageProcessor()
        self.aligner = AlignmentEngine()
        self.logger = SessionLogger()
        self.ui = UserInterface()
        
        self.operator: Optional[str] = None
        self.current_order: Optional[str] = None
        self.gerber_file: Optional[Path] = None
        self.scan_counter = 0

    def _setup_paths(self):
        """Настройка путей из конфигурации"""
        self.source_dir = self.cfg.source_dir
        self.scan_dir = self.cfg.scan_dir
        self.output_dir = self.cfg.output_dir

    def run(self):
        """Главный цикл приложения"""
        try:
            self._initialize()
            self._setup_operator()
            
            while order_number := self.ui.get_order_number():
                self._process_order(order_number)
                
        except KeyboardInterrupt:
            self.ui.show_warning("\nРабота прервана пользователем")
        except Exception as e:
            self._handle_error(e)
        finally:
            self.ui.show_success("Работа завершена")
            self._cleanup()

    # --------------------------
    # Основные этапы обработки
    # --------------------------
    def _initialize(self) -> bool:
        """Инициализация системы"""
        self.ui.show_header()
        self.logger.start_session()
        
        if self.debug_mode:
            print(f"[DEBUG] Конфиг загружен. DPI: {self.cfg.dpi}, Output: {self.output_dir}")
        return True

    def _setup_operator(self):
        """Настройка оператора"""
        self.operator = self.ui.get_operator_name(self.cfg.default_operator)
        self.logger.set_operator(self.operator)

    # def _get_order(self) -> Optional[str]:
    #     """Получение и валидация номера заказа"""
    #     while True:
    #         if order := self.ui.get_order_number():
    #             return order
    #         else:
    #             return None

    def _process_order(self, order: str):
        """Полный цикл обработки заказа"""
        self.current_order = order
        self.scan_counter = 0
        
        self.gerber_file = self._find_and_parse_gerber(order)
        if not self.gerber_file:
            return
        
        self.logger.log_order_start(order)
        
        self._setup_workspace()
        self._process_scans()

    def _find_and_parse_gerber(self, order: str) -> Optional[Path]:
        """Поиск и парсинг Gerber-файла"""
        while True:
            if gerber_file := self.gerber.find_gerber_file(order):
                try:
                    stats = self.gerber.parse_file(gerber_file)
                    self.ui.show_parsing_stats(stats)
                    return gerber_file
                except Exception as e:
                    choice = self.ui.show_parsing_error_menu(str(e))
                    if choice == 3:  # Выйти
                        sys.exit(0)
                    if choice == 2:  # Новый заказ
                        return None
            else:
                choice = self.ui.show_file_not_found_menu(order)
                if choice == 2:  # Выйти
                    sys.exit(0)
                return None

    def _setup_workspace(self):
        """Настройка рабочей директории"""
        workspace = self.output_dir / self.current_order
        workspace.mkdir(parents=True, exist_ok=True)
        
        # Копирование Gerber и генерация эталона
        self.gerber.copy_to_workspace(self.gerber_file, workspace)
        self.gerber.generate_reference_image(workspace)
        
        if self.debug_mode:
            print(f"[DEBUG] Рабочая папка: {workspace}")

    def _process_scans(self):
        """Цикл обработки сканов"""
        while scan_file := self._get_scan_file():
            self.scan_counter += 1
            if self._process_scan(scan_file):
                if (action := self._get_next_action()) == 4:  # Выйти
                    return
                if action in (2, 3):  # Новый Gerber или заказ
                    break

    # --------------------------
    # Методы работы со сканами
    # --------------------------
    def _get_scan_file(self) -> Optional[Path]:
        """Получение файла скана"""
        self.ui.show_scanning_instructions()
        start_time = datetime.now()
        self.ui.wait_for_scan_completion()
        
        if new_files := self.img_processor.find_new_scans(start_time):
            return self.ui.select_scan_file(new_files)
        return self.ui.select_scan_file(self.img_processor.get_all_scan_files(), new_files=False)

    def _process_scan(self, scan_path: Path) -> bool:
        """Обработка одного скана"""
        try:
            scan_copy = self._copy_scan(scan_path)
            processed = self.img_processor.preprocess_scan(scan_copy)
            
            debug_params = {
                'order_number': self.current_order,
                'scan_counter': self.scan_counter,
                'debug_mode': self.debug_mode
            }
            
            result = self.aligner.align_images(
                self.gerber.get_reference_image(),
                processed,
                debug_params
            )
            
            self._handle_alignment_result(result, scan_path)
            return True
            
        except Exception as e:
            self.ui.show_error(f"Ошибка обработки: {e}")
            if self.debug_mode:
                import traceback
                traceback.print_exc()
            return False

    def _copy_scan(self, scan_path: Path) -> Path:
        """Копирование скана в рабочую директорию"""
        dest = self.img_processor.copy_scan_to_workspace(
            scan_path, 
            self.current_order, 
            self.scan_counter
        )
        if self.debug_mode:
            print(f"[DEBUG] Скан сохранен как: {dest}")
        return dest

    def _handle_alignment_result(self, result: Dict, scan_path: Path):
        """Обработка результатов совмещения"""
        self.ui.show_alignment_results(result)
        self.logger.log_scan_result(
            self.current_order,
            str(scan_path),
            result
        )
        
        if self.debug_mode:
            print(f"[DEBUG] Результат: {result['match_percentage']:.1f}% совпадение")

    def _get_next_action(self) -> int:
        """Получение следующего действия от пользователя"""
        return self.ui.show_next_action_menu()

    # --------------------------
    # Вспомогательные методы
    # --------------------------
    def _handle_error(self, error: Exception):
        """Обработка ошибок"""
        self.ui.show_error(f"Критическая ошибка: {error}")
        if self.debug_mode:
            import traceback
            traceback.print_exc()

    def _cleanup(self):
        """Завершение работы"""
        self.logger.end_session()
        self.ui.show_success("Работа завершена")

def main():
    parser = argparse.ArgumentParser(description='Контроль качества трафаретов')
    parser.add_argument('--debug', '-d', action='store_true', help='Режим отладки')
    args = parser.parse_args()

    if args.debug:
        print("\n".join([
            "=" * 60,
            "РЕЖИМ ОТЛАДКИ АКТИВИРОВАН",
            "=" * 60,
            "• Детализация всех этапов",
            "• Сохранение промежуточных результатов",
            "• Полные трассировки ошибок",
            "=" * 60
        ]))

    MaskCompareApp(debug_mode=args.debug).run()

if __name__ == "__main__":
    main()