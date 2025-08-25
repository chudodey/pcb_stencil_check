"""
Централизованная обработка ошибок
"""

import sys
from typing import Optional, Union
from pathlib import Path

# Обычно ErrorHandler автономен, но если нужны UI-импорты:
from .user_interface import UserInterface

class ErrorHandler:
    """Централизованная обработка ошибок с контекстной информацией."""

    @staticmethod
    def handle_exception(exception: Exception, context: str, critical: bool = False, 
                        ui: Optional[UserInterface] = None) -> None:
        """
        Обрабатывает исключение с выводом контекстной информации.
        
        Args:
            exception: Исключение для обработки
            context: Контекст, в котором произошла ошибка
            critical: Является ли ошибка критической (завершает программу)
            ui: Интерфейс пользователя для вывода сообщений
        """
        error_msg = f"{context}: {str(exception)}"
        
        if ui:
            ui.show_error(error_msg)
            if critical:
                ui.show_error("Критическая ошибка, программа будет завершена")
        else:
            print(f"❌ {error_msg}")
            if critical:
                print("❌ Критическая ошибка, программа будет завершена")
        
        if critical:
            sys.exit(1)

    @staticmethod
    def handle_file_error(path: Union[Path, str], operation: str, 
                         exception: Exception, ui: Optional[UserInterface] = None) -> None:
        """Обрабатывает ошибки файловых операций."""
        path_str = str(path)
        context = f"Ошибка {operation} файла '{path_str}'"
        ErrorHandler.handle_exception(exception, context, critical=False, ui=ui)

    @staticmethod
    def handle_image_error(path: Union[Path, str], operation: str, 
                          exception: Exception, ui: Optional[UserInterface] = None) -> None:
        """Обрабатывает ошибки работы с изображениями."""
        path_str = str(path)
        context = f"Ошибка {operation} изображения '{path_str}'"
        ErrorHandler.handle_exception(exception, context, critical=False, ui=ui)

    @staticmethod
    def handle_config_error(config_file: str, exception: Exception, 
                           ui: Optional[UserInterface] = None) -> None:
        """Обрабатывает ошибки конфигурации."""
        context = f"Ошибка конфигурации в файле '{config_file}'"
        ErrorHandler.handle_exception(exception, context, critical=True, ui=ui)

    @staticmethod
    def validate_and_handle(condition: bool, error_message: str, 
                           exception_type: type = ValueError, critical: bool = False,
                           ui: Optional[UserInterface] = None) -> None:
        """Проверяет условие и обрабатывает ошибку при его невыполнении."""
        if not condition:
            exception = exception_type(error_message)
            ErrorHandler.handle_exception(exception, "Ошибка валидации", critical, ui)