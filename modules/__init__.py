"""
Модули системы контроля качества трафаретов
"""

# Импорты основных классов
from .config_manager import ConfigManager
from .session_logger import SessionLogger
from .user_interface import UserInterface
from .file_handler import FileHandler
from .error_handler import ErrorHandler

# Импорты процессоров
from .gerber_processor import GerberProcessor, GerberRasterizer
from .image_processor import ImageProcessor
from .alignment_engine import AlignmentEngine

# Импорты dataclasses (опционально, если нужны напрямую)
from .data_models import EnsureDirResult, ScanResult, OrderLog, SessionData

__all__ = [
    'ConfigManager',
    'SessionLogger',
    'UserInterface',
    'FileHandler',
    'ErrorHandler',
    'GerberProcessor',
    'GerberRasterizer',
    'ImageProcessor',
    'AlignmentEngine',
    'EnsureDirResult',
    'ScanResult',
    'OrderLog',
    'SessionData'
]

__version__ = '1.0.0'
__author__ = 'Арабов Далер Искандарович'
