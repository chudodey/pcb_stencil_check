"""
Модули системы контроля качества трафаретов
"""
from .utils import SessionLogger, ConfigManager, UserInterface

from .gerber_processor import GerberProcessor
from .image_processor import ImageProcessor
from .alignment_engine import AlignmentEngine

__all__ = [
    'ConfigManager',
    'GerberProcessor', 
    'ImageProcessor',
    'AlignmentEngine',
    'SessionLogger',
    'UserInterface'
]

__version__ = '1.0.0'
__author__ = 'Арабов Далер Искандарович'