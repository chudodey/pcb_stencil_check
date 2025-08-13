# tests/test_modules_access.py
"""
Тест наличия и доступа ко всем модулям системы.
Проверяет, что модули из __init__.py доступны и содержат ожидаемые классы.
"""

import pytest
import sys
from pathlib import Path


# Добавляем корень проекта в PYTHONPATH
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from modules import (
    ConfigManager,
    GerberProcessor,
    ImageProcessor,
    AlignmentEngine,
    SessionLogger,
    UserInterface,
    __version__,
    __author__,
)

def test_version_and_author():
    """Проверка версии и автора."""
    assert __version__ == "1.0.0"
    assert __author__ == "Арабов Далер Искандарович"

def test_config_manager():
    """Проверка наличия и доступа к ConfigManager."""
    assert hasattr(ConfigManager, "__init__"), "ConfigManager не имеет метода __init__"
    # Дополнительные проверки, если нужно (например, методы load_config)

def test_gerber_processor():
    """Проверка наличия и доступа к GerberProcessor."""
    assert hasattr(GerberProcessor, "__init__"), "GerberProcessor не имеет метода __init__"

def test_image_processor():
    """Проверка наличия и доступа к ImageProcessor."""
    assert hasattr(ImageProcessor, "__init__"), "ImageProcessor не имеет метода __init__"

def test_alignment_engine():
    """Проверка наличия и доступа к AlignmentEngine."""
    assert hasattr(AlignmentEngine, "__init__"), "AlignmentEngine не имеет метода __init__"

def test_session_logger():
    """Проверка наличия и доступа к SessionLogger."""
    assert hasattr(SessionLogger, "__init__"), "SessionLogger не имеет метода __init__"

def test_user_interface():
    """Проверка наличия и доступа к UserInterface."""
    assert hasattr(UserInterface, "__init__"), "UserInterface не имеет метода __init__"

if __name__ == "__main__":
    pytest.main(["-v", __file__])