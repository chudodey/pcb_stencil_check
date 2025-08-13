import os
import sys
from pathlib import Path
from dataclasses import asdict
from contextlib import contextmanager
import numpy as np
import pytest

# Добавляем путь к модулям в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))

from utils import SessionLogger, ConfigManager, UserInterface, FileHandler

# Путь к директории с тестами
TEST_DIR = os.path.dirname(os.path.abspath(__file__))

@contextmanager
def chdir(path: Path):
    """Контекстный менеджер для временной смены рабочей директории."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

@pytest.fixture
def tmp_proj(tmp_path: Path):
    """Фикстура для создания временной структуры проекта."""
    proj = tmp_path / "proj"
    proj.mkdir()
    with chdir(proj):
        yield proj

def test_filehandler_ensure_dir_creates_directory(tmp_proj: Path):
    """Тестирование создания директории через FileHandler."""
    d = tmp_proj / "results"
    res = FileHandler.ensure_dir(d, as_dir=True)
    assert res.created is True
    assert res.dir_path == d
    assert d.is_dir()

def test_filehandler_yaml_read_write(tmp_proj: Path):
    """Тестирование чтения/записи YAML файлов."""
    p = tmp_proj / "test_config.yaml"
    test_data = {"param1": 123, "param2": "value"}
    FileHandler.write_yaml(p, test_data)
    loaded = FileHandler.read_yaml(p)
    assert loaded == test_data

def test_config_manager_initialization(tmp_proj: Path):
    """Тестирование инициализации ConfigManager."""
    cfg_path = tmp_proj / "config.txt"
    cfg = ConfigManager(str(cfg_path))
    assert cfg_path.exists()
    assert cfg.output_dir.is_dir()
    assert cfg.log_dir.is_dir()

def test_session_logger_lifecycle(tmp_proj: Path):
    """Тестирование жизненного цикла SessionLogger."""
    cfg_path = tmp_proj / "config.txt"
    cfg = ConfigManager(str(cfg_path))
    
    # Создаем необходимые директории
    (tmp_proj / "test_data" / "gerber").mkdir(parents=True)
    
    logger = SessionLogger(cfg)
    logger.start_session()
    logger.set_operator("Тестов Т.Т.")
    
    # Логирование результатов сканирования
    logger.log_scan_result(
        order_id="123456",
        scan_path=cfg.scan_dir / "scan1.png",
        alignment_result={
            "rotation": 1.23,
            "shift_x": 0.12,
            "shift_y": -0.22,
            "match_percentage": 96.7,
            "correlation_peak": 0.88,
        }
    )
    
    # Проверка создания файла лога
    log_path = cfg.output_dir / "123456" / "order_log.yaml"
    assert log_path.exists()
    
    logger.end_session()
    session_logs = list(cfg.log_dir.glob("session_*.yaml"))
    assert len(session_logs) > 0

def test_user_interface_input_validation(monkeypatch):
    """Тестирование валидации ввода в UserInterface."""
    ui = UserInterface()
    
    # Тестирование ввода номера заказа
    monkeypatch.setattr('builtins.input', lambda _: "000001")
    assert ui.get_order_number() == "000001"
    
    # Тестирование ввода имени оператора
    monkeypatch.setattr('builtins.input', lambda _: "Иванов И.И.")
    assert ui.get_operator_name("Default") == "Иванов И.И."

def test_image_io_operations(tmp_proj: Path):
    """Тестирование операций с изображениями."""
    img_path = tmp_proj / "test_image.png"
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Сохранение тестового изображения
    assert FileHandler.write_image(str(img_path), test_img)
    
    # Чтение изображения
    loaded_img = FileHandler.read_image(str(img_path))
    assert loaded_img.shape == (100, 100)