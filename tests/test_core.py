import os
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path
from datetime import datetime
import json

# Add path to modules in PYTHONPATH
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from modules import GerberProcessor, ImageProcessor, AlignmentEngine

# Path to test directories
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_INPUTS_DIR = os.path.join(TEST_DIR, "test_inputs")
TEST_OUTPUTS_DIR = os.path.join(TEST_DIR, "test_outputs")

def ensure_dir(path: str):
    """Создаёт папку, если её нет."""
    os.makedirs(path, exist_ok=True)

ensure_dir(TEST_INPUTS_DIR)
ensure_dir(TEST_OUTPUTS_DIR)

def get_test_pairs():
    """Генерирует пары (gerber_file, scan_file) для тестирования."""
    gerber_files = sorted([f for f in os.listdir(TEST_INPUTS_DIR) 
                         if f.startswith('test_board') and f.endswith('.gbr')],
                         key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    scan_files = sorted([f for f in os.listdir(TEST_INPUTS_DIR) 
                       if f.startswith('test_scan') and f.endswith('.jpg')],
                       key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    for gerber, scan in zip(gerber_files, scan_files):
        # Извлекаем номер проекта
        project_num = int(''.join(filter(str.isdigit, Path(gerber).stem)))
        yield (gerber, scan, f"project{project_num}")

def load_gerber_content(gerber_filename):
    """Загружает содержимое Gerber файла."""
    gerber_path = os.path.join(TEST_INPUTS_DIR, gerber_filename)
    if not os.path.exists(gerber_path):
        pytest.skip(f"Gerber test file not found at {gerber_path}")
    with open(gerber_path, 'r') as f:
        content = f.read()
    if not content.strip():
        pytest.skip(f"Gerber test file {gerber_filename} is empty")
    return content

def load_scan_image(scan_filename):
    """Загружает scan изображение."""
    scan_path = os.path.join(TEST_INPUTS_DIR, scan_filename)
    if not os.path.exists(scan_path):
        pytest.skip(f"Scan image not found at {scan_path}")
    
    try:
        with open(scan_path, 'rb') as f:
            file_bytes = np.fromfile(f, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        pytest.skip(f"Failed to read scan image {scan_filename}: {str(e)}")

    if img is None:
        pytest.skip(f"Failed to decode scan image {scan_filename}")
    return img

def create_combined_image(gerber_img, scan_img):
    """Создает комбинированное изображение (gerber в синем канале, scan в красном+зеленом)."""
    # Конвертируем в grayscale если нужно
    if len(gerber_img.shape) == 3:
        gerber_img = cv2.cvtColor(gerber_img, cv2.COLOR_BGR2GRAY)
    if len(scan_img.shape) == 3:
        scan_img = cv2.cvtColor(scan_img, cv2.COLOR_BGR2GRAY)
    
    # Нормализуем изображения
    gerber_norm = cv2.normalize(gerber_img, None, 0, 255, cv2.NORM_MINMAX)
    scan_norm = cv2.normalize(scan_img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Создаем 3-канальное изображение
    combined = np.zeros((gerber_img.shape[0], gerber_img.shape[1], 3), dtype=np.uint8)
    combined[:,:,0] = scan_norm  # Красный канал - скан
    combined[:,:,1] = scan_norm  # Зеленый канал - скан
    combined[:,:,2] = gerber_norm  # Синий канал - gerber
    
    return combined

def save_project_files(project_dir, files_to_save):
    """Сохраняет файлы проекта с нумерацией."""
    ensure_dir(project_dir)
    
    saved_files = {}
    for idx, (name_suffix, data, extension) in enumerate(files_to_save, 1):
        filename = f"{idx}_{name_suffix}.{extension}"
        filepath = os.path.join(project_dir, filename)
        
        if extension == 'png':
            success, encoded = cv2.imencode('.png', data)
            if success:
                with open(filepath, 'wb') as f:
                    f.write(encoded.tobytes())
                saved_files[name_suffix] = filename
        elif extension == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            saved_files['metrics'] = filename
    
    return saved_files



def test_processing_sequence():
    """Основной тест обработки проектов."""
    all_metrics = {}
    
    for gerber_file, scan_file, project_name in get_test_pairs():
        project_num = ''.join(filter(str.isdigit, project_name))
        project_dir = os.path.join(TEST_OUTPUTS_DIR, project_name)
        
        print(f"\n=== Обработка {project_name} ===")
        print(f"Gerber: {gerber_file}, Scan: {scan_file}")
        
        # Загрузка данных
        gerber_content = load_gerber_content(gerber_file)
        scan_image = load_scan_image(scan_file)
        scan_flipped = cv2.flip(scan_image, 1)  # Добавляем отраженный скан
        
        # Обработка Gerber
        print("1. Обработка Gerber файла...")
        gerber_processor = GerberProcessor(dpi=600)
        gerber_result = gerber_processor.parse(gerber_content)
        gerber_img = gerber_result['result_image']
        
        # Обработка оригинального скана
        print("2. Обработка оригинального скана...")
        img_processor = ImageProcessor(
            dpi=600,
            binary_threshold=200,
            crop_min_area=100.0,
            crop_max_area_ratio=0.1
        )
        scan_result = img_processor.process(scan_image)
        scan_img = scan_result['result_image']
        
        # Обработка отраженного скана
        print("3. Обработка отраженного скана...")
        scan_flipped_result = img_processor.process(scan_flipped)
        scan_flipped_img = scan_flipped_result['result_image']
        
        # Выравнивание оригинального скана
        print("4. Выравнивание оригинального скана...")
        aligner = AlignmentEngine(dpi=600, debug=True, ransac_threshold=3.0, min_contour_area=5)
        alignment_result = aligner.align(gerber_img, scan_img)
        aligned_img = alignment_result['result_image']
        
        # Выравнивание отраженного скана
        print("5. Выравнивание отраженного скана...")
        alignment_flipped_result = aligner.align(gerber_img, scan_flipped_img)
        aligned_flipped_img = alignment_flipped_result['result_image']
        
        # Создание комбинированных изображений
        print("6. Создание комбинированных изображений...")
        combined_img = create_combined_image(gerber_img, aligned_img)
        combined_flipped_img = create_combined_image(gerber_img, aligned_flipped_img)
        
        # # Подготовка метрик
        # metrics = {
        #     'project': project_name,
        #     'gerber_file': gerber_file,
        #     'scan_file': scan_file,
        #     'timestamp': datetime.now().isoformat(),
        #     'metrics': {
        #         'original': {
        #             'inliers': int(alignment_result['metrics']['inliers']),
        #             # 'matches': int(alignment_result['metrics']['matches']),
        #             'error': float(alignment_result['metrics']['error']),
        #             'correlation': float(alignment_result['metrics']['correlation']),
        #             'orientation': alignment_result['metrics']['orientation']
        #         },
        #         'flipped': {
        #             'inliers': int(alignment_flipped_result['metrics']['inliers']),
        #             # 'matches': int(alignment_flipped_result['metrics']['matches']),
        #             'error': float(alignment_flipped_result['metrics']['error']),
        #             'correlation': float(alignment_flipped_result['metrics']['correlation']),
        #             'orientation': alignment_flipped_result['metrics']['orientation']
        #         }
        #     }
        # }
        # all_metrics[project_name] = metrics
        
        # Сохранение файлов проекта
        files_to_save = [
            ('gerber', gerber_img, 'png'),
            ('scan', scan_img, 'png'),
            ('compared', combined_img, 'png'),
            ('compared_flipped', combined_flipped_img, 'png')
            # ('metrics', metrics, 'json')
        ]
        
        saved_files = save_project_files(project_dir, files_to_save)
        print(f"Сохраненные файлы: {saved_files}")
        
        # Проверка метрик
        # assert metrics['metrics']['original']['correlation'] > 0.5 or \
        #        metrics['metrics']['flipped']['correlation'] > 0.5, \
        #        f"Низкая корреляция для {project_name}"
    
    # Сохранение объединенных метрик
    summary_path = os.path.join(TEST_OUTPUTS_DIR, "all_metrics.json")
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n=== Все проекты обработаны. Результаты сохранены в {TEST_OUTPUTS_DIR} ===")
    print(f"Общие метрики сохранены в: {summary_path}")

# def test_edge_cases():
#     """Test edge cases."""
#     # Empty Gerber
#     empty_processor = GerberProcessor()
#     empty_result = empty_processor.parse("")
#     assert empty_result['metrics']['contour_count'] == 0
    
#     # Empty image
#     img_processor = ImageProcessor()
#     empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
#     empty_result = img_processor.process(empty_img)
#     assert empty_result['metrics']['contour_count'] == 0
    
#     # Aligning empty images
#     aligner = AlignmentEngine()
#     empty_img1 = np.zeros((100, 100), dtype=np.uint8)
#     empty_img2 = np.zeros((100, 100), dtype=np.uint8)
#     with pytest.raises(ValueError):
#         aligner.align(empty_img1, empty_img2)