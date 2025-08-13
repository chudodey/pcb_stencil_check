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

# Path to test directory
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_INPUTS_DIR = os.path.join(TEST_DIR, "test_inputs")
TEST_OUTPUTS_DIR = os.path.join(TEST_DIR, "test_outputs")

def ensure_dir(path: str):
    """Создаёт папку, если её нет."""
    os.makedirs(path, exist_ok=True)

ensure_dir(TEST_INPUTS_DIR)
ensure_dir(TEST_OUTPUTS_DIR)

@pytest.fixture
def gerber_content():
    """Fixture with Gerber file content."""
    gerber_path = os.path.join(TEST_INPUTS_DIR, 'test_board1.gbr')
    if not os.path.exists(gerber_path):
        pytest.skip(f"Gerber test file not found at {gerber_path}")
    with open(gerber_path, 'r') as f:
        content = f.read()
    if not content.strip():
        pytest.skip("Gerber test file is empty")
    return content

@pytest.fixture
def scan_image():
    """Fixture with scan image."""
    scan_path = os.path.join(TEST_INPUTS_DIR, 'test_scan1.jpg')
    if not os.path.exists(scan_path):
        pytest.skip(f"Scan image not found at {scan_path}")
    
    # Надежный способ чтения файла с любым путем
    try:
        with open(scan_path, 'rb') as f:
            file_bytes = np.fromfile(f, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception:
        img = None # В случае любой ошибки чтения/декодирования

    if img is None:
        pytest.skip("Failed to read scan image (possibly corrupt or unsupported format)")
    return img

def test_gerber_processing(gerber_content):
    """Test Gerber file processing."""
    print(f"Gerber content length: {len(gerber_content)}")  # Debug output
    
    processor = GerberProcessor(dpi=600)
    result = processor.parse(gerber_content)

    print(f"Result keys: {result.keys()}")  # Debug output
    
    # Check basic result structure
    assert 'result_image' in result
    assert 'metrics' in result
    assert 'params' in result
    assert 'contours' in result
    assert 'apertures' in result
    assert 'debug_info' in result
    
    # Check image type and size
    assert isinstance(result['result_image'], np.ndarray)
    assert result['result_image'].dtype == np.uint8
    assert result['result_image'].size > 0
    
    # Check metrics
    assert result['metrics']['board_width_mm'] > 0
    assert result['metrics']['board_height_mm'] > 0
    assert result['metrics']['contour_count'] > 0
    assert result['metrics']['aperture_count'] > 0
    
    # Save image for verification (robust method)
    output_path = os.path.join(TEST_OUTPUTS_DIR, 'gerber_output.png')
    success, encoded_image = cv2.imencode('.png', result['result_image'])
    if success:
        with open(output_path, 'wb') as f:
            f.write(encoded_image)
    assert os.path.exists(output_path)

def test_image_processing(scan_image):
    """Test scan image processing."""
    print(f"Scan image shape: {scan_image.shape}")  # Debug output
    
    processor = ImageProcessor(dpi=600,
                                binary_threshold=200,
                                crop_min_area = 100.0,        
                                crop_max_area_ratio = 0.1)
    result = processor.process(scan_image)
    
    # Check basic result structure
    assert 'result_image' in result
    assert 'source_image' in result
    assert 'metrics' in result
    assert 'params' in result
    assert 'debug_info' in result
    
    # Check images
    assert isinstance(result['result_image'], np.ndarray)
    assert result['result_image'].dtype == np.uint8
    assert result['result_image'].shape[0] > 0
    assert result['result_image'].shape[1] > 0
    
    # Check metrics
    assert 'processed_size' in result['metrics']
    assert 'contour_count' in result['metrics']
    assert result['metrics']['contour_count'] > 0

    # Save image for verification (robust method)
    output_path = os.path.join(TEST_OUTPUTS_DIR, 'processed_scan.png')
    success, encoded_image = cv2.imencode('.png', result['result_image'])
    if success:
        with open(output_path, 'wb') as f:
            f.write(encoded_image)
    assert os.path.exists(output_path)    

def test_alignment(gerber_content, scan_image):
    """Test image alignment with numpy type support."""
    # Process Gerber
    gerber_processor = GerberProcessor(dpi=600)
    gerber_result = gerber_processor.parse(gerber_content)
    gerber_image = gerber_result['result_image']
    
    # Process scan
    img_processor = ImageProcessor(dpi=600,
                                binary_threshold=200,
                                crop_min_area=100.0,        
                                crop_max_area_ratio=0.1)
    scan_result = img_processor.process(scan_image)
    processed_scan = scan_result['result_image']
    
    # Alignment with debug mode
    aligner = AlignmentEngine(dpi=600, debug=True)
    result = aligner.align(gerber_image, processed_scan)
    
    # 1. Validate result structure
    required_keys = ['result_image', 'metrics', 'debug_info']
    for key in required_keys:
        assert key in result, f"Missing key in result: {key}"
    
    # 2. Validate metrics
    required_metrics = ['inliers', 'matches', 'error', 'correlation', 'orientation']
    for metric in required_metrics:
        assert metric in result['metrics'], f"Missing metric: {metric}"
    
    # 3. Validate values with numpy type support
    assert isinstance(result['metrics']['correlation'], (float, np.floating)), \
        f"Correlation should be float, got {type(result['metrics']['correlation'])}"
    
    correlation = float(result['metrics']['correlation'])  # Convert to Python float
    assert -1.0 <= correlation <= 1.0, f"Correlation {correlation} out of range [-1, 1]"
    
    assert isinstance(result['metrics']['inliers'], (int, np.integer)), "Inliers should be integer"
    assert result['metrics']['inliers'] >= 0, "Inliers count cannot be negative"
    
    assert isinstance(result['metrics']['matches'], (int, np.integer)), "Matches should be integer"
    assert result['metrics']['matches'] >= 0, "Matches count cannot be negative"
    
    assert isinstance(result['metrics']['error'], (float, np.floating)), "Error should be float"
    assert result['metrics']['error'] >= 0, "Error cannot be negative"
    
    # 4. File saving with robust checks
    try:
        # Ensure directory exists
        os.makedirs(TEST_OUTPUTS_DIR, exist_ok=True)
        
        # Save aligned image (robust method)
        aligned_path = os.path.join(TEST_OUTPUTS_DIR, 'aligned_result.png')
        success, encoded_image = cv2.imencode('.png', result['result_image'])
        if not success:
            raise IOError("Failed to encode aligned image")
        with open(aligned_path, 'wb') as f:
            f.write(encoded_image.tobytes())
        assert os.path.exists(aligned_path), f"File {aligned_path} was not created"
        
        # Save difference image (robust method)
        diff_path = os.path.join(TEST_OUTPUTS_DIR, 'difference.png')
        diff = cv2.absdiff(gerber_image, result['result_image'])
        success, encoded_diff = cv2.imencode('.png', diff)
        if not success:
            raise IOError("Failed to encode difference image")
        with open(diff_path, 'wb') as f:
            f.write(encoded_diff.tobytes())
        assert os.path.exists(diff_path), f"File {diff_path} was not created"
        
        # Save debug info
        debug_path = os.path.join(TEST_OUTPUTS_DIR, 'debug_info.json')
        debug_info = {
            'metrics': {
                'inliers': int(result['metrics']['inliers']),
                'matches': int(result['metrics']['matches']),
                'error': float(result['metrics']['error']),
                'correlation': float(result['metrics']['correlation']),
                'orientation': result['metrics']['orientation']
            },
            'debug_info': result['debug_info'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(debug_path, 'w') as f:
            json.dump(debug_info, f, indent=2)
        assert os.path.exists(debug_path), f"File {debug_path} was not created"
            
    except Exception as e:
        # Provide detailed error message
        error_msg = [
            "Failed to save test results:",
            f"- Error: {str(e)}",
            f"- Current working directory: {os.getcwd()}",
            f"- Test outputs directory: {TEST_OUTPUTS_DIR}",
            f"- Directory exists: {os.path.exists(TEST_OUTPUTS_DIR)}",
            f"- Directory writable: {os.access(TEST_OUTPUTS_DIR, os.W_OK)}",
            f"- Image shape: {result['result_image'].shape if hasattr(result['result_image'], 'shape') else 'N/A'}",
            f"- Image dtype: {result['result_image'].dtype if hasattr(result['result_image'], 'dtype') else 'N/A'}"
        ]
        pytest.fail("\n".join(error_msg)) 

def test_edge_cases():
    """Test edge cases."""
    # Empty Gerber
    empty_processor = GerberProcessor()
    empty_result = empty_processor.parse("")
    assert empty_result['metrics']['contour_count'] == 0
    
    # Empty image
    img_processor = ImageProcessor()
    empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
    empty_result = img_processor.process(empty_img)
    assert empty_result['metrics']['contour_count'] == 0
    
    # Aligning empty images
    aligner = AlignmentEngine()
    empty_img1 = np.zeros((100, 100), dtype=np.uint8)
    empty_img2 = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        aligner.align(empty_img1, empty_img2)