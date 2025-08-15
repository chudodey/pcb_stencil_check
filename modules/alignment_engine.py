import cv2
import numpy as np
from typing import Dict, Tuple, List
import math
import logging
from datetime import datetime
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

class AlignmentEngine:
    """
    Комбинированный выравниватель изображений.
    Версия 5.0 - объединяет два подхода для максимальной надежности.
    """

    def __init__(self, dpi: int = 600, ransac_threshold: float = 1.5,
                 min_contour_area: int = 10, debug: bool = False):
        self.params = {
            'dpi': dpi,
            'ransac_threshold': ransac_threshold,
            'min_contour_area': min_contour_area
        }

        # Настройка логирования
        self.debug = debug
        self.logger = logging.getLogger('CombinedAlignmentEngine')
        self.logger.setLevel(logging.DEBUG if debug else logging.WARNING)
        
        if debug:
            if not self.logger.handlers:
                ch = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
            
            self.logger.info(f"Инициализация движка с параметрами: {self.params}")

    def _transform_image_simple(self, image: np.ndarray, rotate: int = 0, flip: int = None) -> np.ndarray:
        """Простое преобразование изображения (для подхода 1)."""
        result = image.copy()
        
        # Поворот
        if rotate == 90:
            result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotate == -90:
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            result = cv2.rotate(result, cv2.ROTATE_180)
        
        # Отражение (если указано)
        if flip is not None:
            result = cv2.flip(result, flip)
        
        return result

    def _transform_image_matrix(self, image: np.ndarray, rotate: int = 0, flip: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Преобразование с возвратом матрицы (для подхода 2)."""
        h, w = image.shape
        M_flip = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        if flip is not None:
            if flip == 1:  # горизонтальное отражение
                M_flip = np.array([[-1, 0, w-1], [0, 1, 0]], dtype=np.float32)
            elif flip == 0:  # вертикальное отражение
                M_flip = np.array([[1, 0, 0], [0, -1, h-1]], dtype=np.float32)

        M_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), -rotate, 1.0)
        
        # Объединяем матрицы
        M_total = M_rotate @ np.vstack([M_flip, [0, 0, 1]])
        M_total = M_total[:2, :]
        
        result = cv2.warpAffine(image, M_total, (w, h))
        
        return result, M_total

    def _get_centroids(self, contours: List[np.ndarray]) -> np.ndarray:
        """Вычисление центроидов контуров."""
        centers = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centers.append([cx, cy])
        return np.array(centers, dtype=np.float32)

    def _extract_contours_and_centroids(self, image: np.ndarray, name: str = "") -> Tuple[List, np.ndarray]:
        """Извлечение контуров и центроидов из бинарного изображения."""
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Фильтрация по площади
        h, w = image.shape
        max_area = h * w * 0.1
        min_area = self.params['min_contour_area']
        
        valid_contours = [
            cnt for cnt in contours 
            if min_area < cv2.contourArea(cnt) < max_area
        ]
        
        centroids = self._get_centroids(valid_contours)
        
        if self.debug:
            self.logger.debug(
                f"{name}: найдено {len(contours)} контуров, "
                f"валидных {len(valid_contours)}, центроидов {len(centroids)}"
            )
            
        return valid_contours, centroids

    def _match_and_estimate(self, scan_centroids: np.ndarray, ref_centroids: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """Сопоставление точек и оценка преобразования."""
        if len(scan_centroids) == 0 or len(ref_centroids) == 0:
            return None, 0, float('inf')
        
        # Сопоставление точек с помощью KD-дерева
        tree = cKDTree(ref_centroids)
        distances, indices = tree.query(scan_centroids, k=1)
        
        matched_ref = ref_centroids[indices]
        matched_scan = scan_centroids
        
        # RANSAC
        try:
            affine_matrix, inliers = cv2.estimateAffinePartial2D(
                matched_scan,
                matched_ref,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.params['ransac_threshold'],
                maxIters=5000,
                confidence=0.999,
                refineIters=20
            )
            
            if affine_matrix is not None and inliers is not None:
                inliers_count = np.sum(inliers)
                
                # Вычисление средней ошибки для инлайнеров
                if inliers_count > 0:
                    inlier_scan = matched_scan[inliers.flatten().astype(bool)]
                    inlier_ref = matched_ref[inliers.flatten().astype(bool)]
                    
                    # Применяем преобразование к инлайнерам скана
                    transformed = cv2.transform(
                        inlier_scan.reshape(-1, 1, 2), 
                        affine_matrix
                    ).reshape(-1, 2)
                    
                    # Вычисляем среднюю ошибку
                    errors = np.linalg.norm(transformed - inlier_ref, axis=1)
                    mean_error = np.mean(errors)
                else:
                    mean_error = float('inf')
                
                return affine_matrix, inliers_count, mean_error
            else:
                return None, 0, float('inf')
                
        except Exception as e:
            if self.debug:
                self.logger.warning(f"Ошибка RANSAC: {str(e)}")
            return None, 0, float('inf')

    def _approach_1_transform_scan(self, reference: np.ndarray, scan: np.ndarray) -> Dict:
        """
        Подход 1: Трансформируем скан (как в версии 3.0).
        """
        if self.debug:
            self.logger.info("Подход 1: Трансформация скана")
        
        # Извлечение контуров эталона (один раз)
        ref_contours, ref_centroids = self._extract_contours_and_centroids(reference, "Эталон")
        
        if len(ref_centroids) == 0:
            return None

        transformations = [
            (0, None, "0°"),
            (90, None, "90°"),
            (180, None, "180°"), 
            (-90, None, "270°"),
            (0, 1, "0°+flip_h"),
            (90, 1, "90°+flip_h"),
            (180, 1, "180°+flip_h"),
            (-90, 1, "270°+flip_h"),
        ]

        best_result = {
            'matrix': None,
            'orientation': None,
            'inliers': -1,
            'error': float('inf'),
            'correlation': -1,
            'transformed_scan': None,
            'approach': 1
        }

        for rotate, flip, orientation_name in transformations:
            try:
                # Трансформация скана
                scan_transformed = self._transform_image_simple(scan, rotate=rotate, flip=flip)
                
                # Извлечение контуров из трансформированного скана
                scan_contours, scan_centroids = self._extract_contours_and_centroids(
                    scan_transformed, f"Скан ({orientation_name})"
                )
                
                if len(scan_centroids) < 3:
                    continue

                # RANSAC
                affine_matrix, inliers_count, mean_error = self._match_and_estimate(
                    scan_centroids, ref_centroids
                )
                
                if affine_matrix is None or inliers_count < 3:
                    continue

                # Финальное выравнивание
                height, width = reference.shape
                aligned = cv2.warpAffine(scan_transformed, affine_matrix, (width, height))
                
                # Корреляция
                correlation = pearsonr(reference.flatten(), aligned.flatten())[0]
                
                if self.debug:
                    self.logger.debug(
                        f"Подход 1, {orientation_name}: инлайнеров={inliers_count}, "
                        f"ошибка={mean_error:.2f}, корреляция={correlation:.4f}"
                    )

                # Обновление лучшего результата
                is_better = (
                    inliers_count > best_result['inliers'] or
                    (inliers_count == best_result['inliers'] and mean_error < best_result['error']) or
                    (inliers_count == best_result['inliers'] and mean_error == best_result['error'] and correlation > best_result['correlation'])
                )
                
                if is_better:
                    best_result.update({
                        'matrix': affine_matrix,
                        'orientation': orientation_name,
                        'inliers': inliers_count,
                        'error': mean_error,
                        'correlation': correlation,
                        'transformed_scan': scan_transformed,
                        'result_image': aligned
                    })

            except Exception as e:
                if self.debug:
                    self.logger.error(f"Подход 1, ошибка {orientation_name}: {str(e)}")
                continue

        return best_result if best_result['matrix'] is not None else None

    def _approach_2_transform_reference(self, reference: np.ndarray, scan: np.ndarray) -> Dict:
        """
        Подход 2: Трансформируем эталон (как в версии 4.0, но исправленный).
        """
        if self.debug:
            self.logger.info("Подход 2: Трансформация эталона")
        
        # Извлечение контуров скана (один раз)
        scan_contours, scan_centroids = self._extract_contours_and_centroids(scan, "Скан")
        
        if len(scan_centroids) == 0:
            return None

        transformations = [
            (0, None, "0°"),
            (90, None, "90°"),
            (180, None, "180°"), 
            (-90, None, "270°"),
            (0, 1, "0°+flip_h"),
            (90, 1, "90°+flip_h"),
            (180, 1, "180°+flip_h"),
            (-90, 1, "270°+flip_h"),
        ]

        best_result = {
            'matrix': None,
            'orientation': None,
            'inliers': -1,
            'error': float('inf'),
            'correlation': -1,
            'ref_transformed': None,
            'ref_transform_matrix': None,
            'approach': 2
        }

        for rotate, flip, orientation_name in transformations:
            try:
                # Трансформация эталона с получением матрицы
                ref_transformed, ref_transform_matrix = self._transform_image_matrix(
                    reference, rotate=rotate, flip=flip
                )
                
                # Извлечение контуров из трансформированного эталона
                ref_contours, ref_centroids = self._extract_contours_and_centroids(
                    ref_transformed, f"Эталон ({orientation_name})"
                )
                
                if len(ref_centroids) < 3:
                    continue

                # RANSAC: скан → трансформированный эталон
                affine_matrix, inliers_count, mean_error = self._match_and_estimate(
                    scan_centroids, ref_centroids
                )
                
                if affine_matrix is None or inliers_count < 3:
                    continue

                # Применение к скану для проверки корреляции
                height, width = ref_transformed.shape
                scan_aligned = cv2.warpAffine(scan, affine_matrix, (width, height))
                
                # Корреляция
                correlation = pearsonr(ref_transformed.flatten(), scan_aligned.flatten())[0]
                
                if self.debug:
                    self.logger.debug(
                        f"Подход 2, {orientation_name}: инлайнеров={inliers_count}, "
                        f"ошибка={mean_error:.2f}, корреляция={correlation:.4f}"
                    )

                # Обновление лучшего результата
                is_better = (
                    inliers_count > best_result['inliers'] or
                    (inliers_count == best_result['inliers'] and mean_error < best_result['error']) or
                    (inliers_count == best_result['inliers'] and mean_error == best_result['error'] and correlation > best_result['correlation'])
                )
                
                if is_better:
                    best_result.update({
                        'matrix': affine_matrix,
                        'orientation': orientation_name,
                        'inliers': inliers_count,
                        'error': mean_error,
                        'correlation': correlation,
                        'ref_transformed': ref_transformed,
                        'ref_transform_matrix': ref_transform_matrix,
                        'scan_aligned': scan_aligned
                    })

            except Exception as e:
                if self.debug:
                    self.logger.error(f"Подход 2, ошибка {orientation_name}: {str(e)}")
                continue

        # Если найдено решение, вычисляем финальную матрицу
        if best_result['matrix'] is not None:
            try:
                # Исправленная композиция матриц
                affine_matrix = best_result['matrix']
                ref_transform_matrix = best_result['ref_transform_matrix']
                
                # Вычисляем обратную матрицу для трансформации эталона
                ref_inverse_matrix = cv2.invertAffineTransform(ref_transform_matrix)
                
                # Комбинируем матрицы: сначала RANSAC, потом обратная трансформация эталона
                final_matrix = ref_inverse_matrix @ np.vstack([affine_matrix, [0, 0, 1]])
                final_matrix = final_matrix[:2, :]
                
                # Применяем к исходному скану
                height, width = reference.shape
                result_image = cv2.warpAffine(scan, final_matrix, (width, height))
                
                best_result['result_image'] = result_image
                best_result['final_matrix'] = final_matrix
                
            except Exception as e:
                if self.debug:
                    self.logger.error(f"Ошибка композиции матриц: {str(e)}")
                return None

        return best_result if best_result['matrix'] is not None else None

    def align(self, reference: np.ndarray, scan: np.ndarray) -> Dict:
        """
        Главный метод выравнивания.
        Пробует оба подхода и возвращает лучший результат.
        """
        if self.debug:
            self.logger.info(f"Начало комбинированного выравнивания. Размеры: эталон {reference.shape}, скан {scan.shape}")

        # Пробуем подход 1: трансформация скана
        result_1 = self._approach_1_transform_scan(reference, scan)
        
        # Пробуем подход 2: трансформация эталона  
        result_2 = self._approach_2_transform_reference(reference, scan)

        # Выбираем лучший результат
        candidates = [r for r in [result_1, result_2] if r is not None]
        
        if not candidates:
            raise ValueError("Оба подхода не смогли найти решение")

        # Сравниваем кандидатов
        best_result = candidates[0]
        for candidate in candidates[1:]:
            is_better = (
                candidate['inliers'] > best_result['inliers'] or
                (candidate['inliers'] == best_result['inliers'] and candidate['error'] < best_result['error']) or
                (candidate['inliers'] == best_result['inliers'] and candidate['error'] == best_result['error'] and candidate['correlation'] > best_result['correlation'])
            )
            
            if is_better:
                best_result = candidate

        if self.debug:
            self.logger.info(
                f"Лучший результат: подход {best_result['approach']}, "
                f"ориентация {best_result['orientation']}, "
                f"инлайнеров={best_result['inliers']}, "
                f"ошибка={best_result['error']:.2f}, "
                f"корреляция={best_result['correlation']:.4f}"
            )

        return {
            'result_image': best_result['result_image'],
            'metrics': {
                'inliers': best_result['inliers'],
                'matches': len(scan) if best_result['approach'] == 2 else len(reference),  # упрощенно
                'error': best_result['error'],
                'correlation': best_result['correlation'],
                'orientation': best_result['orientation'],
                'approach_used': best_result['approach']
            },
            'debug_info': {
                'matrix': best_result['matrix'].tolist(),
                'approach_used': best_result['approach'],
                'candidates_tried': len(candidates)
            }
        }