import cv2
import numpy as np
from typing import Dict, Tuple, List
import math
import logging
from datetime import datetime
from sklearn.neighbors import KDTree

class AlignmentEngine:
    """
    Оптимизированный выравниватель изображений с исправленными ошибками.
    Версия 2.1 с исправлениями на основе анализа логов.
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
        self.logger = logging.getLogger('AlignmentEngine')
        self.logger.setLevel(logging.DEBUG if debug else logging.WARNING)
        
        if debug:
            if not self.logger.handlers:
                ch = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
            
            self.logger.info(f"Инициализация движка с параметрами: {self.params}")

        self._init_transforms()

    def _init_transforms(self) -> None:
        """Инициализация преобразований с исправленными матрицами."""
        self._orientations = {
            '0°': None,
            '90°': cv2.ROTATE_90_CLOCKWISE,
            '180°': cv2.ROTATE_180,
            '270°': cv2.ROTATE_90_COUNTERCLOCKWISE,
            'flip_h': lambda img: cv2.flip(img, 1),
            'flip_v': lambda img: cv2.flip(img, 0)
        }

    def _get_contours(self, image: np.ndarray, name: str = "Изображение") -> Tuple[np.ndarray, np.ndarray]:
        """Извлечение контуров с исправленной обработкой моментов."""
        start_time = datetime.now()
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        centroids = []
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.params['min_contour_area']:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    centroids.append([cx, cy])
                    areas.append(area)

        if self.debug:
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.debug(
                f"{name}: найдено {len(contours)} контуров, "
                f"оставлено {len(centroids)} после фильтрации "
                f"({elapsed:.2f} мс)"
            )
            if len(centroids) > 0:
                self.logger.debug(
                    f"{name}: площади от {np.min(areas):.1f} до {np.max(areas):.1f}"
                )
                
        return np.array(centroids), np.array(areas)

    def _apply_orientation(self, points: np.ndarray, orientation: str, 
                         img_shape: Tuple[int, int]) -> np.ndarray:
        """Применение ориентации к точкам с исправленной логикой."""
        if orientation == '0°' or points.size == 0:
            return points.copy()
            
        if orientation == 'flip_h':
            points[:, 0] = img_shape[1] - points[:, 0]
            return points
        elif orientation == 'flip_v':
            points[:, 1] = img_shape[0] - points[:, 1]
            return points
            
        # Для поворотов используем матрицы преобразований
        center = (img_shape[1]/2, img_shape[0]/2)
        angle = float(orientation[:-1])
        
        if orientation == '90°':
            M = cv2.getRotationMatrix2D(center, -90, 1.0)
        elif orientation == '180°':
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
        elif orientation == '270°':
            M = cv2.getRotationMatrix2D(center, 90, 1.0)
        else:
            return points.copy()
            
        # Преобразование точек
        homogeneous = np.column_stack([points, np.ones(len(points))])
        return (M @ homogeneous.T).T

    def _match_points(self, ref_points: np.ndarray, scan_points: np.ndarray) -> np.ndarray:
        """Улучшенный матчинг точек с автоматическим подбором порога."""
        if len(scan_points) == 0 or len(ref_points) == 0:
            return np.empty((0, 2), dtype=int)

        tree = KDTree(ref_points)
        distances, indices = tree.query(scan_points, k=1)
        
        # Автоподбор порога на основе медианного расстояния
        median_dist = np.median(distances)
        max_dist = max(median_dist * 2, 10.0)  # Минимальный порог 10px
        
        valid = distances[:, 0] < max_dist
        matches = np.column_stack([
            indices[valid, 0],
            np.arange(len(scan_points))[valid]
        ])

        if self.debug:
            self.logger.debug(
                f"Матчинг: {len(matches)} соответствий из {len(scan_points)} "
                f"(порог расстояния {max_dist:.1f} px)"
            )

        return matches

    def _estimate_transform(self, src_points: np.ndarray, dst_points: np.ndarray) -> Tuple[np.ndarray, int]:
        """Оценка преобразования с улучшенной обработкой ошибок."""
        if len(src_points) < 3:
            return None, 0

        try:
            M, inliers = cv2.estimateAffine2D(
                src_points, dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.params['ransac_threshold'],
                confidence=0.999,
                maxIters=5000
            )
            return M, np.sum(inliers) if inliers is not None else 0
        except cv2.error as e:
            if self.debug:
                self.logger.warning(f"Ошибка RANSAC: {str(e)}")
            return None, 0

    def align(self, reference: np.ndarray, scan: np.ndarray) -> Dict:
        """Основной метод с исправленной обработкой ошибок."""
        if self.debug:
            self.logger.info(f"Начало выравнивания. Размеры: эталон {reference.shape}, скан {scan.shape}")

        # Извлечение контуров
        ref_centroids, ref_areas = self._get_contours(reference, "Эталон")
        scan_centroids, scan_areas = self._get_contours(scan, "Скан")

        best = {
            'matrix': None,
            'orientation': None,
            'inliers': -1,
            'matches': 0,
            'error': float('inf')
        }

        # Перебор ориентаций
        for orientation in self._orientations:
            if self.debug:
                self.logger.info(f"Проверка ориентации: {orientation}")

            try:
                # Применение ориентации
                scan_transformed = self._apply_orientation(
                    scan_centroids, orientation, scan.shape
                )

                # Матчинг точек
                matches = self._match_points(ref_centroids, scan_transformed)
                if len(matches) < 3:
                    if self.debug:
                        self.logger.warning(f"Недостаточно соответствий: {len(matches)}")
                    continue

                # Оценка преобразования
                M_affine, inliers = self._estimate_transform(
                    scan_transformed[matches[:, 1]],
                    ref_centroids[matches[:, 0]]
                )

                if M_affine is None or inliers < 3:
                    if self.debug:
                        self.logger.warning("RANSAC не нашел решение")
                    continue

                # Вычисление ошибки
                aligned_points = cv2.transform(
                    scan_transformed[matches[:, 1]].reshape(-1, 1, 2),
                    M_affine
                ).reshape(-1, 2)
                error = np.mean(np.linalg.norm(
                    aligned_points - ref_centroids[matches[:, 0]], axis=1
                ))

                # Комбинирование преобразований
                if orientation != '0°':
                    # Создаем полную матрицу преобразования
                    M_orient = np.eye(3)
                    if orientation == 'flip_h':
                        M_orient[0, 0] = -1
                        M_orient[0, 2] = scan.shape[1]
                    elif orientation == 'flip_v':
                        M_orient[1, 1] = -1
                        M_orient[1, 2] = scan.shape[0]
                    else:
                        angle = float(orientation[:-1])
                        if orientation == '90°':
                            angle = -90
                        elif orientation == '270°':
                            angle = 90
                        M_orient[:2] = cv2.getRotationMatrix2D(
                            (scan.shape[1]/2, scan.shape[0]/2), angle, 1.0
                        )
                    
                    M_affine_ext = np.vstack([M_affine, [0, 0, 1]])
                    M_total = M_affine_ext @ M_orient
                    M_total = M_total[:2, :]
                else:
                    M_total = M_affine

                if self.debug:
                    self.logger.info(
                        f"Ориентация {orientation}: инлайнеров={inliers}, "
                        f"ошибка={error:.2f} px, совпадений={len(matches)}"
                    )

                # Обновление лучшего результата
                if inliers > best['inliers'] or (
                    inliers == best['inliers'] and error < best['error']
                ):
                    best.update(
                        matrix=M_total,
                        orientation=orientation,
                        inliers=inliers,
                        matches=len(matches),
                        error=error
                    )

            except Exception as e:
                if self.debug:
                    self.logger.error(f"Ошибка при обработке {orientation}: {str(e)}")
                continue

        # Проверка результата
        if best['matrix'] is None:
            if self.debug:
                self.logger.error("Выравнивание не удалось")
            raise ValueError("Не удалось найти подходящее преобразование")

        # Применение преобразования
        result_img = cv2.warpAffine(
            scan, best['matrix'], (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_NEAREST
        )

        # Вычисление корреляции
        correlation = cv2.matchTemplate(
            reference.astype(np.float32),
            result_img.astype(np.float32),
            cv2.TM_CCOEFF_NORMED
        )[0][0]

        if self.debug:
            self.logger.info(
                f"Лучший результат: ориентация={best['orientation']}, "
                f"инлайнеров={best['inliers']}, корреляция={correlation:.3f}"
            )

        return {
            'result_image': result_img,
            'metrics': {
                'inliers': best['inliers'],
                'matches': best['matches'],
                'error': best['error'],
                'correlation': correlation,
                'orientation': best['orientation']
            },
            'debug_info': {
                'matrix': best['matrix'].tolist(),
                'ref_contours': len(ref_centroids),
                'scan_contours': len(scan_centroids)
            }
        }