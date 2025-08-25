import cv2
import numpy as np
from typing import Dict, Tuple, List, Tuple
import math
import logging
from datetime import datetime
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
from PIL import Image, ImageDraw, ImageFont


class AlignmentEngine:
    """
    Комбинированный выравниватель изображений.
    Версия 5.0 - объединяет два подхода для максимальной надежности.
    """

    def __init__(self, dpi: int, debug: bool = False,
                 ransac_threshold: float = 3.0,
                 max_iterations: int = 2000,
                 confidence: float = 0.99,
                 refine_iterations: int = 10,
                 min_contour_area: int = 10,
                 consider_reflection: bool = False,
                 rotation_angles: List[int] = None,
                 reference_color: Tuple[int, int, int] = (255, 0, 0),  # RGB
                 scan_color: Tuple[int, int, int] = (0, 255, 255),     # RGB
                 intersection_color: Tuple[int, int, int] = (
                     255, 255, 255),  # RGB
                 info_font_size: float = 1.0,
                 info_text_color: Tuple[int, int, int] = (
                     255, 255, 255),  # RGB
                 # RGB
                 info_background_color: Tuple[int, int, int] = (0, 0, 0)):
        self.params = {
            'dpi': dpi,
            'debug': debug,
            'ransac_threshold': ransac_threshold,
            'max_iterations': max_iterations,
            'confidence': confidence,
            'refine_iterations': refine_iterations,
            'min_contour_area': min_contour_area,
            'consider_reflection': consider_reflection,
            'rotation_angles': rotation_angles
        }
        self.reference_color = reference_color[::-1]  # to BGR
        self.scan_color = scan_color[::-1]
        self.intersection_color = intersection_color[::-1]
        self.info_font_size = info_font_size
        self.info_text_color = info_text_color[::-1]
        self.info_background_color = info_background_color[::-1]

        # Настройка логирования
        self.debug = debug
        self.logger = logging.getLogger('CombinedAlignmentEngine')
        self.logger.setLevel(logging.DEBUG if debug else logging.WARNING)

        if debug:
            if not self.logger.handlers:
                ch = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

            self.logger.info(
                f"Инициализация движка с параметрами: {self.params}")

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
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                maxIters=self.params['max_iterations'],
                confidence=self.params['confidence'],
                refineIters=self.params['refine_iterations']
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
        ref_contours, ref_centroids = self._extract_contours_and_centroids(
            reference, "Эталон")

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
                scan_transformed = self._transform_image_simple(
                    scan, rotate=rotate, flip=flip)

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
                aligned = cv2.warpAffine(
                    scan_transformed, affine_matrix, (width, height))

                # Корреляция
                correlation = pearsonr(
                    reference.flatten(), aligned.flatten())[0]

                if self.debug:
                    self.logger.debug(
                        f"Подход 1, {orientation_name}: инлайнеров={inliers_count}, "
                        f"ошибка={mean_error:.2f}, корреляция={correlation:.4f}"
                    )

                # Обновление лучшего результата
                is_better = (
                    inliers_count > best_result['inliers'] or
                    (inliers_count == best_result['inliers'] and mean_error < best_result['error']) or
                    (inliers_count == best_result['inliers'] and mean_error ==
                     best_result['error'] and correlation > best_result['correlation'])
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
                    self.logger.error(
                        f"Подход 1, ошибка {orientation_name}: {str(e)}")
                continue

        return best_result if best_result['matrix'] is not None else None

    def _approach_2_transform_reference(self, reference: np.ndarray, scan: np.ndarray) -> Dict:
        """
        Подход 2: Трансформируем эталон (как в версии 4.0, но исправленный).
        """
        if self.debug:
            self.logger.info("Подход 2: Трансформация эталона")

        # Извлечение контуров скана (один раз)
        scan_contours, scan_centroids = self._extract_contours_and_centroids(
            scan, "Скан")

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
                scan_aligned = cv2.warpAffine(
                    scan, affine_matrix, (width, height))

                # Корреляция
                correlation = pearsonr(
                    ref_transformed.flatten(), scan_aligned.flatten())[0]

                if self.debug:
                    self.logger.debug(
                        f"Подход 2, {orientation_name}: инлайнеров={inliers_count}, "
                        f"ошибка={mean_error:.2f}, корреляция={correlation:.4f}"
                    )

                # Обновление лучшего результата
                is_better = (
                    inliers_count > best_result['inliers'] or
                    (inliers_count == best_result['inliers'] and mean_error < best_result['error']) or
                    (inliers_count == best_result['inliers'] and mean_error ==
                     best_result['error'] and correlation > best_result['correlation'])
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
                    self.logger.error(
                        f"Подход 2, ошибка {orientation_name}: {str(e)}")
                continue

        # Если найдено решение, вычисляем финальную матрицу
        if best_result['matrix'] is not None:
            try:
                # Исправленная композиция матриц
                affine_matrix = best_result['matrix']
                ref_transform_matrix = best_result['ref_transform_matrix']

                # Вычисляем обратную матрицу для трансформации эталона
                ref_inverse_matrix = cv2.invertAffineTransform(
                    ref_transform_matrix)

                # Комбинируем матрицы: сначала RANSAC, потом обратная трансформация эталона
                final_matrix = ref_inverse_matrix @ np.vstack(
                    [affine_matrix, [0, 0, 1]])
                final_matrix = final_matrix[:2, :]

                # Применяем к исходному скану
                height, width = reference.shape
                result_image = cv2.warpAffine(
                    scan, final_matrix, (width, height))

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
            self.logger.info(
                f"Начало комбинированного выравнивания. Размеры: эталон {reference.shape}, скан {scan.shape}")

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
                (candidate['inliers'] == best_result['inliers'] and candidate['error'] ==
                 best_result['error'] and candidate['correlation'] > best_result['correlation'])
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

        # Вычисляем дополнительные метрики из матрицы преобразования
        final_matrix = best_result.get(
            'final_matrix') or best_result.get('matrix')

        if final_matrix is not None:
            # Извлекаем угол поворота из матрицы
            rotation_rad = math.atan2(final_matrix[1, 0], final_matrix[0, 0])
            rotation_deg = math.degrees(rotation_rad)

            # Извлекаем смещение (в пикселях)
            shift_x_px = final_matrix[0, 2]
            shift_y_px = final_matrix[1, 2]

            # Конвертируем смещение в мм (зная DPI)
            mm_per_pixel = 25.4 / self.params['dpi']
            shift_x_mm = shift_x_px * mm_per_pixel
            shift_y_mm = shift_y_px * mm_per_pixel

            # Вычисляем процент совпадения
            result_image = best_result['result_image']
            match_mask = (reference > 0) & (result_image > 0)
            total_pixels = np.sum(reference > 0) + np.sum(result_image > 0)
            match_percentage = (2 * np.sum(match_mask) /
                                total_pixels * 100) if total_pixels > 0 else 0
        else:
            rotation_deg = 0
            shift_x_mm = 0
            shift_y_mm = 0
            match_percentage = 0

        return {
            'result_image': best_result['result_image'],
            'metrics': {
                'inliers': best_result['inliers'],
                'error': best_result['error'],
                'correlation': best_result['correlation'],
                'orientation': best_result['orientation'],
                'approach_used': best_result['approach'],
                'rotation': rotation_deg,
                'shift_x': shift_x_mm,
                'match_percentage': match_percentage
            },
            'debug_info': {
                'matrix': best_result['matrix'].tolist(),
                'approach_used': best_result['approach'],
                'candidates_tried': len(candidates),
                'rotation_deg': rotation_deg,       # ← Для отладки
                'shift_mm': [shift_x_mm, shift_y_mm]  # ← Для отладки
            }
        }

    def create_combined_image(self, gerber_img: np.ndarray, aligned_scan_img: np.ndarray,
                              order_number: str, operator_name: str, correlation: float) -> np.ndarray:
        if len(gerber_img.shape) == 3:
            gerber_img = cv2.cvtColor(gerber_img, cv2.COLOR_BGR2GRAY)
        if len(aligned_scan_img.shape) == 3:
            aligned_scan_img = cv2.cvtColor(
                aligned_scan_img, cv2.COLOR_BGR2GRAY)

        gerber_norm = cv2.normalize(gerber_img, None, 0, 255, cv2.NORM_MINMAX)
        scan_norm = cv2.normalize(
            aligned_scan_img, None, 0, 255, cv2.NORM_MINMAX)

        h, w = gerber_norm.shape
        # +50 px сверху для текста
        combined = np.zeros((h + 50, w, 3), dtype=np.uint8)

        # Наложение изображений в нижней части
        gerber_mask = gerber_norm > 0
        scan_mask = scan_norm > 0
        combined[50:, :, :][gerber_mask & ~scan_mask] = self.reference_color
        combined[50:, :, :][scan_mask & ~gerber_mask] = self.scan_color
        combined[50:, :, :][gerber_mask & scan_mask] = self.intersection_color

        # Фон для текста
        cv2.rectangle(combined, (0, 0), (w, 50),
                      self.info_background_color, -1)

        # Конвертируем в PIL Image для добавления текста
        pil_img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        print(f"Размер шрифта:  {self.info_font_size}")

        try:
            # Пробуем загрузить шрифт (укажите путь к своему шрифту)
            font = ImageFont.truetype("arial.ttf", self.info_font_size)
        except:
            # Если шрифт не найден, используем стандартный
            font = ImageFont.load_default()

        # Текст
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text = f"Заказ: {order_number} | Дата: {date} | Оператор: {operator_name} | Результат: {correlation:.3f}"

        # Добавляем текст
        draw.text((10, 15), text, font=font, fill=tuple(self.info_text_color))

        # Конвертируем обратно в OpenCV формат
        combined = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return combined
