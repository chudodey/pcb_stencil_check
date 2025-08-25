"""
Модуль для автоматического сравнения Gerber-файлов с отсканированными изображениями трафаретов.
Основной рабочий процесс включает обработку Gerber, сканирование и выравнивание изображений.
"""

import sys
import argparse
import time
from pathlib import Path

# Сторонние библиотеки
import cv2
import numpy as np

# Локальные модули
from modules import (
    ConfigManager, SessionLogger, UserInterface,
    FileHandler, GerberProcessor, GerberRasterizer, ImageProcessor, AlignmentEngine
)
from modules.data_models import OrderData

# Добавляем путь к модулям в PYTHONPATH
sys.path.append(str(Path(__file__).parent))


def initialize_application() -> tuple:
    """
    Инициализация приложения (ШАГ 0 и ШАГ 1).

    Returns:
        tuple: (config, logger, ui) или None при ошибке
    """
    # ШАГ 0: Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description='Анализатор трафаретов')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Режим отладки')
    parser.add_argument('--order', '-o', type=str,
                        help='Предустановленный номер заказа')
    args = parser.parse_args()

    # Создание конфигурации
    app_config = ConfigManager(config_file='config.ini',
                               debug_mode=args.debug,
                               order_number=args.order)

    # Создание интерфейса пользователя
    app_ui = UserInterface(app_config)
    app_ui.show_header()

    # ШАГ 1.1: Проверка окружения Python
    env_issues = app_config.check_python_environment()
    if not app_ui.show_environment_check(env_issues):
        if env_issues and any("Отсутствуют модули" in issue for issue in env_issues):
            app_ui.show_message(
                f"\nВыполните: {app_config.install_command}", "info")
        return None, None, None

    # ШАГ 1.2: Проверка и создание директорий
    dir_status = app_config.check_directories()
    missing_dirs = [name for name, exists in dir_status.items() if not exists]

    if missing_dirs:
        missing_paths = [str(getattr(app_config, dir_name))
                         for dir_name in missing_dirs]
        if app_ui.ask_create_directories(missing_paths):
            created_dirs = app_config.create_missing_directories()
            if created_dirs:
                app_ui.show_success("Созданы директории:")
                for dir_path in created_dirs:
                    app_ui.show_message(f"  • {dir_path}", "info")
            else:
                app_ui.show_error("Не удалось создать директории")
                return None, None, None
        else:
            app_ui.show_error("Обновите config.ini и перезапустите программу")
            return None, None, None

    # Создание логгера
    app_logger = SessionLogger(app_config)
    app_logger.start_session()

    return app_config, app_logger, app_ui

# Функция обратного вызова для анимации


def update_animation(timeout: float, elapsed: float):
    ui.show_scan_waiting_animation(timeout, elapsed)


def main_app_flow(ui: UserInterface, config: ConfigManager, logger: SessionLogger):
    """Основной цикл работы приложения."""

    # ШАГ 2: Получение ФИО оператора
    operator_name = ui.get_operator_name()
    logger.set_operator(operator_name)
    ui.show_success(f"Оператор: {operator_name}")

    current_order = None

    while True:
        # --- ШАГ 3: Запрос номера заказа ---
        if current_order is None:
            order_number_input = ui.get_order_number()
            if not order_number_input:
                if ui.confirm_exit():
                    break
                else:
                    continue

            # --- ШАГ 4: Поиск Gerber-файла ---
            gerber_files = FileHandler.find_gerber_files(
                config.gerber_folder,
                order_number_input,
                config.multiple_files_rule
            )

            ui.show_gerber_search_result(gerber_files, order_number_input)
            if not gerber_files:
                # Если использовался preset, сбрасываем его и возвращаемся к запросу
                if config.preset_order_number:
                    config.preset_order_number = None
                continue

            current_order = OrderData(
                order_number=order_number_input,
                gerber_path=gerber_files[0]
            )

            logger.start_order_processing(
                current_order.order_number,
                current_order.gerber_path.name
            )

        try:
            # --- ШАГ 5: Обработка Gerber-файла ---
            ui.show_message("\n1. Обработка Gerber файла...", "process")
            gerber_content = current_order.gerber_path.read_text(
                encoding='utf-8', errors='ignore')

            # Инициализируем процессор без аргументов
            gerber_processor = GerberProcessor()

            start_time = time.time()
            gerber_result = gerber_processor.parse(gerber_content)
            gerber_result['metrics']['processing_time'] = time.time() - \
                start_time

            if gerber_result['metrics']['contour_count'] == 0:
                ui.show_warning("Gerber-файл не содержит контуров")
                current_order.min_contour_area_mm2 = 0.1  # 0.1 мм²
            else:
                current_order.min_contour_area_mm2 = gerber_result['metrics']['min_contour_area']

            current_order.gerber_metrics = gerber_result['metrics']
            current_order.gerber_contours = gerber_result['contours']
            current_order.gerber_bounds_mm = gerber_result['bounds_mm']

            ui.show_parsing_stats(gerber_result['metrics'])
            logger.log_parsing_results(
                current_order.order_number,
                gerber_result['metrics']
            )

        except (ValueError, IOError, OSError) as e:
            action = ui.ask_parsing_failed_action(str(e))
            if action == 1:  # Показать детали ошибки
                ui.show_error(f"Детали ошибки: {str(e)}")
                continue
            elif action == 2:  # Повторить парсинг
                continue
            elif action == 3:  # Новый заказ
                current_order = None
                continue
            elif action == 4:  # Выйти
                break

        # --- ШАГ 6: Ожидание скана ---
        ui.show_message("\n2. Сканирование трафарета...", "process")
        ui.show_scanning_instructions()

        scan_attempt = 0
        scan_path = None
        scan_dpi = None

        while scan_path is None:
            scan_attempt += 1

            try:
                # Ожидаем новый файл через FileHandler
                scan_path = FileHandler.wait_for_new_file(
                    scan_folder=config.scan_folder,
                    supported_formats=config.supported_image_formats,
                    check_interval=config.file_check_interval,
                    timeout=config.scan_wait_timeout,
                    callback=update_animation
                )

                # Если таймаут - показываем меню опций
                if scan_path is None:
                    last_existing_file = FileHandler.get_last_existing_file(
                        config.scan_folder, config.supported_image_formats)

                    action = ui.ask_scan_timeout_action(last_existing_file)

                    if action == 1:  # Продолжить ожидание
                        continue
                    elif action == 2:  # Использовать существующий файл
                        if last_existing_file:
                            scan_path = last_existing_file
                            ui.show_message(f"Выбран файл: {scan_path.name}")
                        else:
                            ui.show_error("В папке нет файлов изображений")
                            continue
                    elif action == 3:  # Новый заказ
                        current_order = None
                        break
                    elif action == 4:  # Выйти
                        if ui.confirm_exit():
                            break
                        else:
                            continue

                # Валидация файла
                is_valid, msg = FileHandler.validate_image_file(
                    scan_path, config.max_scan_file_size)
                if not is_valid:
                    ui.show_error(f"Файл скана некорректен: {msg}")
                    scan_path = None
                    continue

                # Получение информации о скане
                scan_dpi = FileHandler.get_image_dpi(
                    scan_path, config.default_dpi, config.dpi_priority)
                size_pixels = FileHandler.get_image_size(scan_path)
                size_mm = FileHandler.pixels_to_mm(size_pixels, scan_dpi)
                file_size = scan_path.stat().st_size

                ui.show_scan_info(scan_path, scan_dpi, size_pixels,
                                  size_mm, file_size)

                # Сохраняем данные в current_order
                current_order.scan_path = scan_path
                current_order.scan_dpi = scan_dpi

                # Проверяем, что scan_dpi установлен
                if current_order.scan_dpi is None:
                    current_order.scan_dpi = config.default_dpi
                    ui.show_warning(
                        f"Используется DPI по умолчанию: {config.default_dpi}")

            except TimeoutError:
                # Обрабатывается в основном цикле
                continue
            except (ValueError, FileNotFoundError) as e:
                ui.show_error(f"Ошибка при обработке файла: {e}")
                action = ui.ask_scan_failed_action()
                if action == 1:  # Повторить сканирование
                    scan_path = None
                    continue
                elif action == 2:  # Новый заказ
                    current_order = None
                    break
                elif action == 3:  # Выйти
                    break

        # Если scan_path is None, значит выходим или переходим к новому заказу
        if scan_path is None:
            if current_order is None:
                continue  # Возврат к шагу 3
            else:
                break  # Выход

        # --- ШАГ 7: Предобработка и валидация ---
        try:
            ui.show_message(
                "\n3. Обработка и проверка изображения скана...", "process")

            # Расчет минимального контура из данных Gerber
            min_contour_pixels = FileHandler.mm2_to_pixels(
                current_order.min_contour_area_mm2 * config.min_contour_coefficient,
                current_order.scan_dpi
            )

            img_processor = ImageProcessor(
                dpi=scan_dpi,
                binary_threshold=200,
                crop_min_area=min_contour_pixels,  # ← Важно!
                crop_max_area_ratio=0.1
            )

            file_bytes = np.fromfile(scan_path, dtype=np.uint8)
            scan_image_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            scan_result = img_processor.process(scan_image_original)
            current_order.scan_image_processed = scan_result['result_image']

            # ВАЛИДАЦИЯ результатов предобработки
            validation = img_processor.validate_preprocessing(
                scan_result['result_image'],
                current_order.gerber_metrics
            )

            if not validation['contour_count_ok']:
                raise ValueError(
                    f"Слишком мало контуров: {validation['contour_count']}")

            # Проверка размеров относительно эталона

            gerber_metrics = current_order.gerber_metrics

            # Ожидаемый размер эталона с учетом отступа из конфига

            gerber_size_mm = (
                gerber_metrics['board_width_mm'] + 2 * config.gerber_margin_mm,
                gerber_metrics['board_height_mm'] + 2 * config.gerber_margin_mm
            )

            scan_size_mm = FileHandler.pixels_to_mm(
                scan_result['result_image'].shape[:2][::-1],
                current_order.scan_dpi
            )

            # Проверка допуска размеров
            size_ratio = max(scan_size_mm[0] / gerber_size_mm[0],
                             scan_size_mm[1] / gerber_size_mm[1])
            tolerance = 1 + config.size_tolerance_percent / 100

            if size_ratio > tolerance or size_ratio < 1/tolerance:
                # Вне зависимости от настройки, сначала выводим детализированное предупреждение
                ui.show_dimension_mismatch_warning(
                    gerber_size_mm=gerber_size_mm,
                    scan_size_mm=scan_size_mm,
                    tolerance_percent=config.size_tolerance_percent
                )

                # Теперь спрашиваем пользователя, если это указано в конфиге
                if config.partial_preprocessing_action == 'ask_user':
                    action = ui.ask_preprocessing_action()
                    if action == 1:  # Продолжить обработку
                        pass  # Просто продолжаем
                    elif action == 2:  # Повторить сканирование
                        continue  # Возвращаемся к началу цикла сканирования
                    elif action == 3:  # Новый заказ
                        current_order = None
                        break  # Выходим из внутреннего цикла, возвращаемся к запросу заказа
                    elif action == 4:  # Выйти
                        if ui.confirm_exit():
                            return  # Завершаем функцию
                        else:
                            continue

            ui.show_preprocessing_result(True, scan_result['metrics'])

        except Exception as e:
            # Полноценная обработка ошибок согласно ТЗ
            action = ui.ask_preprocessing_failed_action(str(e))
            if action == 1:  # Повторить сканирование
                continue
            elif action == 2:  # Новый заказ
                current_order = None
                continue
            elif action == 3:  # Выйти
                break

        # --- ШАГ 8: Генерация эталонного изображения ---
        try:
            ui.show_message(
                "\n4. Генерация эталонного изображения...", "process")

            # Пересоздаем Gerber изображение с правильным DPI скана
            rasterizer = GerberRasterizer(
                contours=current_order.gerber_contours,
                bounds=current_order.gerber_bounds_mm
            )

            # Вызываем рендеринг с параметрами
            gerber_image = rasterizer.render(
                dpi=current_order.scan_dpi,
                margin_mm=config.gerber_margin_mm  # <-- Используем значение из конфига
            )
            current_order.gerber_image = gerber_image

            # Показываем информацию о эталоне
            gerber_size_pixels = current_order.gerber_image.shape[:2]
            gerber_size_mm = FileHandler.pixels_to_mm(
                gerber_size_pixels, current_order.scan_dpi)
            ui.show_reference_generation(
                gerber_size_pixels, gerber_size_mm, current_order.scan_dpi)

        except Exception as e:
            ui.show_error(f"Ошибка генерации эталона: {e}")
            continue

        # --- ШАГ 9: Выравнивание, сравнение и сохранение ---
        try:
            ui.show_message("\n5. Выравнивание и сравнение...", "process")

            # Проверяем, что изображения загружены
            if current_order.gerber_image is None:
                raise ValueError("Gerber изображение не загружено")
            if current_order.scan_image_processed is None:
                raise ValueError(
                    "Обработанное сканированное изображение не загружено")

            # Вычисляем минимальный размер контура в пикселях на основе данных Gerber
            min_contour_area_mm2 = current_order.min_contour_area_mm2
            min_contour_coefficient = config.min_contour_coefficient

            # Преобразуем минимальную площадь из мм² в пиксели
            min_contour_pixels = FileHandler.mm2_to_pixels(
                min_contour_area_mm2 * min_contour_coefficient,
                current_order.scan_dpi
            )

            ui.show_debug(
                f"Минимальная площадь контура: {min_contour_area_mm2:.3f} мм²")
            ui.show_debug(
                f"Коэффициент: {min_contour_coefficient}")
            ui.show_debug(
                f"Минимальный контур в пикселях: {min_contour_pixels:.1f} px")

            aligner = AlignmentEngine(
                dpi=current_order.scan_dpi,
                debug=config.debug_mode,
                ransac_threshold=config.ransac_reprojection_threshold,
                max_iterations=config.max_iterations,
                confidence=config.confidence,
                refine_iterations=config.refine_iterations,
                min_contour_area=int(min_contour_pixels),
                consider_reflection=config.consider_reflection,
                rotation_angles=config.rotation_angles,
                reference_color=config.reference_color,
                scan_color=config.scan_color,
                intersection_color=config.intersection_color,
                info_font_size=config.info_font_size,
                info_text_color=config.info_text_color,
                info_background_color=config.info_background_color
            )

            alignment_result = aligner.align(
                current_order.gerber_image,
                current_order.scan_image_processed
            )
            best_alignment = alignment_result

            combined_image = aligner.create_combined_image(
                current_order.gerber_image,
                best_alignment['result_image'],
                order_number=current_order.order_number,
                operator_name=operator_name,
                correlation=best_alignment['metrics']['correlation']
            )

            # ШАГ 10: Вывод результатов
            ui.show_alignment_results(best_alignment['metrics'])
            ui.show_combined_image(combined_image)  # Новый метод для показа

            if best_alignment['metrics']['correlation'] < config.medium_correlation_threshold:
                action = ui.ask_alignment_failed_action(
                    best_alignment['metrics']['correlation'])
                if action == 1:  # Повторить сканирование
                    continue
                elif action == 2:  # Новый заказ
                    current_order = None
                    continue
                elif action == 3:  # Выйти
                    break
                continue

            # --- ШАГ 11: Сохранение результатов ---
            ui.show_message("\n6. Сохранение результатов...", "process")
            workspace = config.create_order_workspace(
                current_order.order_number)

            logger.log_alignment_result(
                current_order.order_number,
                current_order.scan_path,
                best_alignment['metrics'],
                workspace
            )

            saved_files = []

            # Сохранение промежуточных изображений (если включено)
            if config.save_intermediate_images:
                # Сохраняем Gerber изображение
                gerber_img_path = config.get_filename(
                    config.gerber_image_filename,
                    current_order.order_number,
                    workspace
                )
                cv2.imwrite(str(gerber_img_path), current_order.gerber_image)
                saved_files.append(gerber_img_path)

                # Сохраняем оригинальный скан
                original_scan_path = config.get_filename(
                    config.original_scan_filename,
                    current_order.order_number,
                    workspace
                )
                cv2.imwrite(str(original_scan_path), scan_image_original)
                saved_files.append(original_scan_path)

                # Сохраняем обработанный скан
                processed_scan_path = config.get_filename(
                    config.processed_scan_filename,
                    current_order.order_number,
                    workspace
                )
                cv2.imwrite(str(processed_scan_path),
                            current_order.scan_image_processed)
                saved_files.append(processed_scan_path)

            # Сохранение финального изображения
            if config.save_final_image and best_alignment.get('result_image') is not None:
                final_img_path = config.get_filename(
                    config.comparison_result_filename,
                    current_order.order_number,
                    workspace
                )

                cv2.imwrite(str(final_img_path), combined_image)
                saved_files.append(final_img_path)

            ui.show_files_saved(saved_files)

        except (ValueError, RuntimeError, IOError) as e:
            ui.show_error(f"Критическая ошибка на этапе выравнивания: {e}")
            logger.log_error(current_order.order_number, str(e))

            action = ui.ask_alignment_failed_action(0.0)
            if action == 1:  # Повторить сканирование
                continue
            elif action == 2:  # Новый заказ
                current_order = None
                continue
            elif action == 3:  # Выйти
                break

        # --- ШАГ 12: Меню действий ---
        choice = ui.show_main_menu()
        if choice == 1:  # Новый заказ
            current_order = None
            continue
        if choice == 2:  # Выход
            if ui.confirm_exit():
                break

    logger.end_session()
    ui.show_message("Работа программы завершена.", "info")


if __name__ == "__main__":
    try:
        config, logger, ui = initialize_application()
        if config and logger and ui:
            main_app_flow(ui, config, logger)
        else:
            print("Инициализация приложения не удалась. Программа завершена.")
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем.")
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        if 'ui' in locals():
            ui.show_error(f"Необработанное исключение: {e}")
