import cv2
import numpy as np
import sys
from pathlib import Path
from PIL import Image
from scipy import ndimage
from itertools import groupby

# Поддерживаемые расширения
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Параметры обрезки
CROP_THRESHOLD = 245
CROP_MIN_SIZE = 100
CROP_EXTRA = 5  # пикселей со всех сторон


def smart_crop_pil(img_pil: Image.Image, threshold: int = 245, min_object_size: int = 100, extra_crop: int = 5) -> Image.Image:
    """
    Обрезает изображение по содержимому, удаляя белые поля и мелкий шум,
    затем дополнительно обрезает extra_crop пикселей со всех сторон.
    """
    # Обработка прозрачности
    if img_pil.mode == 'RGBA':
        background = Image.new('RGB', img_pil.size, (255, 255, 255))
        background.paste(img_pil, mask=img_pil.split()[-1])
        img_pil = background
    elif img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')

    # Создаём маску нет-белых областей
    img_array = np.array(img_pil)
    avg_intensity = np.mean(img_array, axis=2)
    mask_bool = avg_intensity < threshold

    # Удаляем мелкий шум
    labeled_array, _ = ndimage.label(mask_bool)
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # игнорируем фон
    mask_cleaned = np.isin(labeled_array, np.where(component_sizes >= min_object_size)[0])

    # Обрезка по содержимому
    if np.any(mask_cleaned):
        coords = np.argwhere(mask_cleaned)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped = img_pil.crop((x_min, y_min, x_max + 1, y_max + 1))
    else:
        cropped = img_pil

    # Дополнительная симметричная подрезка (с защитой от перекрытия)
    w, h = cropped.size
    left = extra_crop
    top = extra_crop
    right = w - extra_crop
    bottom = h - extra_crop

    if left < right and top < bottom:
        final_crop = cropped.crop((left, top, right, bottom))
    else:
        final_crop = cropped  # недостаточно места — оставляем как есть

    return final_crop


def is_white_row(row: np.ndarray, threshold: int = 240, white_ratio: float = 0.95) -> bool:
    """
    Определяет, является ли строка пикселей «белой».
    """
    if row.ndim == 2 and row.shape[1] in (3, 4):  # (W, 3) или (W, 4)
        luminance = np.mean(row, axis=1)
    elif row.ndim == 1:  # grayscale
        luminance = row
    else:
        luminance = np.mean(row, axis=-1).flatten()

    white_pixels = np.sum(luminance >= threshold)
    return white_pixels >= white_ratio * luminance.size


def find_split_y(img: np.ndarray, min_gap_height: int = 5) -> int | None:
    """
    Ищет горизонтальную белую полосу между двумя фотографиями.
    Возвращает координату Y для разреза или None.
    """
    h = img.shape[0]
    white_flags = np.array([is_white_row(img[y]) for y in range(h)])

    groups = []
    for k, g in groupby(enumerate(white_flags), key=lambda x: x[1]):
        if k:  # белая полоса
            group = list(g)
            y_start = group[0][0]
            y_end = group[-1][0]
            if (y_end - y_start + 1) >= min_gap_height:
                groups.append((y_start, y_end))

    # Ищем полосу, не у краёв изображения
    for y_start, y_end in groups:
        if y_start > 20 and y_end < h - 20:
            return (y_start + y_end) // 2

    return None


def process_image(input_path: Path, output_dir: Path) -> bool:
    """
    Обрабатывает одно изображение: разделяет (если нужно) и обрезает.
    """
    # Загружаем через OpenCV для анализа структуры
    img_cv = cv2.imread(str(input_path))
    if img_cv is None:
        print(f"⚠️ Не удалось загрузить: {input_path}")
        return False

    split_y = find_split_y(img_cv)
    base_name = input_path.stem
    ext = input_path.suffix.lower()

    # Загружаем через PIL для качественной обрезки
    try:
        img_pil = Image.open(input_path)
    except Exception as e:
        print(f"⚠️ PIL не смог открыть {input_path}: {e}")
        return False

    if split_y is not None:
        # Разделяем на два изображения
        top_cv = img_cv[:split_y]
        bottom_cv = img_cv[split_y:]

        top_pil = Image.fromarray(cv2.cvtColor(top_cv, cv2.COLOR_BGR2RGB))
        bottom_pil = Image.fromarray(cv2.cvtColor(bottom_cv, cv2.COLOR_BGR2RGB))

        top_cropped = smart_crop_pil(top_pil, CROP_THRESHOLD, CROP_MIN_SIZE, CROP_EXTRA)
        bottom_cropped = smart_crop_pil(bottom_pil, CROP_THRESHOLD, CROP_MIN_SIZE, CROP_EXTRA)

        top_path = output_dir / f"{base_name}_top{ext}"
        bottom_path = output_dir / f"{base_name}_bottom{ext}"

        top_cropped.save(top_path, quality=95, optimize=True)
        bottom_cropped.save(bottom_path, quality=95, optimize=True)

        print(f"✅ Разделено и обрезано: {top_path.name}, {bottom_path.name}")

    else:
        # Одно фото — обрезаем оригинал
        cropped = smart_crop_pil(img_pil, CROP_THRESHOLD, CROP_MIN_SIZE, CROP_EXTRA)
        output_path = output_dir / f"{base_name}{ext}"
        cropped.save(output_path, quality=95, optimize=True)
        print(f"📎 Обрезано (одно фото): {output_path.name}")

    return True


def main():
    if len(sys.argv) != 3:
        print("Использование: python scan_processor.py <входная_папка> <выходная_папка>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.is_dir():
        print(f"Ошибка: входная папка не существует: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    ]

    if not image_files:
        print(f"Нет поддерживаемых изображений в: {input_dir}")
        return

    print(f"Найдено изображений: {len(image_files)}")
    processed = 0
    for img_path in sorted(image_files):
        try:
            if process_image(img_path, output_dir):
                processed += 1
        except Exception as e:
            print(f"❌ Ошибка при обработке {img_path}: {e}")

    print(f"\nГотово! Обработано: {processed} из {len(image_files)} файлов.")


if __name__ == "__main__":
    main()