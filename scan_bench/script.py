
import os

directory = '/home/rock/git/scan_bench/imgs'  # Замените на путь к вашей директории

# Получаем список файлов в директории и сортируем их по числовому значению
files = sorted(os.listdir(directory), key=lambda x: float(os.path.splitext(x)[0]))

# Переименовываем файлы
for i, filename in enumerate(files):
    old_path = os.path.join(directory, filename)
    new_filename = str(i+1) + ".jpg"  # Новое имя файла
    new_path = os.path.join(directory, new_filename)
    os.rename(old_path, new_path)