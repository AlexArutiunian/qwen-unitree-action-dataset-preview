# Отчет по задаче: объем RGB-картинок в `shared_data`

Дата проверки: 2026-05-14 09:01 MSK

## Краткий итог

На сервере `arutiunyan_ag@93.175.18.15:33322` папка `shared_data` занимает:

```text
108G    shared_data
```

Объем RGB/color картинок, посчитанный как сумма размеров файлов изображений:

```text
252965849 bytes
241.247 MiB
0.235593 GiB
1564 файла
```

В этот итог включены:

```text
selected_rgb     files=204    bytes=119837748    MiB=114.286
sent_color.jpg   files=679    bytes=66482658     MiB=63.403
topic_color.jpg  files=681    bytes=66645443     MiB=63.558
```

Если считать только явно названные папки `selected_rgb`, то получается:

```text
119837748 bytes
114.286 MiB
0.111610 GiB
204 файла
```

## Что именно считалось

В `shared_data` лежат разные типы данных: ROS bag базы `.db3`, `metadata.yaml`, JSON, depth-файлы `.npy`, previews, маски, annotated-картинки и обычные изображения.

Чтобы ответить на вопрос "только rgb картинки", я не включал `.db3` ROS bag файлы, потому что это базы с данными внутри, а не отдельные картинки-файлы. Основной итог выше считает файловые RGB/color кадры:

- все изображения внутри директорий `selected_rgb`;
- файлы `topic_color.jpg`;
- файлы `sent_color.jpg`.

Маски `color_mask.png` и debug/annotated изображения `annotated_color_detector.jpg` в основной итог не включены.

## Команды и результаты

### 1. Подключение к серверу и поиск `shared_data`

Команда из задачи:

```bash
ssh -p 33322 arutiunyan_ag@93.175.18.15
```

Проверочная команда:

```bash
ssh -p 33322 arutiunyan_ag@93.175.18.15 'pwd; ls -lah; find . -maxdepth 3 -type d -iname "*shared*" -o -type d -name "shared_data"'
```

Ключевой результат:

```text
/home/arutiunyan_ag
./shared_data
```

### 2. Общий размер `shared_data`

Команда:

```bash
ssh -p 33322 arutiunyan_ag@93.175.18.15 'du -sh shared_data; du -h --max-depth=2 shared_data | sort -h | tail -60'
```

Ключевой результат:

```text
108G    shared_data
55G     shared_data/scans_bfk
17G     shared_data/elevator21
9.0G    shared_data/cv_module
7.8G    shared_data/doors21
5.5G    shared_data/15_640_480
3.3G    shared_data/15_hd
108G    shared_data
```

### 3. Просмотр структуры папок

Команда:

```bash
ssh -p 33322 arutiunyan_ag@93.175.18.15 'find shared_data -maxdepth 3 -type d | sort | sed -n "1,200p"'
```

Ключевой результат: найдены, среди прочего, явные RGB-директории:

```text
shared_data/door_selected/15_640_480/selected_rgb
shared_data/door_selected/15_hd/selected_rgb
shared_data/door_selected_test/15_640_480/selected_rgb
shared_data/door_selected_test/15_hd/selected_rgb
```

### 4. Проверка примеров файлов

Команда:

```bash
ssh -p 33322 arutiunyan_ag@93.175.18.15 'find shared_data -maxdepth 4 -type f | sed -n "1,160p"'
```

В результате видно, что в датасете есть `.db3`, `.yaml`, `.json`, `.npy`, `.jpg`, `.png`. В логах встречаются RGB/color файлы:

```text
sent_color.jpg
topic_color.jpg
```

### 5. Сравнение всех картинок и картинок с `rgb/color` в пути

Команда:

```bash
ssh -p 33322 arutiunyan_ag@93.175.18.15 'find shared_data -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -printf "%s\t%p\n" | awk "BEGIN{total=0;count=0} {total+=\$1; count++} END{printf \"all_images_count=%d\nall_images_bytes=%d\nall_images_gib=%.6f\n\", count,total,total/1024/1024/1024}"; find shared_data -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \( -ipath "*rgb*" -o -ipath "*color*" \) -printf "%s\t%p\n" | awk "BEGIN{total=0;count=0} {total+=\$1; count++} END{printf \"rgb_color_images_count=%d\nrgb_color_images_bytes=%d\nrgb_color_images_gib=%.6f\n\", count,total,total/1024/1024/1024}"'
```

Результат:

```text
all_images_count=2915
all_images_bytes=353387350
all_images_gib=0.329118
rgb_color_images_count=1582
rgb_color_images_bytes=253320165
rgb_color_images_gib=0.235923
```

Этот широкий поиск включает также `color_mask.png` и `annotated_color_detector.jpg`, поэтому для основного ответа ниже использован более точный фильтр.

### 6. Размер только `selected_rgb`

Команда:

```bash
ssh -p 33322 arutiunyan_ag@93.175.18.15 'for d in $(find shared_data -type d -iname "selected_rgb" | sort); do c=$(find "$d" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l); b=$(find "$d" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) -printf "%s\n" | awk "{s+=\$1} END{print s+0}"); mib=$(awk -v b="$b" "BEGIN{printf \"%.3f\", b/1024/1024}"); printf "%s\tfiles=%s\tbytes=%s\tMiB=%s\n" "$d" "$c" "$b" "$mib"; done'
```

Результат:

```text
shared_data/door_selected/15_640_480/selected_rgb        files=100    bytes=25705295    MiB=24.514
shared_data/door_selected/15_hd/selected_rgb             files=100    bytes=91502076    MiB=87.263
shared_data/door_selected_test/15_640_480/selected_rgb   files=2      bytes=536959      MiB=0.512
shared_data/door_selected_test/15_hd/selected_rgb        files=2      bytes=2093418     MiB=1.996
```

Итого по `selected_rgb`: `119837748 bytes`, `114.286 MiB`, `204 файла`.

### 7. Финальный подсчет RGB/color кадров без масок и annotated-файлов

Команда:

```bash
ssh -p 33322 arutiunyan_ag@93.175.18.15 'find shared_data -type f \( -ipath "*/selected_rgb/*" -o -iname "topic_color.jpg" -o -iname "sent_color.jpg" \) -printf "%s\t%p\n" | awk "BEGIN{total=0;count=0} {total+=\$1; count++} END{printf \"raw_rgb_color_count=%d\nraw_rgb_color_bytes=%d\nraw_rgb_color_mib=%.3f\nraw_rgb_color_gib=%.6f\n\", count,total,total/1024/1024,total/1024/1024/1024}"; printf "\n--- breakdown ---\n"; find shared_data -type f \( -ipath "*/selected_rgb/*" -o -iname "topic_color.jpg" -o -iname "sent_color.jpg" \) -printf "%s\t%p\n" | awk -F"\t" "{if (\$2 ~ /\/selected_rgb\//) key=\"selected_rgb\"; else if (\$2 ~ /\/topic_color.jpg$/) key=\"topic_color.jpg\"; else if (\$2 ~ /\/sent_color.jpg$/) key=\"sent_color.jpg\"; else key=\"other\"; bytes[key]+=\$1; count[key]++} END{for(k in bytes) printf \"%s\tfiles=%d\tbytes=%d\tMiB=%.3f\n\", k,count[k],bytes[k],bytes[k]/1024/1024}" | sort'
```

Результат:

```text
raw_rgb_color_count=1564
raw_rgb_color_bytes=252965849
raw_rgb_color_mib=241.247
raw_rgb_color_gib=0.235593

--- breakdown ---
selected_rgb      files=204    bytes=119837748    MiB=114.286
sent_color.jpg    files=679    bytes=66482658     MiB=63.403
topic_color.jpg   files=681    bytes=66645443     MiB=63.558
```

## Вывод

Основной ответ: RGB/color картинки в `shared_data` занимают примерно `241.247 MiB` (`0.235593 GiB`) при подсчете отдельных файлов `selected_rgb`, `topic_color.jpg` и `sent_color.jpg`.

Строгий вариант только по папкам `selected_rgb`: примерно `114.286 MiB` (`0.111610 GiB`).
