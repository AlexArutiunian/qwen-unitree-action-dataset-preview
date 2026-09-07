# Отчет: обзорные кадры из rosbag `.db3`

Дата: 2026-05-14

## Итог

Локальная папка с кадрами:

```text
/home/alpc/humanoid/nanoVLM/rgb_bag_keyframes_preview
```

Сделано:

```text
51 rosbag/run с RGB-топиком
429 отдельных кадров frame_*.jpg
51 contact_sheet.jpg
45M общий размер папки
```

Для каждой записи взято до 10 RGB-кадров, равномерно по всей записи: начало, середина, конец. Если в записи меньше 10 кадров, сохранены все доступные кадры.

## Как смотреть

В каждой папке run есть:

```text
contact_sheet.jpg   # быстрый обзор всех выбранных кадров этой записи
frame_00...jpg      # отдельные кадры
index.csv           # индексы кадров внутри rosbag
```

Общий индекс:

```text
rgb_bag_keyframes_preview/summary.csv
rgb_bag_keyframes_preview/index.csv
```

## Источник

Кадры извлекались с сервера:

```text
ssh -p 33322 arutiunyan_ag@93.175.18.15
```

Из всех `.db3` в:

```text
/home/arutiunyan_ag/shared_data
```

Использованный RGB-топик:

```text
/camera/camera/color/image_raw
sensor_msgs/msg/Image
```

Фильтр топиков:

- тип `sensor_msgs/msg/Image`;
- имя содержит `color` или `rgb`;
- имя не содержит `depth`.

## Результат экстракции

```text
runs 51
images 429
errors 0
```

Короткие mini-db3 из `scans_bfk/frame_previews/mini_db3` содержали только по 2 RGB-кадра, поэтому там сохранено по 2, а не по 10.
