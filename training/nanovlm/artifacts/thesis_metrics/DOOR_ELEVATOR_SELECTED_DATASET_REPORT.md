# Отчет: выбранный door/elevator датасет

Дата: 2026-05-14

## Папка

```text
/home/alpc/humanoid/nanoVLM/door_elevator_selected_dataset
```

## Что внутри

Разложено только по верхним классам:

```text
door/
elevator/
```

Состояния `open`, `closed`, `partially_open` не размечались специально, их нужно выбрать руками.

## Количество

```text
door images:      402
elevator images:  116
total images:     518
contact sheets:     9
size:             28M
```

## Как семплировалось

Для каждого выбранного run взято:

- 50 равномерных кадров по всей записи;
- плюс старые 10 keyframe-индексов из `rgb_bag_keyframes_preview`;
- потом дубликаты индексов удалены.

Поэтому в run получилось 56 или 58 кадров.

## Выбранные runs

```text
door:
  004 cv_module/my_door_opening
  005 cv_module/my_door_opening10
  007 cv_module/my_door_opening12
  014 cv_module/my_door_opening8
  018 doors21/door_outside1
  019 doors21/door_outside2
  022 doors21/door_outside2

elevator:
  026 elevator21/buttons_entry_elevator3
  027 elevator21/buttons_outside_elevator1
```

## Файлы навигации

```text
door_elevator_selected_dataset/summary.csv
door_elevator_selected_dataset/index.csv
```

В каждой run-папке есть:

```text
contact_sheet.jpg
index.csv
*.jpg
```
