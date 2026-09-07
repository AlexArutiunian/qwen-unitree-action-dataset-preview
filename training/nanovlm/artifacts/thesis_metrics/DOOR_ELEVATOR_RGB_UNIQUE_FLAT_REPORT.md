# Отчет: чистый flat RGB датасет без повторов

Дата: 2026-05-14

## Папка

```text
/home/alpc/humanoid/nanoVLM/door_elevator_rgb_unique_flat
```

## Структура

```text
door_elevator_rgb_unique_flat/
  door/
  elevator/
  manifest.csv
  README.txt
```

В папках `door/` и `elevator/` лежат только RGB `.jpg` изображения. Вложенных run-папок нет.

## Количество

Исходно было:

```text
518 изображений
```

После удаления почти повторяющихся кадров:

```text
405 изображений всего
301 door
104 elevator
113 near-duplicates removed
0 exact duplicates removed
19M размер
```

## Как сохранено происхождение

Так как вложенные папки убраны, источник перенесен в имя файла:

```text
door__run_004__cv_module__my_door_opening__my_door_opening_0__idx_000158__sample_045__ts_1777128755112939069.jpg
```

В имени есть:

- класс: `door` / `elevator`;
- номер run;
- исходная папка/имя `.db3`;
- индекс сообщения внутри rosbag;
- sample id;
- timestamp.

Полная таблица соответствий лежит в:

```text
door_elevator_rgb_unique_flat/manifest.csv
```

## Как убирались повторы

Для каждого изображения считался perceptual hash:

```text
16x16 dHash
```

Кадр считался повтором, если внутри того же класса (`door` или `elevator`) уже был кадр с расстоянием:

```text
Hamming distance <= 10
```

Так убираются почти одинаковые соседние кадры, но остаются разные ракурсы, сцены и состояния для ручной разметки `open/closed/partially_open`.
