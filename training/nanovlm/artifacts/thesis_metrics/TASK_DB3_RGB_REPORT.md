# Отчет: существующие RGB-файлы и RGB-кадры внутри `.db3`

Дата проверки: 2026-05-14 09:17 MSK

## Краткий итог

Существующие RGB/color файлы с сервера скопированы локально сюда:

```text
/home/alpc/humanoid/nanoVLM/extracted_rgb_existing
```

Сверка локальной папки:

```text
count=1564
bytes=252965849
MiB=241.247
```

RGB-кадры внутри всех найденных rosbag `.db3`:

```text
db_files=58
matched_topics=51
topic=/camera/camera/color/image_raw
type=sensor_msgs/msg/Image
messages=32693
bytes=50103294696
MiB=47782.225
GiB=46.662329
errors=0
```

Если арифметически сложить уже лежащие отдельными файлами RGB/color картинки и RGB-сообщения внутри `.db3`, получится:

```text
bytes=50356260545
MiB=48023.472
GiB=46.897922
```

Важно: это сумма без дедупликации. Часть `selected_rgb`/`topic_color.jpg`/`sent_color.jpg` может быть извлечена из тех же rosbag, поэтому для оценки "сколько места занимают RGB-данные в rosbag" главным числом является `46.662329 GiB`, а для "сколько уже есть отдельными файлами" — `241.247 MiB`.

## Что было скопировано

Копировались только уже существующие отдельные RGB/color файлы:

- все файлы внутри `*/selected_rgb/*`;
- `topic_color.jpg`;
- `sent_color.jpg`.

Команда:

```bash
mkdir -p extracted_rgb_existing
rsync -av --relative -e 'ssh -p 33322' --files-from=<(ssh -p 33322 arutiunyan_ag@93.175.18.15 'find shared_data -type f \( -ipath "*/selected_rgb/*" -o -iname "topic_color.jpg" -o -iname "sent_color.jpg" \) -print') arutiunyan_ag@93.175.18.15:/home/arutiunyan_ag/ extracted_rgb_existing/
```

После первой синхронизации не хватало двух файлов из-за прерванной/неполной передачи в одном подкаталоге. Проверка:

```bash
rsync -avcn --relative -e 'ssh -p 33322' --files-from=<(ssh -p 33322 arutiunyan_ag@93.175.18.15 'find shared_data -type f \( -ipath "*/selected_rgb/*" -o -iname "topic_color.jpg" -o -iname "sent_color.jpg" \) -print') arutiunyan_ag@93.175.18.15:/home/arutiunyan_ag/ extracted_rgb_existing/
```

Она показала недостающие файлы:

```text
shared_data/saved_logs_yolo_inside_button/20260505_233854_172193099/sent_color.jpg
shared_data/saved_logs_yolo_inside_button/20260505_233854_172193099/topic_color.jpg
```

Они были докачаны командой:

```bash
rsync -av --relative -e 'ssh -p 33322' --files-from=<(ssh -p 33322 arutiunyan_ag@93.175.18.15 'printf "%s\n" shared_data/saved_logs_yolo_inside_button/20260505_233854_172193099/sent_color.jpg shared_data/saved_logs_yolo_inside_button/20260505_233854_172193099/topic_color.jpg') arutiunyan_ag@93.175.18.15:/home/arutiunyan_ag/ extracted_rgb_existing/
```

Финальная локальная проверка:

```bash
find extracted_rgb_existing -type f -printf '%s\n' | awk '{c++; s+=$1} END{print "count=" c; print "bytes=" s; printf "MiB=%.3f\n", s/1024/1024}'
```

Результат:

```text
count=1564
bytes=252965849
MiB=241.247
```

## Как считались RGB-кадры внутри `.db3`

На сервере нет команды `sqlite3`, поэтому `.db3` читались через стандартный Python-модуль `sqlite3`.

Сначала проверено количество rosbag-баз:

```bash
ssh -p 33322 arutiunyan_ag@93.175.18.15 'find shared_data -name "*.db3" -print | wc -l'
```

Результат:

```text
58
```

Фильтр RGB-топиков:

- тип сообщения: `sensor_msgs/msg/Image` или `sensor_msgs/msg/CompressedImage`;
- имя топика содержит `color` или `rgb`;
- имя топика не содержит `depth`.

Итоговый расчет:

```bash
ssh -p 33322 arutiunyan_ag@93.175.18.15 "python3 -u - <<'PY'
import os, sqlite3, time
rows=[]
errors=[]
dbs=[]
for root, dirs, files in os.walk('shared_data'):
    for fn in files:
        if fn.endswith('.db3'):
            dbs.append(os.path.join(root, fn))
dbs.sort()
print(f'db_files {len(dbs)}', flush=True)
for idx, db in enumerate(dbs, 1):
    t0=time.time()
    try:
        con=sqlite3.connect(db)
        cur=con.cursor()
        topics=[]
        for tid,name,typ in cur.execute('select id,name,type from topics'):
            lname=name.lower()
            if typ in ('sensor_msgs/msg/Image','sensor_msgs/msg/CompressedImage') and ('color' in lname or 'rgb' in lname) and 'depth' not in lname:
                topics.append((tid,name,typ))
        if topics:
            ids=','.join('?' for _ in topics)
            q=f'select topic_id, count(*), coalesce(sum(length(data)),0) from messages where topic_id in ({ids}) group by topic_id'
            stats={tid:(cnt,b) for tid,cnt,b in cur.execute(q, [tid for tid,_,_ in topics])}
            for tid,name,typ in topics:
                cnt,b=stats.get(tid,(0,0))
                rows.append((db,name,typ,cnt,b))
                print(f'ROW\t{db}\t{name}\t{typ}\t{cnt}\t{b}\t{b/1024/1024:.3f}', flush=True)
        print(f'PROGRESS\t{idx}/{len(dbs)}\tmatched={len(topics)}\tseconds={time.time()-t0:.2f}\t{db}', flush=True)
        con.close()
    except Exception as e:
        errors.append((db,str(e)))
        print(f'ERROR\t{db}\t{e}', flush=True)
print('SUMMARY')
print('db_files', len(dbs))
print('matched_topics', len(rows))
print('total_messages', sum(r[3] for r in rows))
print('total_bytes', sum(r[4] for r in rows))
print('total_mib', f'{sum(r[4] for r in rows)/1024/1024:.3f}')
print('total_gib', f'{sum(r[4] for r in rows)/1024/1024/1024:.6f}')
print('errors', len(errors))
print('BY_TOPIC')
by={}
for db,name,typ,cnt,b in rows:
    key=(name,typ)
    by[key]=[by.get(key,[0,0])[0]+cnt, by.get(key,[0,0])[1]+b]
for (name,typ),(cnt,b) in sorted(by.items()):
    print(f'{name}\t{typ}\t{cnt}\t{b}\t{b/1024/1024:.3f}')
PY"
```

Итоговый результат:

```text
SUMMARY
db_files 58
matched_topics 51
total_messages 32693
total_bytes 50103294696
total_mib 47782.225
total_gib 46.662329
errors 0
BY_TOPIC
/camera/camera/color/image_raw    sensor_msgs/msg/Image    32693    50103294696    47782.225
```

## Разбивка RGB из `.db3` по крупным группам

```text
15_640_480:       2,812 frames,  2.414 GiB
15_hd:              489 frames,  1.259 GiB
cv_module:        4,484 frames,  3.849 GiB
doors21:          3,743 frames,  3.213 GiB
elevator21:       7,891 frames,  6.773 GiB
red_triangle:     1,020 frames,  0.875 GiB
scans_bfk:       10,126 frames, 25.774 GiB
root rosbag:      2,128 frames,  2.505 GiB
```

Проверка суммы по группам совпадает с общим SQL-итогом: `32,693` кадра и `46.662329 GiB`.

## Вывод

По всем найденным `.db3` в `shared_data` RGB-кадры занимают `50,103,294,696 bytes`, то есть `47,782.225 MiB` или `46.662329 GiB`.

Отдельно уже существующие RGB/color картинки скопированы в `extracted_rgb_existing` и занимают `252,965,849 bytes`, то есть `241.247 MiB`.
