from pathlib import Path
import pandas as pd
import shutil
import csv

SRC = Path("robotpct_final")
OUT = Path("robot_mcq_10_all_A_manual")

items_ru = [
    [1, "Какой вариант лучше всего характеризует, что делает робот?", "Наклонился назад и смотрит на доску.", "Присел и коснулся руками пола.", "Стоит ровно перед стеной.", "Удивляется."],
    [2, "Что делает робот?", "Стоит ровно.", "Машет рукой.", "Готовится к нажатию на кнопку.", "Удивляется."],
    [3, "Что делает робот?", "Машет приветствие.", "Раскинул руки в стороны.", "Наклонился назад.", "Закрыл камеру рукой."],
    [4, "Что делает робот?", "Открывает дверь.", "Машет приветствие.", "Раскинул правую руку в сторону.", "Закрыл камеру рукой."],
    [5, "Открыта ли дверь для робота?", "Да — он может идти.", "Нет — она полностью закрыта и он не сможет пройти.", "Он уже прошел через дверь.", "Он только начинает ее открывать."],
    [6, "В какой позе робот?", "Он раскинул обе руки в стороны.", "Он машет левой рукой.", "Он поднял только левую руку вверх.", "Робот открывает дверь."],
    [7, "В какой позе робот?", "Развел руки горизонтально в стороны.", "Руки в стороны ниже горизонта.", "Машет левой рукой.", "Поднял руки над головой."],
    [8, "Что делает робот?", "Топает.", "Упал на человека.", "Ровно стоит.", "Машет руками."],
    [9, "Что делает робот?", "Машет правой рукой.", "Машет левой рукой.", "Стоит в Т-позе.", "Идет в сторону от наблюдателя."],
    [10, "Что лучше всего характеризует картинку?", "Два робота: один выключен, другой включен.", "Роботы ходят по комнате.", "Роботы висят на подвесе.", "Робот смотрит на другого робота."],
]

items_en = [
    [1, "Which option best describes what the robot is doing?", "Leaning backward and looking at the board.", "Squatting and touching the floor with its hands.", "Standing straight in front of the wall.", "Looking surprised."],
    [2, "What is the robot doing?", "Standing straight.", "Waving its hand.", "Getting ready to press a button.", "Looking surprised."],
    [3, "What is the robot doing?", "Waving hello.", "Spreading both arms out to the sides.", "Leaning backward.", "Covering the camera with its hand."],
    [4, "What is the robot doing?", "Opening the door.", "Waving hello.", "Stretching its right arm out to the side.", "Covering the camera with its hand."],
    [5, "Is the door open for the robot?", "Yes, it can go through.", "No, it is fully closed and the robot cannot pass.", "It has already gone through the door.", "It is only starting to open the door."],
    [6, "What pose is the robot in?", "It has spread both arms out to the sides.", "It is waving with its left hand.", "It has raised only its left hand upward.", "The robot is opening the door."],
    [7, "What pose is the robot in?", "It has spread its arms horizontally out to the sides.", "Its arms are out to the sides but below horizontal.", "It is waving with its left hand.", "It has raised its arms above its head."],
    [8, "What is the robot doing?", "Stomping.", "Falling onto a person.", "Standing straight.", "Waving its arms."],
    [9, "What is the robot doing?", "Waving with its right hand.", "Waving with its left hand.", "Standing in a T-pose.", "Walking away from the observer."],
    [10, "Which option best describes the image?", "Two robots are shown: one is powered off and the other is powered on.", "The robots are walking around the room.", "The robots are hanging from a suspension system.", "One robot is looking at the other robot."],
]

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["idx", "question", "A", "B", "C", "D"])
        w.writerows(rows)

write_csv(SRC / "q.csv", items_ru)
write_csv(SRC / "qe.csv", items_en)

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "images").mkdir(parents=True, exist_ok=True)

rows = []
for row in items_en:
    idx, question, A, B, C, D = row
    src = SRC / f"{idx}.png"
    if not src.exists():
        raise FileNotFoundError(src)

    dst_rel = f"images/robot_{idx:02d}.png"
    shutil.copy2(src, OUT / dst_rel)

    choices = [A, B, C, D]
    rows.append({
        "id": f"robot_{idx:02d}",
        "image": dst_rel,
        "question": question,
        "choices": str(choices),
        "gt_idx": 0,
        "answer": A,
        "task": "robot_mcq_10_all_A_manual",
        "source_path": str(src),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT / "metadata.csv", index=False)

print("DONE:", OUT)
print(df[["id", "image", "question", "answer", "gt_idx", "source_path"]].to_string(index=False, max_colwidth=220))
print("\ngt_idx counts:")
print(df["gt_idx"].value_counts().sort_index().to_string())
