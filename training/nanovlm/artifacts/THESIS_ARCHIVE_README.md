# Thesis Metrics Archive

Дата сборки: 2026-05-19.

Назначение архива: компактная выжимка важных файлов по метрикам диплома из трех рабочих частей проекта:

- `deploy`: скрипты benchmark/evaluation, отчеты API/LoRA-сравнения, физические метрики робота, итоговый DOCX/MD документ.
- `lora-robot`: конфиги обучения, eval-скрипты, README, LoRA training/eval артефакты без весов моделей.
- `nanoVLM`: итоговые VLM/MCQ отчеты, графики, confusion matrix, pairwise win matrices, CSV с метриками.

Главный документ:

- `deploy/thesis_metrics_doc/robot_metrics_detailed_report.docx`
- `deploy/thesis_metrics_doc/robot_metrics_detailed_report.md`

Сознательно исключено:

- `.env` и ключи API;
- виртуальные окружения `.venv*`;
- веса моделей, `.safetensors`, большие tokenizer/model файлы;
- сырые большие датасеты изображений;
- `__pycache__` и временные файлы;
- raw request dumps из API-прогонов.

Дополнительно включено:

- `extra/dataset_mapping/100_statis_downloads_correct_mapping.csv` — правильная таблица соответствий JSON и русских описаний.
- `extra/prompt_engineering/g1_upper_body_joint_direction.*` — FK/URDF cheat sheet для prompt/RAG.
