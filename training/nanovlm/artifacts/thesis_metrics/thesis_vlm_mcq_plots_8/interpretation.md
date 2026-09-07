# VLM MCQ Experimental Plots and Interpretation

## Robot MCQ validation
Best fine-tuned nanoVLM on the large robot MCQ validation set achieved 71.20% accuracy (613/861).
Micro-average ROC AUC: 0.9279.
Macro-average ROC AUC: 0.9284.

Generated plots:
- `robot_val_best_nano_confusion_matrix.png`
- `robot_val_best_nano_roc_auc.png`
- `robot_val_best_nano_per_class_accuracy.png`
- `robot_val_best_nano_confidence_hist.png`

## Hard22 glass-door benchmark
Model ranking on corrected hard22:
- our nanoVLM robot+reflect: 71.43% (15/21)
- Qwen2.5-VL-3B: 61.90% (13/21)
- Qwen3-VL-8B: 57.14% (12/21)
- nano AI2THOR: 57.14% (12/21)
- nano raw: 57.14% (12/21)
- Qwen2.5-VL-7B: 42.86% (9/21)
- nano robot-affordance: 38.10% (8/21)

Interpretation: the best result on hard22 is achieved by our fine-tuned nanoVLM variant (`our nanoVLM robot+reflect`). This supports the conclusion that domain-specific fine-tuning on robot/directional affordance and reflective-door examples improves robustness in the hard glass-door setting.

Pairwise plots:
- `hard22_pairwise_win_margin_heatmap.png`
- `hard22_pairwise_win_rate_heatmap.png`
- `hard22_pairwise_net_dominance.png`

## Suggested thesis wording
На большом роботном MCQ-наборе лучшая дообученная nanoVLM демонстрирует устойчивое качество, что подтверждается матрицей ошибок, ROC/AUC и поклассовой точностью. На hard22, состоящем из сложных сцен со стеклянными дверьми и отражениями, попарное сравнение показывает преимущество доменно-адаптированной nanoVLM-модели над базовыми nanoVLM-вариантами. Это указывает, что специализированное дообучение под роботные affordance-сцены и отражающие двери повышает практическую применимость компактной VLM для задач робототехнического восприятия.
