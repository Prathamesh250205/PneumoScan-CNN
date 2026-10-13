| Model | Seeds | Accuracy | F1 | AUC | Sensitivity | Specificity | NIH AUC | Lung attention, pediatric | Lung attention, NIH |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet18 | 3 | 88.2 ± 2.9% | 0.913 ± 0.019 | 0.954 ± 0.013 | 0.989 ± 0.005 | 0.704 ± 0.083 | 0.794 ± 0.023 | 46.8 ± 5.5% | 39.0 ± 5.3% |
| MobileNetV2 | 3 | 83.9 ± 2.6% | 0.886 ± 0.016 | 0.956 ± 0.025 | 0.996 ± 0.001 | 0.578 ± 0.072 | 0.822 ± 0.024 | 67.4 ± 3.7% | 58.3 ± 2.3% |
| EfficientNet-B0 | 3 | 84.3 ± 0.6% | 0.888 ± 0.004 | 0.960 ± 0.002 | 0.996 ± 0.003 | 0.588 ± 0.017 | 0.758 ± 0.012 | 46.9 ± 7.1% | 26.8 ± 15.0% |

Chance lung attention (lung share of image): pediatric 38.5%, NIH 33.4%. Mean ± SD across seeds; SD uses ddof=1.

MobileNetV2 − ResNet18 AUC, per seed: pediatric +0.017, -0.012, +0.001; NIH -0.017, +0.078, +0.026.
