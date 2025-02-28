# Model Training Report

## Overview
This report presents the training and evaluation results of the deep learning model. The model was trained for 20 epochs, and its performance was measured on training, validation, and test datasets.

## Training Details
- **Device Used**: CPU
- **Number of Epochs**: 20

## Training and Validation Results
| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|-------|------------|----------------|----------|--------------|
| 1  | 1.8501  | 36.53% | 1.7433  | 41.04% |
| 2  | 1.5530  | 43.27% | 1.5026  | 44.70% |
| 3  | 1.4351  | 45.43% | 1.3650  | 46.77% |
| 4  | 1.4159  | 46.58% | 1.2934  | 47.74% |
| 5  | 1.2981  | 48.60% | 1.3152  | 48.96% |
| 6  | 1.2214  | 50.88% | 1.1595  | 53.90% |
| 7  | 1.0555  | 57.86% | 0.9455  | 62.20% |
| 8  | 0.9671  | 62.95% | 0.8574  | 65.79% |
| 9  | 0.8148  | 68.84% | 0.7620  | 70.55% |
| 10 | 0.6688  | 74.36% | 0.5778  | 80.06% |
| 11 | 0.5520  | 79.19% | 0.6352  | 77.87% |
| 12 | 0.5307  | 80.56% | 0.4607  | 82.62% |
| 13 | 0.3918  | 85.94% | 0.4010  | 85.98% |
| 14 | 0.3375  | 87.97% | 0.3089  | 89.02% |
| 15 | 0.2666  | 90.72% | 0.3472  | 86.46% |
| 16 | 0.2291  | 91.54% | 0.2772  | 90.49% |
| 17 | 0.2883  | 89.45% | 0.3355  | 87.80% |
| 18 | 0.2279  | 91.83% | 0.2633  | 90.98% |
| 19 | 0.1953  | 93.37% | 0.3171  | 88.48% |
| 20 | 0.1942  | 93.28% | 0.1878  | 92.68% |

## Test Performance
- **Test Loss**: 0.1804
- **Test Accuracy**: 93.07%

## Conclusion
The model demonstrated significant improvement over the training process, achieving a validation accuracy of **92.68%** and a test accuracy of **93.07%**. The loss consistently decreased, indicating effective learning.

### Next Steps
- Fine-tune hyperparameters to further improve performance.
- Try training on a **GPU** to accelerate learning.
- Experiment with **data augmentation** or **regularization** techniques for better generalization.

---
**Author**: Anand Dubey
