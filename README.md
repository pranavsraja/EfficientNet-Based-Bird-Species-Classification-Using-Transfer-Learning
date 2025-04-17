# EfficientNet-Based Bird Species Classification Using Transfer Learning

This repository contains a Deep Learning Project focused on **fine-grained image classification** using the **CUB-200-2011 dataset**. The task involves training and evaluating an **EfficientNet-B5** model to accurately classify 200 bird species.

---

## Project Overview
 
- **Notebook**: `DL_assignment_task_2.ipynb`  
- **Dataset**: [CUB-200-2011 (Caltech-UCSD Birds)](http://www.vision.caltech.edu/datasets/cub_200_2011/)

---

## Objective

- Build a deep learning pipeline for **fine-grained classification**.
- Train EfficientNet-B5 using **transfer learning** and **data augmentation**.
- Evaluate using accuracy, precision, recall, F1-score, and confusion matrix.
- Save the best model and create a test-time prediction pipeline.

---

## Key Features

### Data Preparation
- Split dataset into training, validation, and test sets
- Applied augmentation (crop, flip, rotation, jitter) using `torchvision.transforms`
- Loaded data using `ImageFolder` and `DataLoader`

### Model & Training
- Loaded pretrained **EfficientNet-B5** from `torchvision.models`
- Replaced final classifier to output 200 classes
- Used **AdamW optimizer**, **CrossEntropy loss**, **ReduceLROnPlateau scheduler**
- Early stopping mechanism included
- Best model saved as `efficientnet_birds_model.pth`

### Evaluation
- Accuracy, precision (macro), recall (macro), F1-score (macro)
- Confusion matrix for entire dataset and top 10 classes
- Per-class accuracy for top performing classes
- Classification report saved to CSV

### Prediction & Visualization
- Test image prediction with confidence score
- Sample prediction visualization with side-by-side input & predicted label
- Metrics visualized via bar plots and line graphs
- Full confusion matrix and prediction report exported as images

---

## Performance Summary

- **Test Accuracy**: ~76.35%  
- **Precision (Macro)**: 0.77  
- **Recall (Macro)**: 0.76  
- **F1 Score (Macro)**: 0.76  
- Detailed metrics available in `classification_report.csv`

---

## Tools and Libraries

- PyTorch  
- Torchvision  
- Scikit-learn  
- Matplotlib & Seaborn  
- PIL  
- Google Colab

---

## To Run Locally

```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

Make sure to organize the dataset in the following structure:
```
CUB_200_2011/
 ┣ train_subset/
 ┣ val_subset/
 ┣ test_images/
```

---

## Inference on Custom Image

To classify a new bird image:
- Load the saved model and class names
- Apply the same transforms
- Visualize the predicted label with confidence

---

## Author

- **Pranav Sunil Raja**  
- MSc Data Science & AI, Newcastle University  
- GitHub: [@pranavsraja](https://github.com/pranavsraja)

---

## License

This project is done for education and academic purposes. Dataset use must comply with the [CUB-200 license](http://www.vision.caltech.edu/datasets/cub_200_2011/).
