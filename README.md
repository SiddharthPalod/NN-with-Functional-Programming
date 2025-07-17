# Functional Neural Network for Pima Diabetes Prediction (OCaml)

## Overview
This project implements a functional neural network in OCaml to predict diabetes using the Pima Indians Diabetes dataset. I achieved 75% accuracy with a pure OCaml functional programming-based implementation.

## [Demo Link](https://youtu.be/cwHScudxyVs)

---

## Features & Approaches
- **Feature Selection:** Only the four most predictive features are used: Pregnancies, Glucose, BMI, and DiabetesPedigreeFunction.
- **Data Normalization:** All features are standardized (z-score normalization) before training.
- **Deep & Dense Architectures:** The neural network supports multiple hidden layers and dense connections. The best architecture found (via Optuna in Python) uses 3 hidden layers with 76, 72, and 68 units.
- **Dropout Regularization:** Dropout is applied to hidden layers to reduce overfitting.
- **Early Stopping:** Training halts if the test loss does not improve for a set number of epochs (patience).
- **Mini-batch Training:** Training uses mini-batch gradient descent for better convergence and generalization.
- **Data Augmentation:** Various strategies were tested:
  - Simple noise-based augmentation (adding random noise to features)
  - Oversampling and undersampling
  - Ultimately, the best results were achieved by using the Optuna-augmented CSV generated in Python, rather than in-code augmentation.
- **Hyperparameter Tuning:** Learning rate, dropout, batch size, and patience are all configurable.
- **Metrics & Reporting:**
  - Training and test metrics are saved to `metrics.csv`.
  - A classification report (precision, recall, F1-score, support) is saved to `classification_report.csv` after each run.

---

## Usage

### 1. **Dependencies:**
- OCaml (tested with 4.12+)
- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) 

### 2. **Prepare Data:**
- Place your training CSV (e.g., `diabetes.csv` or an Optuna-augmented CSV) in the project directory.
- By default, the code expects `diabetes.csv`. To use a different file, change the filename in `main.ml`:
  ```ocaml
  let diabetes_file = "diabetes.csv" in
  ```

### 3. **Build:**
```sh
make ./diabetes_nn
```

### 4. **Run:**
```sh
./diabetes_nn
```

### 5. **Outputs:**
- `metrics.csv`: Per-epoch training and test loss/accuracy.
- `classification_report.csv`: Precision, recall, F1-score, and support for each class on the test set.

---

## Example Output

### `metrics.csv`
```
epoch,train_loss,test_loss,train_acc,test_acc
1,0.521413,0.570221,0.746933,0.663793
2,0.496121,0.591093,0.725460,0.603448
3,0.522734,0.560455,0.731595,0.750000
...
```

### `classification_report.csv`
```
Class,Precision,Recall,F1-score,Support
1,0.6800,0.4250,0.5231,40
0,0.7473,0.8947,0.8144,76
```

---

## Results
- With the best configuration and original data, the model achieves up to **75% test accuracy**.
- Using large, Optuna-augmented data can lead to memory issues in OCaml; for best results, use moderate augmentation or the original dataset.
- The model outputs a detailed classification report for further analysis.

---

## Authors
- Siddharth Palod

---
