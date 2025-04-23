# 🩺 Disease Prediction using SVM

This project aims to build a **machine learning model** to predict the likelihood of a disease in patients based on medical attributes using the **Support Vector Machine (SVM)** algorithm.

## 📂 Project Structure

```
disease-prediction-svm/
├── disease-prediction-svm.ipynb  # Main Jupyter Notebook
├── README.md                     # Project description and usage
└── requirements.txt              # (Optional) Dependencies list
```

## 🧠 Model Used

- **Support Vector Machine (SVM)**: A supervised learning model effective for classification tasks. Suitable for binary classification such as disease detection.

## 📊 Dataset

The model uses the **Pima Indians Dataset** which contains features such as:
- Number of Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Pedigree Function
- Age
- Outcome (0 = No Disease, 1 = Disease)

> Dataset is typically available from open sources like [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) or UCI Machine Learning Repository.

## 🚀 How to Run

1. Clone the repository or download the notebook.
2. Open the notebook in [Google Colab](https://colab.research.google.com/) or JupyterLab.
3. Run each cell sequentially to:
   - Load the dataset
   - Preprocess the data
   - Train the SVM model
   - Evaluate the model performance

## 📈 Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

## 📌 Requirements

You can install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

Typical packages include:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## 📷 Visualizations

The notebook includes various plots such as:
- Count plots of diseased vs. non-diseased cases
- Correlation heatmaps
- Confusion Matrix

## 📬 Contact

For suggestions or collaborations, feel free to connect!

