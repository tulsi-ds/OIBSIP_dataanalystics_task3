# OIBSIP_dataanalystics_task3: Wine Quality Prediction

## 📌 Objective
Build and compare multiple machine learning classification models to predict wine quality (`good` vs. `bad`) based on physicochemical properties, identifying the strongest chemical predictors of high-quality wine.

## 🛠️ Tools & Technologies Used
* **Programming Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebook

## 🔍 Steps Performed
1. **Data Loading & Cleaning:**
   * Loaded the Wine Quality dataset (`1,143 rows x 13 columns`), verified zero missing values, removed duplicate entries, and dropped the non-predictive `Id` identifier column.
2. **Exploratory Data Analysis (EDA) & Target Engineering:**
   * Visualized feature distributions, acidity scatter plots, alcohol vs. quality bar charts, and a correlation heatmap.
   * Engineered binary (`good` for quality ≥ 6, `bad` for quality < 6; 621 good vs. 522 bad) and 3-class (`low`, `medium`, `high`) target labels from the raw quality scores.
3. **Data Splitting & Feature Scaling:**
   * Split the dataset into **80% training** and **20% testing (229 samples)** sets using stratified sampling (`random_state=42`).
   * Applied `StandardScaler` to normalize features for gradient- and distance-based classifiers.
4. **Model Training & Comparison:**
   * Trained and evaluated three classification algorithms with balanced class weights:
     * **Random Forest Classifier** (`n_estimators=200`)
     * **Support Vector Classifier (SVC)**
     * **Stochastic Gradient Descent (SGD) Classifier**
5. **Evaluation & Feature Importance:**
   * Evaluated all models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix heatmaps.
   * Extracted Random Forest feature importances to rank key chemical drivers.

## 💡 Outcome in Brief
* **Model Comparison Summary:**

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | **81.22%** | **0.8045** | **0.8629** | **0.8327** |
| **SVC** | 77.29% | 0.8000 | 0.7742 | 0.7869 |
| **SGD Classifier** | 75.55% | 0.7742 | 0.7742 | 0.7742 |

* **Key Findings:**
  * **Best Model:** **Random Forest Classifier** achieved the highest overall performance with **81.22% accuracy** and an **F1 score of 0.8327**.
  * **Top Predictors:** `alcohol` and `sulphates` emerged as the strongest positive predictors of wine quality, followed by `volatile acidity` and `total sulfur dioxide`.
