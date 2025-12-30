# Sonar Signal Classification Using Machine Learning

> A robust machine learning pipeline to classify underwater objects as either Rocks or Mines based on sonar frequency patterns.

## 📌 Overview
This project focuses on the critical task of underwater object detection. Using the Sonar Dataset, we develop and evaluate multiple machine learning models to distinguish between rocks and metal cylinders (mines) based on their sonar reflections. It covers the entire data science lifecycle: from data cleaning and statistical feature selection to deep learning implementation.


## ⚓ Problem Statement
**The Challenge:** Accurately detect whether an underwater object is a rock or a metal cylinder using sonar signal patterns.
* **Complexity:** The dataset consists of high-dimensional data (60 features) with a very small sample size (208 records), making models prone to overfitting.
* **Safety Critical:** The primary goal is to ensure high **Recall**, as missing a mine (False Negative) is much more costly than a false alarm.

## 💼 Business Problem
Manual underwater identification and mine-clearing operations are extremely dangerous, slow, and expensive for both defense and commercial maritime sectors. 
* **Operational Risk:** Undetected underwater mines pose a lethal threat to naval vessels and civilian shipping lanes.
* **Economic Impact:** Accidental encounters with mines lead to catastrophic equipment loss and environmental damage.
* **The Need:** There is a critical requirement for a reliable, automated system that can process sonar data in real-time to provide high-accuracy classification, allowing human operators to focus on neutralized threats from a safe distance.

## ⚓ Problem Statement
**The Challenge:** High-dimensional sonar data (60 frequency features) must be analyzed with limited samples (208 records) to ensure reliable classification.
* **Technical Hurdle:** The risk of overfitting is high due to the small sample-to-feature ratio.
* **Success Metric:** In this safety-critical context, **Recall** is the primary metric; missing a single mine (False Negative) has significantly higher consequences than a false alarm.


## 📂 Project Structure

sonar-signal-classification/
├── data/
│   └── sonar1.csv              # Raw Sonar Dataset
├── notebooks/
│   └── Capstone_P2.ipynb       # Main analysis and modeling notebook
├── reports/
│   └── Presentation_Slides.pdf # Project presentation and insights
├── .gitignore                   # Files to exclude (caches, checkpoints)
├── README.md                    # Project documentation
└── requirements.txt             # List of Python dependencies


## 📊 Dataset
* **Source:** Sonar Dataset (208 total records).
* **Features:** 60 continuous frequency bands (energy values normalized between 0.0 and 1.0).
* **Target:** Binary classification — **Rock (R)** or **Mine (M)**.
* **Balance:** 53% Mines vs. 47% Rocks (relatively balanced).

## 🛠 Tools and Technologies
* **Language:** Python 3.x
* **Data Analysis:** `Pandas`, `NumPy`, `SciPy`
* **Visualization:** `Matplotlib`, `Seaborn`, `Missingno (for null analysis)`
* **Feature Engineering:** `Scikit-learn` (StandardScaler, LabelEncoder), `Statsmodels` (VIF analysis)
* **Machine Learning:** `Logistic Regression`, `SVM`, `Random Forest`, `LightGBM`
* **Deep Learning:** `TensorFlow` / `Keras` (Artificial Neural Networks)

## 🧪 Methods
1.  **Data Quality Check:** Handled 27 missing values across `Attribute7` and `Attribute12`.
2.  **Preprocessing:** Target encoding (Rock=0, Mine=1) and outlier treatment using 3x IQR capping.
3. **Statistical Distribution:** Analyzed the skewness and kurtosis of the continuous features. Many attributes exhibited non-normal distributions, which informed our decision to use robust scaling.
4.  **Feature Selection:** Reduced dimensionality from 60 features to **22 significant features** using statistical tests (T-tests/U-tests) to improve model efficiency and reduce noise.
5.  **Collinearity Analysis:** Performed Variance Inflation Factor (VIF) checks and Correlation Heatmaps to ensure feature independence.
6.  **Modeling:** Evaluated multiple architectures, including Logistic Regression, SVM, and Random Forests, before finalizing LightGBM and ANN for their superior handling of complex signal variance.


## Research Questions & Key Findings

**1. Which frequency bands are most critical for distinguishing Mines from Rocks?**
* **Finding:** Through statistical T-tests and feature importance ranking, attributes such as **Attribute36, Attribute45, and Attribute10** were identified as the most significant. Metal cylinders (Mines) show distinct energy peaks in these specific bands that are typically absent in natural Rock formations.

**2. Can dimensionality reduction improve model performance on a small dataset?**
* **Finding:** **Yes.** Reducing the feature space from 60 to **22 significant attributes** using statistical filtering significantly reduced noise. This prevented the models from overfitting to the small sample size, leading to a more generalized and stable performance on the test set.

**3. Which model architecture provides the highest safety margin (Recall)?**
* **Finding:** While LightGBM provided a high ROC-AUC (0.90), the **Artificial Neural Network (ANN)** achieved a superior **Recall of 90.91%**. In a maritime safety context, the ANN is the preferred model as it minimizes the risk of a "False Negative" (missing a mine).



## 💡 Key Insights
* **Crucial Predictors:** Features such as `Attribute36`, `Attribute45`, and `Attribute10` showed the highest separation power between classes.
* **Data Scarcity:** Given the small dataset, regularization (L1/L2) was essential in the Neural Network to prevent the model from simply "memorizing" the training data.
* **Signal Analysis:** Certain frequency bands exhibit higher energy variance in Mines compared to Rocks, allowing for clear statistical differentiation.

## 🖥️ Model Output
### Model Performance Summary:
* **LightGBM Performance:**
    * **ROC-AUC:** 0.9045
    * **Test Accuracy:** ~87.5%
* **Deep Learning Performance:**
    * **Recall for Mines:** ~95.4% (Crucial for safety-first applications).
    * **Validation Accuracy:** ~79.4

Note: The Neural Network demonstrated superior recall, which is vital for safety-critical applications.

*(Visualizations such as Correlation Heatmaps, Confusion Matrices, and ROC Curves are available within the Jupyter Notebook provided in this repo.)*

## ⚙️ How to Run This Project
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/your-username/sonar-signal-classification.git](https://github.com/your-username/sonar-signal-classification.git)
    cd sonar-signal-classification
    ```
2.  **Install Requirements:**
    ```bash
    pip install pandas scikit-learn tensorflow lightgbm seaborn matplotlib
    ```
3.  **Run the Notebook:**
    Launch Jupyter Notebook or Google Colab and open `Capstone_P2 (1).ipynb`.
4.  **Data:** Ensure `sonar1.csv` is in the root directory or update the path in the notebook.

## 🏁 Result & Conclusion
The project successfully demonstrates that statistical feature reduction combined with modern gradient boosting (LightGBM) and Neural Networks can achieve high predictive power even on small datasets. 
* **LightGBM** provided the best overall classification balance with an **AUC of 0.90**.
* The **ANN** achieved the mission-critical goal of **90%+ Recall**, ensuring maximum safety in mine detection scenarios.

## 🔮 Future Work
* **1D-CNN Integration:** Use Convolutional Neural Networks to treat the 60 features as a sequence/waveform for better feature extraction.
* **Data Augmentation:** Implement synthetic data generation (SMOTE or GANs) to increase the sample size.
* **Ensemble Stacking:** Combine LightGBM, XGBoost, and SVM into a Meta-Classifier to further boost accuracy.
* **Deployment:** Wrap the model in a FastAPI or Flask wrapper for real-time sonar signal processing.

## 👤 Author & Contact
* **Name:** Shristi Raushan
* **Email:** shristiraushan@gmail.com
* **LinkedIn:** [linkedin.com/in/your-profile](https://linkedin.com/in/shristi-raushan/)