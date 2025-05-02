# 🚀 AutoML SaaS Platform
An interactive web application built with Streamlit to automate the end-to-end machine learning workflow, from data upload and exploration to model training and evaluation. Designed for data analysts, business analysts, data scientists, and students looking to quickly gain insights and build predictive models without extensive coding.

<!-- Optional: Add a GIF or Screenshot here -->
<!-- ![App Demo](link_to_your_demo_gif_or_screenshot.png) -->

## ✨ Features

This platform provides a comprehensive suite of tools covering the typical ML lifecycle:

*   **📤 Data Upload:** Supports CSV, Excel (.xls, .xlsx), and JSON file formats. Includes sample datasets (Iris, Diabetes, Titanic) for quick testing.
*   **📊 Exploratory Data Analysis (EDA):**
    *   Automated generation of dataset overview (rows, columns, missing values).
    *   Column type detection (Numeric, Categorical, Binary, Datetime, ID, Text).
    *   Detailed column statistics and missing value analysis.
    *   Target variable analysis (distribution, imbalance checks, skewness).
    *   Automated insights (high cardinality, potential multicollinearity, skewed features).
*   **🎨 Custom Visualization Builder:** Create interactive plots on the fly:
    *   Histograms
    *   Box Plots (with optional grouping)
    *   Bar Plots (for categorical features)
    *   Scatter Plots (with optional color/size mapping)
    *   Correlation Heatmaps
    *   Pie Charts
    *   Line Plots (time-series or numeric vs. numeric)
*   **⚙️ Data Preprocessing:** Configurable steps to prepare data for modeling:
    *   Missing Value Imputation (Simple Median/Mode or KNN Imputation).
    *   Feature Scaling (StandardScaler).
    *   Categorical Encoding (LabelEncoder).
    *   Datetime Feature Extraction (Year, Month, Day, etc.).
    *   Handling of ID and Text columns (Dropping).
    *   **Advanced:** Feature Selection (SelectKBest) & Class Balancing (SMOTE, Random Over/Under Sampling - requires `imbalanced-learn`).
*   **🤖 Model Training:** Train multiple classification or regression models with ease:
    *   Supports: Logistic Regression, Random Forest, Gradient Boosting, KNN, Decision Tree, Linear Regression, Ridge, Lasso.
    *   Configurable train/test split ratio and random seed.
    *   Timeout mechanism to prevent excessively long training times.
*   **📈 Model Evaluation:** Compare and analyze model performance:
    *   Summary table with key metrics (Accuracy, F1, Precision, Recall for Classification; R², MSE, MAE, RMSE for Regression).
    *   Interactive comparison plots.
    *   Detailed analysis per model:
        *   Classification: Metrics, Classification Report, Confusion Matrix, ROC Curve (for binary).
        *   Regression: Metrics, Actual vs. Predicted plot, Residuals plot & distribution.
*   **💾 Download Results:** Export model performance summaries and test set predictions.

## 🛠️ Technology Stack

*   **Core:** Python 3.8+
*   **Web Framework:** Streamlit
*   **Data Handling:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn
*   **Visualization:** Plotly, Matplotlib, Seaborn
*   **Optional:** imbalanced-learn (for class balancing)
*   **Excel Support:** openpyxl

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

*   Python (version 3.8 or higher recommended)
*   `pip` (Python package installer)
*   `git` (for cloning the repository)

### Installation

1.  **Clone the repository:**
    ```
    git clone <your-repository-url> # Replace with your repo URL
    cd <repository-directory-name>
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```
    pip install -r requirements.txt
    ```
    *(Note: If you need class balancing features, uncomment `imbalanced-learn` in `requirements.txt` before installing.)*

### Running the Application

1.  Navigate to the project directory in your terminal (if you aren't already there).
2.  Run the Streamlit application:
    ```
    streamlit run your_app_script_name.py # Replace with the actual name of your Python script
    ```
3.  The application should automatically open in your default web browser.

## 🚦 Workflow

The application guides you through the following steps using the sidebar navigation:

1.  **Upload Data:** Upload your dataset or select a sample dataset. Confirm/select the target column and problem type (Classification/Regression).
2.  **Explore Data:** Review automated EDA insights, statistics, and create custom visualizations to understand your data.
3.  **Preprocess Data:** Configure and apply preprocessing steps like imputation, scaling, encoding, and feature handling.
4.  **Train Models:** Select the machine learning models you want to train and configure training options (test split, random seed).
5.  **Evaluate Models:** Compare the performance of trained models using tables and plots. Select individual models for detailed metric analysis, confusion matrices, ROC curves, or regression plots. Download results and predictions.

## 🤝 Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the GitHub repository. If you'd like to contribute code:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some amazing feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. *(You should create a LICENSE file containing the MIT License text)*

## ✍️ Author

*   **Arkaprabha Banerjee**

---

*This README was generated based on the project code and common best practices.*
