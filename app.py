import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
import base64
import logging
import traceback
import time
import json
from datetime import datetime
from contextlib import contextmanager
import signal
import re
# Machine Learning Imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    r2_score, mean_squared_error, mean_absolute_error, precision_recall_curve,
    roc_curve, auc, precision_score, recall_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
# import logging
import warnings
warnings.filterwarnings("ignore", message="imblearn not found")

# Set up a logger
logger = logging.getLogger(__name__)  # Use the current module name
logger.setLevel(logging.DEBUG)  # Set the logging level (could be INFO, WARNING, etc.)

# Imbalanced Learn (optional, handle import error)
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logger.warning("imblearn not found. Class balancing features will be disabled. Install with: pip install imbalanced-learn")

# --- Configuration & Setup ---

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoML-SaaS")

# Page configuration
st.set_page_config(
    page_title="AutoML SaaS Platform", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)
# Hide traceback in the Streamlit app for cleaner user experience
# Errors are still logged to the console
st.set_option('client.showErrorDetails', False) 

# --- Session State Initialization ---
# Use functions for cleaner initialization if needed, but direct check is fine
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_proc' not in st.session_state:
    st.session_state.df_proc = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'ct' not in st.session_state: # Column Types dictionary
    st.session_state.ct = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'trained_models' not in st.session_state: # Store trained model instances
    st.session_state.trained_models = {} 
if 'X_test' not in st.session_state: # Store test set for evaluation
    st.session_state.X_test = None
if 'y_test' not in st.session_state: # Store test target for evaluation
    st.session_state.y_test = None
if 'app_mode' not in st.session_state: # Store current navigation state
    st.session_state.app_mode = "Upload Data" 
    
# --- Error Handling ---

@contextmanager
def st_exception_handler(message="An error occurred"):
    """
    Context manager for handling exceptions in Streamlit apps.
    Logs the full error and traceback, displays a user-friendly message.
    """
    try:
        yield
    except Exception as e:
        st.error(f"⚠️ {message}: {str(e)}")
        logger.error(f"{message}: {str(e)}\n{traceback.format_exc()}")
        # Optionally stop execution if the error is critical
        # st.stop() 

# --- Helper Functions ---

@st.cache_data # Cache data loading based on file content
def load_data(uploaded_file):
    """
    Loads data from uploaded file (CSV, Excel, JSON).
    Includes robust parsing, column name cleaning, and auto date conversion.
    Returns: Tuple (DataFrame, error_message or None)
    """
    try:
        ext = uploaded_file.name.split('.')[-1].lower()
        df = None
        
        if ext == 'csv':
            # Try multiple common separators and encodings
            try:
                # Try with auto-detection first (often works well)
                df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
            except Exception as e1:
                logger.warning(f"Auto-detect CSV failed: {e1}. Trying common options...")
                uploaded_file.seek(0) # Reset file pointer
                try:
                    df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
                except Exception as e2:
                    logger.warning(f"Comma-separated CSV failed: {e2}. Trying latin1 encoding...")
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, sep=',', encoding='latin1')
                    except Exception as e3:
                        logger.warning(f"Comma/latin1 failed: {e3}. Trying tab-separated...")
                        uploaded_file.seek(0)
                        try:
                           df = pd.read_csv(uploaded_file, sep='\t', encoding='utf-8')
                        except Exception as e4:
                           logger.error(f"All CSV parsing attempts failed: {e4}")
                           return None, f"Failed to parse CSV file. Ensure it's correctly formatted and encoded (UTF-8 preferred). Error: {e4}"
                           
        elif ext in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        elif ext == 'json':
            try:
                df = pd.read_json(uploaded_file)
            except ValueError as e: # Handle complex JSON structures
                logger.warning(f"Direct JSON read failed: {e}. Trying line-delimited...")
                uploaded_file.seek(0)
                try:
                    df = pd.read_json(uploaded_file, lines=True)
                except Exception as e_lines:
                    logger.error(f"JSON parsing failed: {e_lines}")
                    return None, f"Failed to parse JSON. Ensure it's a standard format (object array or line-delimited). Error: {e_lines}"
        else:
            return None, f"Unsupported file format: '{ext}'. Please upload CSV, Excel, or JSON files."
        
        if df is None:
             return None, "Could not load data from file."
             
        if df.empty:
            return None, "The uploaded file contains no data or failed to parse."
        
        # --- Data Cleaning Post-Load ---
        # Clean column names: remove leading/trailing spaces, replace spaces with underscores
        # df.columns = [str(col).strip().replace(' ', '_').replace('[^A-Za-z0-9_]+', '', regex=True) for col in df.columns]
        df.columns = [re.sub(r'[^A-Za-z0-9_]+', '', str(col).strip().replace(' ', '_')) for col in df.columns]

        df.columns = [f"col_{i}" if col == "" else col for i, col in enumerate(df.columns)] # Handle empty column names
        
        # Auto-convert potential date columns (more robustly)
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Attempt conversion, if >80% success, apply it
                converted = pd.to_datetime(df[col], errors='coerce')
                if converted.notna().mean() > 0.8:
                    df[col] = converted
                    logger.info(f"Auto-converted column '{col}' to datetime.")
            except Exception: 
                # Ignore columns that cause errors during conversion attempt
                pass 
                    
        return df, None # Return DataFrame and None for error
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}\n{traceback.format_exc()}")
        return None, f"An unexpected error occurred during data loading: {str(e)}"

#@st.cache_data # Caching based on DataFrame hash
def auto_detect_target(_df):
    """
    Heuristically detects the target column based on common names and properties.
    Input: DataFrame
    Returns: Detected target column name (string)
    """
    df = _df.copy() # Work on a copy to avoid side effects if we modify it later
    try:
        # Priority 1: Common target column names (case-insensitive)
        common_names = ['target', 'label', 'class', 'y', 'output', 'result', 
                        'outcome', 'dependent_variable', 'response', 'prediction',
                        'status', 'score', 'grade', # Classification-like
                        'price', 'sales', 'revenue', 'value', 'amount', # Regression-like
                        'churn', 'conversion', 'is_fraud', 'defaulted'] # Specific domains
        
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        for name in common_names:
            if name in df_cols_lower:
                logger.info(f"Detected target by common name: {df_cols_lower[name]}")
                return df_cols_lower[name]

        # Priority 2: Column named 'target' or similar suffix/prefix
        for col in df.columns:
            col_lower = col.lower()
            if col_lower.endswith('_target') or col_lower.startswith('target_') or \
               col_lower.endswith('_label') or col_lower.startswith('label_') or \
               col_lower.endswith('_class') or col_lower.startswith('class_'):
                 logger.info(f"Detected target by suffix/prefix: {col}")
                 return col

        # Priority 3: Last column if it has reasonable cardinality (not ID-like, not constant)
        last_col = df.columns[-1]
        if 1 < df[last_col].nunique() < len(df) * 0.8: # Avoid high cardinality or constant
             logger.info(f"Detected target as last column with reasonable cardinality: {last_col}")
             return last_col

        # Priority 4: Find columns with low-to-moderate cardinality (potential classification targets)
        candidates = {}
        for col in df.columns:
             unique_count = df[col].nunique()
             # Exclude ID-like columns (very high unique values) and constant columns
             if 1 < unique_count <= max(20, min(50, len(df) * 0.1)): # Flexible threshold
                 candidates[col] = unique_count
        
        if candidates:
             # Prefer column with lowest cardinality among candidates
             best_candidate = min(candidates, key=candidates.get)
             logger.info(f"Detected target by low/moderate cardinality: {best_candidate}")
             return best_candidate

        # Fallback: Default to the last column if no better heuristic worked
        logger.warning("Target detection heuristics failed, defaulting to the last column.")
        return df.columns[-1]
        
    except Exception as e:
        logger.error(f"Error detecting target column: {str(e)}\n{traceback.format_exc()}")
        # Safe fallback
        return df.columns[-1] if not df.empty else None

#@st.cache_data # Caching based on Series hash
def detect_problem_type(_y):
    """
    Detects ML problem type (Classification/Regression/Time Series) based on target series.
    Input: Target pandas Series
    Returns: Problem type (string)
    """
    y = _y.dropna() # Analyze non-missing values
    if y.empty:
        logger.warning("Target column is empty or all NaN, defaulting problem type to Classification.")
        return "Classification" 
        
    try:
        # 1. Datetime suggests Time Series (though regression/classification on time is possible)
        # We treat 'Time Series' as a special category for potential future dedicated handling.
        if pd.api.types.is_datetime64_any_dtype(y):
            logger.info("Detected problem type: Time Series (based on datetime target)")
            # For now, treat Time Series forecasting as Regression for modeling purposes
            # Could be refined later to support specific time series models.
            return "Regression" 
            
        # 2. Categorical/Object type strongly suggests Classification
        if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y):
             logger.info("Detected problem type: Classification (based on object/category dtype)")
             return "Classification"
             
        # 3. Numeric type analysis
        if pd.api.types.is_numeric_dtype(y):
            unique_count = y.nunique()
            
            # Integer type with very few unique values -> likely Classification
            # Threshold: e.g., < 20 unique values or < 5% of total data points (whichever is larger, up to a max like 50)
            classification_threshold = max(20, min(50, int(len(y) * 0.05)))
            
            if pd.api.types.is_integer_dtype(y) and unique_count <= classification_threshold:
                logger.info(f"Detected problem type: Classification (based on integer dtype with {unique_count} unique values)")
                return "Classification"
                
            # Binary numeric values (0/1) -> Classification
            elif unique_count == 2 and y.isin([0, 1]).all():
                 logger.info("Detected problem type: Classification (based on binary numeric values)")
                 return "Classification"
                 
            # Otherwise, assume Regression for numeric types
            else:
                logger.info("Detected problem type: Regression (based on numeric dtype)")
                return "Regression"
                
        # Fallback if type is unusual
        logger.warning(f"Could not reliably determine problem type for dtype {y.dtype}. Defaulting to Classification.")
        return "Classification"
        
    except Exception as e:
        logger.error(f"Error detecting problem type: {str(e)}\n{traceback.format_exc()}")
        return "Classification" # Safe fallback

#@st.cache_data # Cache based on DataFrame hash
def identify_column_types(_df):
    """
    Identifies column types (numeric, categorical, datetime, binary, id, text).
    Input: DataFrame
    Returns: Dictionary mapping type name to list of column names.
    """
    df = _df.copy()
    ct = {
        'numeric': [], 'categorical': [], 'datetime': [], 
        'binary': [], 'id': [], 'text': [], 'other': []
    }
    n_rows = len(df)
    if n_rows == 0: return ct # Handle empty DataFrame

    for col in df.columns:
        try:
            dtype = df[col].dtype
            nunique = df[col].nunique(dropna=False) # Include NaN in uniqueness check for some cases
            
            # 1. Handle columns with only NaN values
            if df[col].isna().all():
                ct['other'].append(col) # Treat as 'other' or potentially drop later
                logger.debug(f"Column '{col}': Type 'other' (all NaN)")
                continue

            # 2. Datetime
            if pd.api.types.is_datetime64_any_dtype(dtype):
                ct['datetime'].append(col)
                logger.debug(f"Column '{col}': Type 'datetime'")
                continue

            # 3. Numeric Types
            if pd.api.types.is_numeric_dtype(dtype):
                # Check for ID-like numeric columns (high cardinality, often integers, sometimes monotonic)
                # Threshold: >95% unique values and integer-like (or just very high unique count)
                if nunique >= n_rows * 0.95 and (pd.api.types.is_integer_dtype(dtype) or nunique / n_rows > 0.99):
                     ct['id'].append(col)
                     logger.debug(f"Column '{col}': Type 'id' (numeric, high cardinality)")
                # Check for Binary numeric (0/1 or two distinct numeric values)
                elif nunique <= 2:
                     ct['binary'].append(col)
                     logger.debug(f"Column '{col}': Type 'binary' (numeric)")
                # Otherwise, it's a standard numeric column
                else:
                     ct['numeric'].append(col)
                     logger.debug(f"Column '{col}': Type 'numeric'")
                continue # Move to next column

            # 4. Categorical / Object / Text Types
            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                # Check for ID-like string columns (high cardinality)
                if nunique >= n_rows * 0.95:
                    ct['id'].append(col)
                    logger.debug(f"Column '{col}': Type 'id' (object/string, high cardinality)")
                # Check for Binary object/category (two distinct values)
                elif nunique <= 2:
                    ct['binary'].append(col)
                    logger.debug(f"Column '{col}': Type 'binary' (object/category)")
                # Check for Text-like columns (high average string length, moderate-high cardinality)
                # Heuristic: avg length > 30 chars, or nunique > 50% of rows (but not ID)
                elif df[col].astype(str).str.len().mean(skipna=True) > 30 or (nunique > n_rows * 0.5 and nunique < n_rows * 0.95):
                    ct['text'].append(col)
                    logger.debug(f"Column '{col}': Type 'text'")
                # Otherwise, assume Categorical
                else:
                    ct['categorical'].append(col)
                    logger.debug(f"Column '{col}': Type 'categorical'")
                continue # Move to next column

            # 5. Other types (Boolean, etc.) - Treat boolean as binary for simplicity
            if pd.api.types.is_bool_dtype(dtype):
                ct['binary'].append(col)
                logger.debug(f"Column '{col}': Type 'binary' (boolean)")
                continue

            # Fallback for unhandled types
            ct['other'].append(col)
            logger.warning(f"Column '{col}' has unhandled dtype '{dtype}'. Classifying as 'other'.")

        except Exception as e:
            logger.error(f"Error identifying type for column '{col}': {str(e)}\n{traceback.format_exc()}")
            ct['other'].append(col) # Assign to 'other' if analysis fails

    logger.info(f"Column Types Identified: { {k: len(v) for k, v in ct.items()} }")
    return ct

# --- Preprocessing Function ---

def preprocess_data(df_orig, target_col, ct, scale_numeric=True, encode_categorical=True, impute_method='simple', datetime_handling='extract', drop_id=True, drop_text=True):
    """
    Preprocesses the DataFrame based on identified column types and user options.
    Handles imputation, scaling, encoding, datetime features, and dropping irrelevant columns.
    Returns: Tuple (processed_DataFrame, label_encoders_dict or None, error_message or None)
    """
    try:
        df = df_orig.copy()
        logger.info(f"Starting preprocessing. Options: scale={scale_numeric}, encode={encode_categorical}, impute='{impute_method}', datetime='{datetime_handling}', drop_id={drop_id}, drop_text={drop_text}")

        # 1. Drop ID and Text columns (if requested)
        cols_to_drop = []
        if drop_id:
            cols_to_drop.extend(ct.get('id', []))
        if drop_text:
            cols_to_drop.extend(ct.get('text', []))
        # Also drop 'other' columns identified previously
        cols_to_drop.extend(ct.get('other', []))
        
        # Ensure target column is not dropped
        cols_to_drop = [col for col in cols_to_drop if col != target_col]
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop, errors='ignore')
            logger.info(f"Dropped columns: {cols_to_drop}")
            # Update column type lists after dropping
            for col_type_list in ct.values():
                 for dropped_col in cols_to_drop:
                      if dropped_col in col_type_list:
                           col_type_list.remove(dropped_col)

        # 2. Handle Datetime columns
        if datetime_handling == 'extract' and ct.get('datetime'):
            for col in ct['datetime']:
                if col in df.columns: # Check if not dropped
                    try:
                        df[f"{col}_year"] = df[col].dt.year
                        df[f"{col}_month"] = df[col].dt.month
                        df[f"{col}_day"] = df[col].dt.day
                        df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                        df[f"{col}_hour"] = df[col].dt.hour
                        # Add more extractions if needed (e.g., weekofyear, quarter)
                        df = df.drop(columns=[col])
                        logger.info(f"Extracted features from datetime column '{col}' and dropped original.")
                        # Update ct: remove from datetime, add new features to numeric
                        if col in ct['datetime']: ct['datetime'].remove(col)
                        new_dt_cols = [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek", f"{col}_hour"]
                        ct['numeric'].extend([c for c in new_dt_cols if c in df.columns]) # Add newly created columns
                    except Exception as e_dt:
                         logger.warning(f"Could not extract features from datetime column '{col}': {e_dt}. Skipping.")
        elif datetime_handling == 'drop' and ct.get('datetime'):
            dt_cols_to_drop = [col for col in ct['datetime'] if col in df.columns and col != target_col]
            df = df.drop(columns=dt_cols_to_drop, errors='ignore')
            logger.info(f"Dropped datetime columns: {dt_cols_to_drop}")
            for col in dt_cols_to_drop:
                if col in ct['datetime']: ct['datetime'].remove(col)
        # Add 'numeric' or 'cyclic' options later if needed

        # 3. Imputation (handle missing values)
        numeric_cols_for_impute = [col for col in ct.get('numeric', []) if col in df.columns and col != target_col]
        categorical_cols_for_impute = [col for col in ct.get('categorical', []) + ct.get('binary', []) if col in df.columns and col != target_col]

        if impute_method == 'simple':
            # Numeric: Median imputation
            if numeric_cols_for_impute:
                num_imputer = SimpleImputer(strategy='median')
                df[numeric_cols_for_impute] = num_imputer.fit_transform(df[numeric_cols_for_impute])
                logger.info(f"Applied median imputation to numeric columns: {numeric_cols_for_impute}")
            # Categorical: Mode imputation
            if categorical_cols_for_impute:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                df[categorical_cols_for_impute] = cat_imputer.fit_transform(df[categorical_cols_for_impute])
                logger.info(f"Applied mode imputation to categorical/binary columns: {categorical_cols_for_impute}")

        elif impute_method == 'knn':
             # KNN imputation (only for numeric features, requires no NaNs in other features used for KNN)
            if numeric_cols_for_impute:
                 # First, impute non-numeric cols with mode temporarily if they have NaNs, to allow KNN to work
                 temp_cat_imputer = SimpleImputer(strategy='most_frequent')
                 if categorical_cols_for_impute and df[categorical_cols_for_impute].isnull().any().any():
                      df[categorical_cols_for_impute] = temp_cat_imputer.fit_transform(df[categorical_cols_for_impute])
                      logger.debug("Temporarily imputed categoricals with mode for KNN.")
                      
                 try:
                     knn_imputer = KNNImputer(n_neighbors=5)
                     df[numeric_cols_for_impute] = knn_imputer.fit_transform(df[numeric_cols_for_impute])
                     logger.info(f"Applied KNN imputation (k=5) to numeric columns: {numeric_cols_for_impute}")
                 except ValueError as e_knn:
                      logger.warning(f"KNN Imputation failed: {e_knn}. Falling back to median imputation for numeric columns.")
                      num_imputer = SimpleImputer(strategy='median')
                      df[numeric_cols_for_impute] = num_imputer.fit_transform(df[numeric_cols_for_impute])
                 
            # Impute categorical normally after KNN (or if KNN wasn't used)
            if categorical_cols_for_impute:
                 cat_imputer = SimpleImputer(strategy='most_frequent')
                 df[categorical_cols_for_impute] = cat_imputer.fit_transform(df[categorical_cols_for_impute])
                 if not numeric_cols_for_impute: # Log only if KNN wasn't attempted
                      logger.info(f"Applied mode imputation to categorical/binary columns: {categorical_cols_for_impute}")
        # Add more methods like 'iterative' later if needed

        # 4. Encoding Categorical Features
        label_encoders = {}
        # Combine categorical and binary for encoding, exclude target
        cols_to_encode = [col for col in ct.get('categorical', []) + ct.get('binary', []) if col in df.columns and col != target_col]
        
        if encode_categorical and cols_to_encode:
            # Option 1: Simple Label Encoding (can be problematic for non-tree models)
            # We'll use this for simplicity as per the original code, but OneHot is generally better
            for col in cols_to_encode:
                try:
                    le = LabelEncoder()
                    # Convert to string first to handle mixed types or numeric representations
                    df[col] = le.fit_transform(df[col].astype(str)) 
                    label_encoders[col] = le # Store encoder if needed later (e.g., inverse transform)
                except Exception as e_enc:
                     logger.warning(f"Label encoding failed for column '{col}': {e_enc}. Skipping.")
            logger.info(f"Applied label encoding to columns: {cols_to_encode}")
            # Option 2: One-Hot Encoding (consider adding as an option)
            # df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
            # logger.info(f"Applied one-hot encoding to columns: {cols_to_encode}")

        # 5. Scaling Numeric Features
        cols_to_scale = [col for col in ct.get('numeric', []) if col in df.columns and col != target_col]
        # Also include label-encoded binary columns if needed (often optional)
        # cols_to_scale.extend([col for col in ct.get('binary', []) if col in df.columns and col != target_col and col in label_encoders])
        
        if scale_numeric and cols_to_scale:
            scaler = StandardScaler() # Or RobustScaler() for outlier robustness
            try:
                 df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
                 logger.info(f"Applied StandardScaler to numeric columns: {cols_to_scale}")
            except Exception as e_scale:
                 logger.warning(f"Scaling failed: {e_scale}. Skipping scaling.")

        # Ensure target column is numeric for regression if it wasn't already
        if st.session_state.problem_type == "Regression" and target_col in df.columns:
             if not pd.api.types.is_numeric_dtype(df[target_col]):
                 try:
                     df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
                     # Check if conversion introduced NaNs and handle if necessary
                     if df[target_col].isnull().any():
                          median_target = df[target_col].median()
                          df[target_col] = df[target_col].fillna(median_target)
                          logger.warning(f"Target column '{target_col}' converted to numeric and NaNs imputed with median ({median_target}).")
                 except Exception as e_targ_conv:
                     logger.error(f"Failed to convert regression target '{target_col}' to numeric: {e_targ_conv}")
                     return df_orig, None, f"Failed to convert regression target '{target_col}' to numeric."

        # Final check for NaNs after all steps
        if df.isnull().any().any():
             nan_cols = df.columns[df.isnull().any()].tolist()
             logger.warning(f"NaN values remain after preprocessing in columns: {nan_cols}. Applying final median/mode fill.")
             for col in nan_cols:
                 if col == target_col: continue # Avoid filling target again if already handled
                 if pd.api.types.is_numeric_dtype(df[col]):
                     df[col] = df[col].fillna(df[col].median())
                 else:
                     df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Missing")

        logger.info("Preprocessing finished.")
        return df, label_encoders, None # Return processed DataFrame, encoders, and None for error

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}\n{traceback.format_exc()}")
        return df_orig, None, f"An unexpected error occurred during preprocessing: {str(e)}"


# --- Model Training & Evaluation ---

# Model Definitions (keep them simple, hyperparameters can be tuned later)
CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    #"AdaBoost": AdaBoostClassifier(random_state=42), # Can be slow
    "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
    #"Support Vector Machine": SVC(probability=True, random_state=42), # Can be very slow
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(n_jobs=-1),
    "Ridge Regression": Ridge(random_state=42),
    "Lasso Regression": Lasso(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    #"AdaBoost": AdaBoostRegressor(random_state=42), # Can be slow
    "K-Nearest Neighbors": KNeighborsRegressor(n_jobs=-1),
    #"Support Vector Regressor": SVR(), # Can be very slow
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

# Timeout function using signal (Unix-like systems only)
# def train_with_timeout(model, X_train, y_train, timeout_seconds=60):
#     """Trains a model with a specified timeout (works on Unix-like systems)."""
#     # Check if signal is available (won't work on Windows)
#     if not hasattr(signal, 'SIGALRM'):
#         logger.warning("Timeout functionality requires Unix-like OS (signal.SIGALRM not available). Training without timeout.")
#         try:
#             start_time = time.time()
#             model.fit(X_train, y_train)
#             end_time = time.time()
#             logger.info(f"Model trained in {end_time - start_time:.2f} seconds (no timeout).")
#             return model, None # Return model and no error
#         except Exception as e:
#              logger.error(f"Error during model training (no timeout): {e}\n{traceback.format_exc()}")
#              return None, str(e) # Return None for model and the error message
# Timeout function using signal (Unix-like systems only) - MODIFIED TO REMOVE SIGNAL
def train_with_timeout(model, X_train, y_train, timeout_seconds=60):
    """
    Trains a model, logging the time taken. 
    NOTE: The original signal-based timeout has been removed 
    due to incompatibility with Streamlit's threading and non-Unix systems.
    Long-running models will now run to completion.
    """
    logger.info(f"Attempting to train {type(model).__name__}. Timeout functionality (via signal) is disabled.")
    start_time = time.time()
    try:
        # Directly fit the model without using signal for timeout
        model.fit(X_train, y_train)
        end_time = time.time()
        training_duration = end_time - start_time
        logger.info(f"Model {type(model).__name__} trained successfully in {training_duration:.2f} seconds.")
        # If you want to enforce a soft limit check *after* training (less useful but possible):
        # if training_duration > timeout_seconds:
        #     logger.warning(f"Model {type(model).__name__} training ({training_duration:.2f}s) exceeded the nominal timeout limit of {timeout_seconds}s.")
        
        return model, None # Return trained model, no error

    except Exception as e:
        end_time = time.time()
        logger.error(f"Error during model training for {type(model).__name__} after {end_time - start_time:.2f} seconds: {e}\n{traceback.format_exc()}")
        return None, str(e) # Return None for model, the error message
    # No finally block needed specifically for signal cleanup anymore

    # --- Timeout logic for Unix-like systems ---
    class TimeoutException(Exception): pass

    def timeout_handler(signum, frame):
        raise TimeoutException(f"Model training exceeded timeout of {timeout_seconds} seconds.")

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    start_time = time.time()
    try:
        model.fit(X_train, y_train)
        signal.alarm(0) # Disable the alarm if fit completes successfully
        end_time = time.time()
        logger.info(f"Model trained in {end_time - start_time:.2f} seconds.")
        return model, None # Return trained model, no error
    except TimeoutException as e:
        logger.warning(str(e))
        return None, str(e) # Return None for model, timeout error message
    except Exception as e:
        signal.alarm(0) # Disable alarm on other exceptions too
        logger.error(f"Error during model training: {e}\n{traceback.format_exc()}")
        return None, str(e) # Return None for model, other error message
    finally:
        # Restore the original signal handler
        signal.signal(signal.SIGALRM, old_handler)

#@st.cache_data(show_spinner=False) # Caching model training can be tricky due to model objects
def train_and_evaluate_models(_X_train, _X_test, _y_train, _y_test, models_dict, problem_type, timeout_per_model=120):
    """
    Trains multiple models, evaluates them, and handles errors/timeouts.
    Returns: Tuple (results_DataFrame, predictions_dict, trained_models_dict)
    """
    results = []
    predictions = {}
    trained_models_output = {}
    
    # Use Streamlit containers for progress updates
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    total_models = len(models_dict)

    for i, (name, model_instance) in enumerate(models_dict.items()):
        current_progress = (i + 1) / total_models
        status_text.info(f"⏳ Training Model {i+1}/{total_models}: {name}...")
        
        try:
            # Train with timeout
            trained_model, error_msg = train_with_timeout(
                model_instance, _X_train, _y_train, timeout_seconds=timeout_per_model
            )
            
            if error_msg or trained_model is None:
                st.warning(f"Skipping model '{name}': {error_msg or 'Training failed.'}")
                results.append({"Model": name, "Status": "Failed/Timeout", "Details": error_msg or 'Unknown Error'})
                continue # Skip to the next model
                
            # --- Evaluation ---
            y_pred = trained_model.predict(_X_test)
            
            # Store results and predictions
            predictions[name] = y_pred
            trained_models_output[name] = trained_model # Store the trained model object
            
            model_metrics = {"Model": name, "Status": "Success"}
            
            if problem_type == "Classification":
                # Use zero_division=0 to avoid warnings for metrics like precision
                acc = accuracy_score(_y_test, y_pred)
                f1 = f1_score(_y_test, y_pred, average='weighted', zero_division=0)
                precision = precision_score(_y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(_y_test, y_pred, average='weighted', zero_division=0)
                
                model_metrics.update({
                    "Accuracy": acc, "F1_Score": f1, "Precision": precision, "Recall": recall
                })
                
            else: # Regression
                r2 = r2_score(_y_test, y_pred)
                mse = mean_squared_error(_y_test, y_pred)
                mae = mean_absolute_error(_y_test, y_pred)
                rmse = np.sqrt(mse)
                
                model_metrics.update({
                    "R2_Score": r2, "MSE": mse, "MAE": mae, "RMSE": rmse
                })
            
            results.append(model_metrics)
            
            status_text.info(f"✅ Trained Model {i+1}/{total_models}: {name}")
            
        except Exception as e:
            # Catch any unexpected errors during the loop for a specific model
            error_details = f"Unexpected error training {name}: {str(e)}"
            logger.error(f"{error_details}\n{traceback.format_exc()}")
            st.warning(f"Skipping model '{name}': {error_details}")
            results.append({"Model": name, "Status": "Failed", "Details": str(e)})
            continue
        finally:
             # Update progress bar regardless of success or failure
             progress_bar.progress(current_progress)

    progress_bar.empty() # Remove progress bar after completion
    status_text.success(f"🏁 Model training and evaluation finished for {len(predictions)} models.")
    
    results_df = pd.DataFrame(results)
    return results_df, predictions, trained_models_output

def plot_model_performance(results_df, problem_type):
    """Generates interactive plots comparing model performance."""
    if results_df.empty or 'Status' not in results_df.columns:
        st.warning("No successful model results available to plot.")
        return

    # Filter out failed models
    successful_results = results_df[results_df['Status'] == 'Success'].copy()
    if successful_results.empty:
        st.warning("No models trained successfully.")
        return

    st.subheader("📊 Model Performance Comparison")

    try:
        if problem_type == "Classification":
            metrics = ["Accuracy", "F1_Score", "Precision", "Recall"]
            # Ensure metrics exist in the dataframe
            metrics = [m for m in metrics if m in successful_results.columns]
            if not metrics:
                 st.warning("No standard classification metrics found in results.")
                 return
                 
            successful_results[metrics] = successful_results[metrics].astype(float) # Ensure numeric type for plotting
            
            # Melt for easier plotting with plotly express
            results_melted = successful_results.melt(
                id_vars="Model", 
                value_vars=metrics, 
                var_name="Metric", 
                value_name="Score"
            )
            
            fig = px.bar(
                results_melted, 
                x="Model", 
                y="Score", 
                color="Metric", 
                barmode='group', 
                title="Classification Model Performance",
                labels={"Score": "Metric Value"},
                text='Score' # Display score on bars
            )
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig.update_layout(yaxis_range=[0, 1.05], uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig, use_container_width=True)
            
        else: # Regression
            # Plot R2 Score (higher is better)
            if "R2_Score" in successful_results.columns:
                 successful_results["R2_Score"] = successful_results["R2_Score"].astype(float)
                 fig_r2 = px.bar(
                     successful_results.sort_values("R2_Score", ascending=False), 
                     x="Model", 
                     y="R2_Score", 
                     title="Regression Model Performance - R² Score (Higher is better)",
                     color="R2_Score",
                     color_continuous_scale="Blues",
                     text='R2_Score'
                 )
                 fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                 st.plotly_chart(fig_r2, use_container_width=True)
            else:
                 st.warning("R2 Score not found in results.")

            # Plot Error Metrics (lower is better)
            error_metrics = ["RMSE", "MAE", "MSE"]
            error_metrics = [m for m in error_metrics if m in successful_results.columns]
            if error_metrics:
                 successful_results[error_metrics] = successful_results[error_metrics].astype(float)
                 results_melted_err = successful_results.melt(
                     id_vars="Model", 
                     value_vars=error_metrics, 
                     var_name="Metric", 
                     value_name="Error"
                 )
                 fig_err = px.bar(
                     results_melted_err, 
                     x="Model", 
                     y="Error", 
                     color="Metric", 
                     barmode='group', 
                     title="Regression Model Performance - Error Metrics (Lower is better)",
                     labels={"Error": "Error Value"},
                     text='Error'
                 )
                 fig_err.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                 fig_err.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                 st.plotly_chart(fig_err, use_container_width=True)
            else:
                 st.warning("No standard error metrics (RMSE, MAE, MSE) found in results.")

    except Exception as e:
        logger.error(f"Error plotting model performance: {e}\n{traceback.format_exc()}")
        st.error(f"Could not generate performance plots: {e}")

def show_model_details(selected_model_name, trained_model, X_test, y_test, y_pred, problem_type):
    """Displays detailed metrics and plots for a specific selected model."""
    st.write(f"### Detailed Evaluation: {selected_model_name}")
    
    try:
        if problem_type == "Classification":
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
                st.metric("Weighted F1 Score", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                st.metric("Weighted Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
                st.metric("Weighted Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
            
            with col2:
                # Classification report
                st.text("Classification Report:")
                try:
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.3f}"))
                except Exception as e_report:
                    st.warning(f"Could not generate classification report: {e_report}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            try:
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(
                    cm, 
                    text_auto=True, 
                    labels=dict(x="Predicted Label", y="True Label", color="Count"),
                    x=np.unique(y_test), # Use actual labels if possible
                    y=np.unique(y_test),
                    color_continuous_scale='Blues'
                )
                fig_cm.update_layout(title="Confusion Matrix")
                st.plotly_chart(fig_cm, use_container_width=True)
            except Exception as e_cm:
                st.warning(f"Could not generate confusion matrix: {e_cm}")
            
            # ROC curve (if binary classification and probabilities available)
            if len(np.unique(y_test)) == 2 and hasattr(trained_model, "predict_proba"):
                st.subheader("ROC Curve")
                try:
                    y_prob = trained_model.predict_proba(X_test)[:, 1] # Probability of positive class
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc = px.area(
                        x=fpr, y=tpr,
                        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                        title=f"Receiver Operating Characteristic (AUC = {roc_auc:.4f})"
                    )
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
                    fig_roc.update_xaxes(constrain='domain')
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception as e_roc:
                    st.warning(f"Could not generate ROC curve: {e_roc}")
                    
        else: # Regression
            st.subheader("Regression Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")
                st.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test, y_pred):.4f}")
            with col2:
                st.metric("Mean Absolute Error (MAE)", f"{mean_absolute_error(y_test, y_pred):.4f}")
                st.metric("Root Mean Squared Error (RMSE)", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
            # col3 can be used for other metrics like Median Absolute Error if needed
            
            # Actual vs Predicted plot
            st.subheader("Actual vs. Predicted Values")
            try:
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig_ap = px.scatter(
                    x=y_test, y=y_pred, 
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'}, 
                    title="Actual vs. Predicted",
                    trendline="ols", # Add a regression line to see the trend
                    trendline_color_override="red"
                )
                # Add y=x line for reference
                fig_ap.add_shape(type='line', line=dict(dash='dash'), 
                                 x0=min_val, y0=min_val, x1=max_val, y1=max_val)
                fig_ap.update_layout(xaxis_range=[min_val,max_val], yaxis_range=[min_val,max_val])
                st.plotly_chart(fig_ap, use_container_width=True)
            except Exception as e_ap:
                 st.warning(f"Could not generate Actual vs Predicted plot: {e_ap}")

            # Residuals plot
            st.subheader("Residuals Analysis")
            try:
                residuals = y_test - y_pred
                # Residuals vs Predicted
                fig_rp = px.scatter(
                    x=y_pred, y=residuals, 
                    labels={'x': 'Predicted Values', 'y': 'Residuals (Actual - Predicted)'}, 
                    title="Residuals vs. Predicted Values"
                )
                # Add horizontal line at y=0
                fig_rp.add_hline(y=0, line_dash="dash")
                st.plotly_chart(fig_rp, use_container_width=True)
                
                # Residuals Distribution
                fig_rh = px.histogram(
                    residuals, nbins=50, title="Distribution of Residuals"
                )
                st.plotly_chart(fig_rh, use_container_width=True)
            except Exception as e_res:
                 st.warning(f"Could not generate Residuals plots: {e_res}")

    except Exception as e:
        logger.error(f"Error showing model details for {selected_model_name}: {e}\n{traceback.format_exc()}")
        st.error(f"Could not display details for model {selected_model_name}: {e}")

# --- Visualization & EDA Functions ---

#@st.cache_data # Cache plots based on df hash and arguments
def custom_plot_builder(_df, _ct):
    """Allows users to create custom plots based on column types."""
    st.subheader("🛠️ Custom Plot Builder")
    
    # Use a copy to prevent modifying cached data if needed
    df = _df.copy()
    ct = _ct.copy() 
    
    plot_options = ["Histogram", "Box Plot", "Bar Plot (Categorical)", "Scatter Plot", 
                    "Correlation Heatmap", "Pie Chart", "Line Plot (Time/Numeric)"]
    plot_type = st.selectbox("Select plot type", plot_options)
    
    try: # Wrap plot generation in try-except
        if plot_type == "Histogram":
            num_cols = ct.get('numeric', [])
            if num_cols:
                col = st.selectbox("Select numeric column", num_cols)
                if col:
                    bins = st.slider("Number of bins", 5, 100, 30, key="hist_bins")
                    color_by = st.selectbox("Color by (optional categorical)", [None] + ct.get('categorical', []) + ct.get('binary', []), key="hist_color")
                    fig = px.histogram(df, x=col, nbins=bins, title=f"Histogram of {col}", color=color_by)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available for histogram.")
                
        elif plot_type == "Box Plot":
            num_cols = ct.get('numeric', [])
            if num_cols:
                y_col = st.selectbox("Select numeric column (Y-axis)", num_cols, key="box_y")
                x_col_options = [None] + ct.get('categorical', []) + ct.get('binary', [])
                x_col = st.selectbox("Group by (X-axis, optional categorical)", x_col_options, key="box_x")
                
                if y_col:
                    fig = px.box(df, y=y_col, x=x_col, title=f"Box Plot of {y_col}" + (f" by {x_col}" if x_col else ""))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No numeric columns available for box plot.")
                
        elif plot_type == "Bar Plot (Categorical)":
            cat_cols = ct.get('categorical', []) + ct.get('binary', [])
            if cat_cols:
                col = st.selectbox("Select categorical/binary column", cat_cols, key="bar_col")
                if col:
                    limit = st.slider("Max categories to show", 5, 50, 15, key="bar_limit")
                    # Calculate value counts, handling potential NaNs
                    vc = df[col].value_counts(dropna=False).nlargest(limit).reset_index()
                    vc.columns = [col, 'Count']
                    vc[col] = vc[col].astype(str) # Ensure consistent type for plotting
                    
                    sort_by = st.radio("Sort by", ["Frequency", "Category Name"], key="bar_sort")
                    if sort_by == "Category Name":
                        vc = vc.sort_values(by=col)
                    
                    fig = px.bar(
                        vc, x=col, y='Count', title=f"Bar Plot of {col} (Top {limit} categories)",
                        color=col, text='Count'
                    )
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No categorical or binary columns available for bar plot.")
                
        elif plot_type == "Scatter Plot":
            num_cols = ct.get('numeric', [])
            if len(num_cols) >= 2:
                x_col = st.selectbox("Select X-axis (numeric)", num_cols, key="scatter_x")
                y_col_options = [c for c in num_cols if c != x_col]
                if not y_col_options:
                     st.warning("Need at least two distinct numeric columns.")
                     return
                y_col = st.selectbox("Select Y-axis (numeric)", y_col_options, key="scatter_y")
                
                color_by_options = [None] + ct.get('categorical', []) + ct.get('binary', []) + ct.get('numeric', []) # Allow coloring by numeric too
                color_by = st.selectbox("Color by (optional)", color_by_options, key="scatter_color")
                size_by_options = [None] + ct.get('numeric', [])
                size_by = st.selectbox("Size by (optional numeric)", size_by_options, key="scatter_size")
                
                if x_col and y_col:
                    fig = px.scatter(
                        df, x=x_col, y=y_col, color=color_by, size=size_by,
                        title=f"Scatter Plot: {x_col} vs {y_col}" + (f" (Color: {color_by})" if color_by else "") + (f" (Size: {size_by})" if size_by else ""),
                        hover_data=df.columns # Show all data on hover
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for scatter plot.")
                
        elif plot_type == "Correlation Heatmap":
            num_cols = ct.get('numeric', [])
            if len(num_cols) >= 2:
                # Use all numeric columns for correlation
                with st.spinner("Calculating correlation matrix..."):
                     corr = df[num_cols].corr()
                
                fig = px.imshow(
                    corr, text_auto='.2f', # Format text labels to 2 decimal places
                    title="Correlation Heatmap of Numeric Features",
                    color_continuous_scale="RdBu_r", # Red-Blue diverging scale
                    zmin=-1, zmax=1 # Ensure range is -1 to 1
                )
                fig.update_layout(height=600) # Adjust height if needed
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for correlation heatmap.")
                
        elif plot_type == "Pie Chart":
            cat_cols = ct.get('categorical', []) + ct.get('binary', [])
            if cat_cols:
                col = st.selectbox("Select categorical/binary column for pie chart", cat_cols, key="pie_col")
                if col:
                    limit = st.slider("Max slices (plus 'Other')", 2, 20, 8, key="pie_limit")
                    vc = df[col].value_counts(dropna=False)
                    
                    # Group small slices into 'Other'
                    if len(vc) > limit:
                        vc_top = vc.nlargest(limit - 1)
                        other_count = vc.nsmallest(len(vc) - (limit - 1)).sum()
                        vc_final = pd.concat([vc_top, pd.Series({'Other': other_count})])
                    else:
                        vc_final = vc
                        
                    vc_final.index = vc_final.index.astype(str) # Ensure index is string
                        
                    fig = px.pie(
                        values=vc_final.values, names=vc_final.index, 
                        title=f"Distribution of {col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No categorical or binary columns available for pie chart.")
                
        elif plot_type == "Line Plot (Time/Numeric)":
            # Prioritize datetime columns for X-axis
            if ct.get('datetime'):
                x_col = st.selectbox("Select Time column (X-axis)", ct['datetime'], key="line_time_x")
                y_col = st.selectbox("Select Value column (Y-axis, numeric)", ct.get('numeric', []), key="line_time_y")
                
                if x_col and y_col:
                    color_by_options = [None] + ct.get('categorical', []) + ct.get('binary', [])
                    color_by = st.selectbox("Group lines by (optional categorical)", color_by_options, key="line_time_color")
                    
                    # Sort by time before plotting
                    plot_df = df.sort_values(by=x_col)
                    
                    fig = px.line(
                        plot_df, x=x_col, y=y_col, color=color_by,
                        title=f"{y_col} over Time" + (f" by {color_by}" if color_by else "")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select both a time column and a numeric value column.")
            
            # Fallback to numeric vs numeric line plot if no datetime
            elif len(ct.get('numeric', [])) >= 2:
                st.info("No datetime columns found. Select two numeric columns for a line plot (X-axis will be sorted).")
                x_col = st.selectbox("Select X-axis (numeric, will be sorted)", ct['numeric'], key="line_num_x")
                y_col_options = [c for c in ct['numeric'] if c != x_col]
                if not y_col_options:
                     st.warning("Need at least two distinct numeric columns.")
                     return
                y_col = st.selectbox("Select Y-axis (numeric)", y_col_options, key="line_num_y")
                
                if x_col and y_col:
                     color_by_options = [None] + ct.get('categorical', []) + ct.get('binary', [])
                     color_by = st.selectbox("Group lines by (optional categorical)", color_by_options, key="line_num_color")
                     
                     # Sort by X-axis value
                     plot_df = df.sort_values(by=x_col)
                     fig = px.line(
                         plot_df, x=x_col, y=y_col, color=color_by,
                         title=f"{y_col} vs {x_col}" + (f" by {color_by}" if color_by else "")
                     )
                     st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need a datetime column or at least 2 numeric columns for line plot.")
                
    except Exception as e:
         logger.error(f"Error building custom plot '{plot_type}': {e}\n{traceback.format_exc()}")
         st.error(f"Could not generate plot '{plot_type}': {e}")


def eda_and_insights(df, ct, target_col, problem_type):
    """Performs Exploratory Data Analysis and generates automated insights."""
    st.header("🔎 Data Overview & Insights")
    
    # --- Basic Info ---
    st.subheader("Basic Information")
    col1, col2, col3 = st.columns(3)
    missing_total = df.isna().sum().sum()
    missing_pct = (missing_total / (df.shape[0] * df.shape[1])) * 100 if (df.shape[0] * df.shape[1]) > 0 else 0
    with col1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with col3:
        st.metric("Missing Values", f"{missing_total:,} ({missing_pct:.1f}%)")
    
    # --- Column Types ---
    st.subheader("Column Types Overview")
    col_types_summary = {k: len(v) for k, v in ct.items() if len(v) > 0} # Only show types present
    if col_types_summary:
        fig_types = px.bar(
            x=list(col_types_summary.keys()), 
            y=list(col_types_summary.values()),
            title="Detected Column Types",
            labels={"x": "Type", "y": "Count"},
            text=list(col_types_summary.values())
        )
        fig_types.update_traces(textposition='outside')
        st.plotly_chart(fig_types, use_container_width=True)
    
    with st.expander("View Column Details"):
        # Create a more informative details table
        details = []
        for col in df.columns:
            col_info = {"Column": col, "Type": "Unknown", "Missing": df[col].isna().sum(), "Unique": df[col].nunique()}
            for type_name, cols in ct.items():
                if col in cols:
                    col_info["Type"] = type_name.capitalize()
                    break
            col_info["Missing %"] = f"{col_info['Missing'] / df.shape[0] * 100:.1f}%" if df.shape[0] > 0 else "N/A"
            details.append(col_info)
        st.dataframe(pd.DataFrame(details))
    
    # --- Missing Values Analysis ---
    missing_per_col = df.isnull().sum()
    missing_per_col = missing_per_col[missing_per_col > 0] # Filter only columns with missing values
    if not missing_per_col.empty:
        st.subheader("Missing Values Analysis")
        missing_df = missing_per_col.sort_values(ascending=False).reset_index()
        missing_df.columns = ['Column', 'Missing Count']
        missing_df['Missing %'] = (missing_df['Missing Count'] / df.shape[0]) * 100
        
        fig_missing = px.bar(
            missing_df, 
            x='Column', 
            y='Missing Count',
            title="Missing Values per Column",
            hover_data=['Missing %'],
            text='Missing Count'
        )
        fig_missing.update_traces(textposition='outside')
        st.plotly_chart(fig_missing, use_container_width=True)
        
        high_missing_cols = missing_df[missing_df['Missing %'] > 50]
        if not high_missing_cols.empty:
             st.warning(f"⚠️ Columns with >50% missing values: {', '.join(high_missing_cols['Column'].tolist())}. Consider dropping or careful imputation.")
    else:
         st.success("✅ No missing values detected in the dataset.")

    # --- Target Variable Analysis ---
    if target_col and target_col in df.columns:
        st.subheader(f"🎯 Target Variable Analysis: '{target_col}' (Type: {problem_type})")
        
        if problem_type == "Classification":
            # Class distribution
            st.write("**Class Distribution**")
            target_counts = df[target_col].value_counts(dropna=False).reset_index()
            target_counts.columns = [target_col, 'Count']
            target_counts[target_col] = target_counts[target_col].astype(str) # Ensure labels are strings for plotting
            
            fig_target_pie = px.pie(
                target_counts, values='Count', names=target_col,
                title=f"Distribution of Target: {target_col}"
            )
            fig_target_bar = px.bar(
                target_counts, x=target_col, y='Count',
                title=f"Counts for Target: {target_col}", text='Count'
            )
            fig_target_bar.update_traces(textposition='outside')
            
            col1_target, col2_target = st.columns(2)
            with col1_target:
                 st.plotly_chart(fig_target_pie, use_container_width=True)
            with col2_target:
                 st.plotly_chart(fig_target_bar, use_container_width=True)
            
            # Class imbalance check
            if df[target_col].nunique() > 1: # Only relevant if more than one class
                 max_class_pct = df[target_col].value_counts(normalize=True).max() * 100
                 if max_class_pct > 70:
                     st.warning(f"⚠️ **Potential Class Imbalance:** The most frequent class accounts for {max_class_pct:.1f}% of the data. This might affect model performance. Consider using techniques like SMOTE or class weighting if needed.")
                
        elif problem_type == "Regression":
            # Distribution of target
            st.write("**Distribution and Properties**")
            fig_target_hist = px.histogram(
                df, x=target_col, nbins=50, title=f"Distribution of Target: {target_col}"
            )
            fig_target_box = px.box(
                df, y=target_col, title=f"Box Plot of Target: {target_col}"
            )
            
            col1_target, col2_target = st.columns(2)
            with col1_target:
                 st.plotly_chart(fig_target_hist, use_container_width=True)
            with col2_target:
                 st.plotly_chart(fig_target_box, use_container_width=True)

            # Skewness and Outlier check
            skewness = df[target_col].skew()
            kurt = df[target_col].kurt()
            st.write(f"**Skewness:** {skewness:.2f}")
            st.write(f"**Kurtosis:** {kurt:.2f}")
            if abs(skewness) > 1:
                st.warning(f"⚠️ **High Skewness:** The target variable is significantly skewed. Consider transformations (e.g., log, Box-Cox) if models assume normality (like Linear Regression).")
            if kurt > 3: # Kurtosis > 3 indicates heavy tails (potential outliers)
                 st.info("ℹ️ **High Kurtosis:** Suggests heavier tails than a normal distribution, possibly indicating outliers.")
                 
    else:
         st.warning("Target column not identified or not present in DataFrame.")

    # --- Automated Insights Generation ---
    st.subheader("💡 Automated Insights & Checks")
    insights = []
    
    # High Cardinality Categoricals (excluding target)
    high_card_cats = []
    for col in ct.get('categorical', []):
         if col != target_col and df[col].nunique() > 50: # Threshold for high cardinality
              high_card_cats.append(f"'{col}' ({df[col].nunique()} unique values)")
    if high_card_cats:
         insights.append(f"⚠️ **High Cardinality Categoricals:** Features {', '.join(high_card_cats)} have many unique values. Consider grouping, target encoding, or embedding techniques.")

    # High Correlation Between Numeric Features (excluding target)
    numeric_cols_no_target = [c for c in ct.get('numeric', []) if c != target_col]
    if len(numeric_cols_no_target) > 1:
        try:
            corr_matrix = df[numeric_cols_no_target].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = [(col, row, upper_tri.loc[row, col]) for row in upper_tri.index for col in upper_tri.columns if upper_tri.loc[row, col] > 0.9] # Threshold 0.9
            
            if high_corr_pairs:
                 pair_strings = [f"('{p[0]}' & '{p[1]}' corr: {p[2]:.2f})" for p in high_corr_pairs]
                 insights.append(f"🔄 **Potential Multicollinearity:** Found pairs of numeric features with high correlation (>0.9): {', '.join(pair_strings)}. Consider feature selection or dimensionality reduction (e.g., PCA).")
        except Exception as e_corr:
            logger.warning(f"Could not calculate feature correlations: {e_corr}")

    # Skewed Numeric Features (excluding target)
    highly_skewed_features = []
    if numeric_cols_no_target:
        try:
            skewness = df[numeric_cols_no_target].skew()
            highly_skewed_features = [f"'{idx}' (skew: {val:.2f})" for idx, val in skewness.items() if abs(val) > 1.5] # Threshold 1.5
        except Exception as e_skew:
             logger.warning(f"Could not calculate skewness for features: {e_skew}")
             
    if highly_skewed_features:
         insights.append(f"📈 **Skewed Numeric Features:** Features {', '.join(highly_skewed_features)} are highly skewed. Consider applying transformations (log, sqrt, Box-Cox) for models sensitive to feature distribution.")

    # Display Insights
    if insights:
        for insight in insights:
            st.info(insight)
    else:
        st.success("✅ Initial automated checks did not reveal major data issues like high correlation or skewness among features.")


# --- Utility Functions ---

def get_table_download_link(df, filename="data.csv", link_text="Download CSV"):
    """Generates a link allowing the data in a given panda dataframe to be downloaded"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# --- Main App Logic ---

st.title("🚀 AutoML SaaS Platform")
st.markdown("Upload your data, explore insights, preprocess, train models, and evaluate performance - all in one place.")

# Sidebar for Navigation
with st.sidebar:
    st.image("data:image/jpeg;base64,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", use_column_width=True)
    st.markdown("---")
    st.header("Workflow Steps")
    
    # Use radio buttons for navigation, updating session state
    app_mode = st.radio(
        "Select Step:",
        ["Upload Data", "Explore Data", "Preprocess Data", "Train Models", "Evaluate Models"],
        key='app_mode_radio', # Use a distinct key for the widget
        index=["Upload Data", "Explore Data", "Preprocess Data", "Train Models", "Evaluate Models"].index(st.session_state.app_mode) # Set index based on current state
    )
    # Update session state if radio button changes
    if app_mode != st.session_state.app_mode:
        st.session_state.app_mode = app_mode
        st.rerun() # Rerun script to reflect the new mode immediately

    st.markdown("---")
    # Display current state in sidebar
    if st.session_state.data_uploaded:
        st.success("✅ Data Loaded")
        st.write(f"Shape: `{st.session_state.df.shape}`")
        st.write(f"Target: `{st.session_state.target_col}`")
        st.write(f"Problem: `{st.session_state.problem_type}`")
    else:
        st.info("ℹ️ Upload data to begin.")
    
    if st.session_state.df_proc is not None:
         st.success("✅ Data Preprocessed")
         st.write(f"Shape: `{st.session_state.df_proc.shape}`")
         
    if st.session_state.model_results is not None:
         st.success("✅ Models Trained")


# Main content area based on selected app_mode
if st.session_state.app_mode == "Upload Data":
    st.header("📤 Step 1: Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV, Excel, or JSON file", 
        type=["csv", "xlsx", "xls", "json"],
        help="Ensure your file is properly formatted. UTF-8 encoding is recommended for CSV."
    )
    
    # --- Sample Data Option ---
    st.markdown("---")
    st.markdown("##### Or try with sample data:")
    sample_option = st.selectbox(
        "Select sample dataset", 
        ["None", "Iris (Classification)", "Diabetes (Regression)", "Titanic (Classification)"],
        key="sample_data_select"
    )

    # --- Process Upload or Sample Data ---
    df_to_process = None
    data_source_name = ""

    if uploaded_file is not None:
        data_source_name = uploaded_file.name
        with st.spinner(f"Loading and analyzing '{data_source_name}'..."):
            df, err = load_data(uploaded_file)
            if err:
                st.error(f"Error loading file: {err}")
            else:
                df_to_process = df
                st.success(f"✅ File '{data_source_name}' loaded successfully!")
    
    elif sample_option != "None":
        data_source_name = f"Sample: {sample_option}"
        with st.spinner(f"Loading {data_source_name}..."):
            if sample_option == "Iris (Classification)":
                from sklearn.datasets import load_iris
                data = load_iris()
                df_sample = pd.DataFrame(data.data, columns=data.feature_names)
                df_sample['target'] = data.target_names[data.target] # Use names for clarity
            elif sample_option == "Diabetes (Regression)":
                from sklearn.datasets import load_diabetes
                data = load_diabetes()
                df_sample = pd.DataFrame(data.data, columns=data.feature_names)
                df_sample['target'] = data.target
            elif sample_option == "Titanic (Classification)":
                 try:
                    # Load a slightly cleaned version for better demo
                    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                    df_sample = pd.read_csv(url)
                    # Basic cleaning for demo
                    df_sample['Age'].fillna(df_sample['Age'].median(), inplace=True)
                    df_sample['Embarked'].fillna(df_sample['Embarked'].mode()[0], inplace=True)
                    df_sample.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True) # Drop problematic/high cardinality cols
                    df_sample.rename(columns={'Survived': 'target'}, inplace=True) # Rename target
                    df_sample['target'] = df_sample['target'].map({0: 'Not Survived', 1: 'Survived'}) # Use names
                 except Exception as e:
                      st.error(f"Failed to load Titanic sample data: {e}")
                      df_sample = None
            
            if df_sample is not None:
                df_to_process = df_sample
                st.success(f"✅ Sample data '{sample_option}' loaded!")

    # --- Post-Load Processing (Target, Type Detection) ---
    if df_to_process is not None:
        st.session_state.df = df_to_process
        st.session_state.data_uploaded = True
        
        # Auto-detect target and problem type
        with st_exception_handler("Target/Problem Type Detection"):
            st.session_state.target_col = auto_detect_target(st.session_state.df)
            if st.session_state.target_col:
                st.session_state.problem_type = detect_problem_type(st.session_state.df[st.session_state.target_col])
            else:
                 st.error("Could not identify a target column.")
                 st.stop() # Stop if target is essential
                 
            # Identify column types
            st.session_state.ct = identify_column_types(st.session_state.df)
        
        st.subheader("Initial Setup")
        st.info(f"Dataset: **{data_source_name}** | Shape: **{st.session_state.df.shape}**")
        
        # Allow user override of target and problem type
        col1, col2 = st.columns(2)
        with col1:
             target_options = st.session_state.df.columns.tolist()
             selected_target_idx = target_options.index(st.session_state.target_col) if st.session_state.target_col in target_options else 0
             new_target = st.selectbox(
                 "🎯 Select Target Column:", 
                 target_options, 
                 index=selected_target_idx,
                 key="target_select_override"
             )
             if new_target != st.session_state.target_col:
                 st.session_state.target_col = new_target
                 # Redetect problem type if target changes
                 st.session_state.problem_type = detect_problem_type(st.session_state.df[st.session_state.target_col])
                 st.rerun() # Rerun to update UI based on new target/type

        with col2:
            problem_options = ["Classification", "Regression"]
            selected_problem_idx = problem_options.index(st.session_state.problem_type) if st.session_state.problem_type in problem_options else 0
            new_problem_type = st.selectbox(
                "⚙️ Select Problem Type:",
                problem_options,
                index=selected_problem_idx,
                key="problem_type_override"
            )
            if new_problem_type != st.session_state.problem_type:
                 st.session_state.problem_type = new_problem_type
                 st.rerun() # Rerun to update UI

        st.success(f"Ready! Target: **{st.session_state.target_col}** | Problem Type: **{st.session_state.problem_type}**")

        # Show data sample
        st.subheader("Data Preview (First 10 Rows)")
        st.dataframe(st.session_state.df.head(10))
        st.markdown(get_table_download_link(st.session_state.df.head(100), filename="data_preview.csv", link_text="Download Preview CSV"), unsafe_allow_html=True)
        
        # Navigation button
        if st.button("Continue to Data Exploration →", key="upload_next"):
             st.session_state.app_mode = "Explore Data"
             st.rerun()

elif st.session_state.app_mode == "Explore Data":
    if not st.session_state.data_uploaded or st.session_state.df is None:
        st.warning("⚠️ Please upload data first on the 'Upload Data' step.")
        if st.button("Go to Upload Data"):
            st.session_state.app_mode = "Upload Data"
            st.rerun()
    else:
        st.header("📊 Step 2: Explore Data")
        df = st.session_state.df
        ct = st.session_state.ct
        target_col = st.session_state.target_col
        problem_type = st.session_state.problem_type
        
        # Perform EDA and show insights
        with st_exception_handler("Exploratory Data Analysis"):
            eda_and_insights(df, ct, target_col, problem_type)
        
        # Custom plot builder section
        with st.expander("🎨 Create Custom Visualizations", expanded=False):
             with st_exception_handler("Custom Plot Generation"):
                custom_plot_builder(df, ct)
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Upload Data", key="explore_back"):
                 st.session_state.app_mode = "Upload Data"
                 st.rerun()
        with col2:
            if st.button("Continue to Preprocessing →", key="explore_next"):
                 st.session_state.app_mode = "Preprocess Data"
                 st.rerun()

elif st.session_state.app_mode == "Preprocess Data":
    if not st.session_state.data_uploaded or st.session_state.df is None:
        st.warning("⚠️ Please upload data first on the 'Upload Data' step.")
        if st.button("Go to Upload Data"):
            st.session_state.app_mode = "Upload Data"
            st.rerun()
    else:
        st.header("⚙️ Step 3: Preprocess Data")
        df = st.session_state.df
        ct = st.session_state.ct # Use the identified column types
        target_col = st.session_state.target_col
        problem_type = st.session_state.problem_type
        
        st.subheader("Preprocessing Configuration")
        
        col1_prep, col2_prep = st.columns(2)
        with col1_prep:
            st.markdown("**Core Operations:**")
            scale_opt = st.checkbox("Scale Numeric Features (StandardScaler)", value=True, key="prep_scale")
            encode_opt = st.checkbox("Encode Categorical/Binary Features (LabelEncoder)", value=True, key="prep_encode", help="Uses simple Label Encoding. OneHotEncoding might be better for non-tree models but increases features.")
            impute_opt = st.selectbox(
                "Missing Value Imputation",
                ["Simple (Median/Mode)", "KNN Imputation (Numeric only)"],
                key="prep_impute", help="KNN can be more accurate but slower."
            )
            impute_method = 'knn' if impute_opt == "KNN Imputation (Numeric only)" else 'simple'
            
        with col2_prep:
            st.markdown("**Feature Handling:**")
            datetime_opt = st.selectbox(
                "Datetime Handling", 
                ["Extract Features (Year, Month, Day etc.)", "Drop Datetime Columns"], 
                key="prep_datetime", help="Extract creates new numeric features."
            )
            datetime_handling = 'extract' if datetime_opt == "Extract Features (Year, Month, Day etc.)" else 'drop'
            drop_id_opt = st.checkbox("Drop Detected ID Columns", value=True, key="prep_drop_id")
            drop_text_opt = st.checkbox("Drop Detected Text Columns", value=True, key="prep_drop_text")

        # --- Advanced Options Expander ---
        with st.expander("🔬 Advanced Preprocessing Options"):
            st.markdown("**Feature Selection:**")
            use_feature_selection = st.checkbox("Apply Feature Selection (SelectKBest)", value=False, key="prep_fs")
            k_features = 10 # Default k
            if use_feature_selection:
                 max_k = len(df.columns) - 1 - len(ct.get('id',[])) - len(ct.get('text',[])) # Estimate max possible features
                 k_features = st.slider(
                     "Number of features to select (K)", 
                     min_value=1, 
                     max_value=max(1, max_k), # Ensure max_value is at least 1
                     value=min(10, max(1, max_k)), # Sensible default
                     key="prep_fs_k"
                 )

            st.markdown("**Class Balancing (Classification Only):**")
            handle_imbalance = False
            balance_method = None
            if problem_type == "Classification" and IMBLEARN_AVAILABLE:
                handle_imbalance = st.checkbox("Apply Class Balancing (Requires imblearn)", value=False, key="prep_balance", help="Helps if classes are unevenly distributed.")
                if handle_imbalance:
                    balance_method_options = ["SMOTE (Oversampling)", "Random Oversampling", "Random Undersampling"]
                    balance_method = st.radio(
                        "Balancing Method", balance_method_options, key="prep_balance_method"
                    )
                    # Map radio button selection to internal names if needed
                    if balance_method == "SMOTE (Oversampling)": balance_method = "smote"
                    elif balance_method == "Random Oversampling": balance_method = "over"
                    elif balance_method == "Random Undersampling": balance_method = "under"
            elif problem_type == "Classification" and not IMBLEARN_AVAILABLE:
                 st.warning("Class balancing requires `imbalanced-learn`. Install it (`pip install imbalanced-learn`) and restart.")


        # --- Apply Preprocessing ---
        if st.button("🚀 Apply Preprocessing Steps", key="prep_apply"):
            # Make a fresh copy of column types before preprocessing modifies them
            ct_for_processing = {k: list(v) for k, v in ct.items()} 
            
            with st.spinner("⚙️ Applying preprocessing steps... Please wait."):
                # --- Run Preprocessing ---
                df_proc, label_encoders, error_prep = preprocess_data(
                    df, target_col, ct_for_processing, # Pass the copy of ct
                    scale_numeric=scale_opt, 
                    encode_categorical=encode_opt, 
                    impute_method=impute_method, 
                    datetime_handling=datetime_handling,
                    drop_id=drop_id_opt,
                    drop_text=drop_text_opt
                )
                
                if error_prep:
                     st.error(f"Preprocessing failed: {error_prep}")
                     st.stop()

                # --- Apply Advanced Steps (if selected) ---
                X = df_proc.drop(columns=[target_col], errors='ignore')
                y = df_proc[target_col]

                # Feature Selection (SelectKBest)
                if use_feature_selection and k_features:
                    with st_exception_handler("Feature Selection"):
                        logger.info(f"Applying SelectKBest with k={k_features}")
                        fs_func = f_classif if problem_type == "Classification" else f_regression
                        selector = SelectKBest(score_func=fs_func, k=min(k_features, X.shape[1])) # Ensure k is not > num features
                        
                        # Temporarily handle potential NaNs in target for selector fitting
                        y_filled = y.fillna(y.median() if pd.api.types.is_numeric_dtype(y) else y.mode()[0])
                        
                        X_new = selector.fit_transform(X, y_filled)
                        selected_cols = X.columns[selector.get_support()]
                        
                        # Reconstruct DataFrame with selected features + target
                        df_proc = pd.DataFrame(X_new, columns=selected_cols, index=df_proc.index)
                        df_proc[target_col] = y
                        X = df_proc.drop(columns=[target_col]) # Update X
                        
                        st.info(f"Applied SelectKBest. Selected {len(selected_cols)} features: `{', '.join(selected_cols)}`")

                # Class Balancing
                if problem_type == "Classification" and handle_imbalance and balance_method:
                    with st_exception_handler("Class Balancing"):
                         logger.info(f"Applying class balancing using: {balance_method}")
                         sampler = None
                         if balance_method == "smote":
                              sampler = SMOTE(random_state=42)
                         elif balance_method == "over":
                              sampler = RandomOverSampler(random_state=42)
                         elif balance_method == "under":
                              sampler = RandomUnderSampler(random_state=42)
                         
                         if sampler:
                             X_res, y_res = sampler.fit_resample(X, y)
                             # Reconstruct DataFrame (Important: index gets reset here)
                             df_proc = pd.DataFrame(X_res, columns=X.columns)
                             df_proc[target_col] = y_res
                             X, y = X_res, y_res # Update X and y for subsequent steps
                             
                             st.info(f"Applied '{balance_method}'. New shape: {df_proc.shape}. New class distribution: `{ {k: v for k, v in y_res.value_counts().items()} }`")

                # --- Store Results ---
                st.session_state.df_proc = df_proc
                # Optional: store label encoders if needed later
                # st.session_state.label_encoders = label_encoders 

            st.success("✅ Preprocessing pipeline completed successfully!")
            
            # Show processed data preview
            st.subheader("Processed Data Preview (First 100 Rows)")
            st.dataframe(st.session_state.df_proc.head(100))
            st.markdown(get_table_download_link(st.session_state.df_proc.head(100), filename="processed_data_preview.csv", link_text="Download Processed Preview CSV"), unsafe_allow_html=True)
            
            # Show shape comparison
            st.metric("Shape After Preprocessing", f"{st.session_state.df_proc.shape[0]} rows, {st.session_state.df_proc.shape[1]} columns")


        # Navigation buttons
        col1_nav, col2_nav = st.columns(2)
        with col1_nav:
            if st.button("← Back to Exploration", key="prep_back"):
                 st.session_state.app_mode = "Explore Data"
                 st.rerun()
        with col2_nav:
            # Enable next step only if preprocessing has been run successfully
            if st.session_state.df_proc is not None:
                if st.button("Continue to Model Training →", key="prep_next"):
                    st.session_state.app_mode = "Train Models"
                    st.rerun()
            else:
                st.info("ℹ️ Apply preprocessing steps to enable model training.")


elif st.session_state.app_mode == "Train Models":
    if st.session_state.df_proc is None:
        st.warning("⚠️ Please preprocess your data first on the 'Preprocess Data' step.")
        if st.button("Go to Preprocessing"):
            st.session_state.app_mode = "Preprocess Data"
            st.rerun()
    else:
        st.header("🤖 Step 4: Train Machine Learning Models")
        df_proc = st.session_state.df_proc
        target_col = st.session_state.target_col
        problem_type = st.session_state.problem_type
        
        # --- Model Selection ---
        st.subheader("Model Selection")
        available_models = CLASSIFICATION_MODELS if problem_type == "Classification" else REGRESSION_MODELS
        model_options = list(available_models.keys())
        
        # Sensible defaults: Choose a few diverse models
        default_selection = []
        if problem_type == "Classification":
             default_selection = ["Logistic Regression", "Random Forest", "Decision Tree"]
             default_selection = [m for m in default_selection if m in model_options] # Ensure defaults exist
        else:
             default_selection = ["Linear Regression", "Random Forest", "Decision Tree"]
             default_selection = [m for m in default_selection if m in model_options]

        # Allow selecting all models easily
        select_all = st.checkbox("Select All Models", value=False, key="train_select_all")
        if select_all:
             selected_models = st.multiselect(
                 f"Select {problem_type} models to train:", 
                 model_options, 
                 default=model_options, # Select all if checkbox is true
                 key="train_model_select"
             )
        else:
              selected_models = st.multiselect(
                 f"Select {problem_type} models to train:", 
                 model_options, 
                 default=default_selection, # Use defined defaults
                 key="train_model_select"
             )
             
        models_to_run = {name: available_models[name] for name in selected_models if name in available_models}
        
        # --- Training Options ---
        st.subheader("Training Configuration")
        col1_train, col2_train = st.columns(2)
        with col1_train:
            test_size_opt = st.slider("Test Set Size (%)", 10, 50, 25, key="train_test_split", help="Percentage of data held out for final model evaluation.") / 100
            random_state_opt = st.number_input("Random Seed", value=42, key="train_random_state", help="Ensures reproducibility of train/test split and model training.")
        with col2_train:
             timeout_opt = st.number_input("Max Training Time per Model (seconds)", min_value=10, max_value=600, value=120, key="train_timeout", help="Prevents models from running indefinitely. May not work on Windows.")
             # Cross-validation option (can add later)
             # use_cv = st.checkbox("Use Cross-Validation during training", value=False, key="train_use_cv")
             # cv_folds = 5
             # if use_cv:
             #     cv_folds = st.slider("Number of CV Folds", 2, 10, 5, key="train_cv_folds")

        # --- Train Button and Execution ---
        if st.button("🚀 Train Selected Models", key="train_start"):
            if not models_to_run:
                st.warning("⚠️ Please select at least one model to train.")
            else:
                with st_exception_handler("Model Training Pipeline"):
                    # Prepare data
                    X = df_proc.drop(columns=[target_col], errors='ignore')
                    y = df_proc[target_col]
                    
                    if X.empty or y.empty:
                         st.error("Feature set (X) or target (y) is empty after preprocessing. Check preprocessing steps.")
                         st.stop()
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size_opt, random_state=random_state_opt, 
                        stratify=y if problem_type == "Classification" and y.nunique() > 1 else None # Stratify for classification
                    )
                    logger.info(f"Data split: Train shape={X_train.shape}, Test shape={X_test.shape}")
                    
                    # Store test set for evaluation step
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    # --- Run Training & Evaluation ---
                    results_df, predictions_dict, trained_models_dict = train_and_evaluate_models(
                        X_train, X_test, y_train, y_test, models_to_run, problem_type, timeout_per_model=timeout_opt
                    )
                    
                    # --- Store Results in Session State ---
                    st.session_state.model_results = results_df
                    st.session_state.predictions = predictions_dict
                    st.session_state.trained_models = trained_models_dict # Store trained model objects
                    
                    # Display summary table immediately
                    st.subheader("📊 Model Performance Summary")
                    st.dataframe(results_df.style.format(precision=4)) # Format for better readability
                    
                    # Provide download link for results
                    st.markdown(get_table_download_link(results_df, filename="model_performance_summary.csv", link_text="Download Performance Summary CSV"), unsafe_allow_html=True)

                    st.success("✅ Model training finished. Proceed to 'Evaluate Models' for detailed analysis.")

        # Navigation buttons
        col1_nav, col2_nav = st.columns(2)
        with col1_nav:
            if st.button("← Back to Preprocessing", key="train_back"):
                 st.session_state.app_mode = "Preprocess Data"
                 st.rerun()
        with col2_nav:
            # Enable next step only if models have been trained
            if st.session_state.model_results is not None:
                if st.button("Continue to Evaluation →", key="train_next"):
                    st.session_state.app_mode = "Evaluate Models"
                    st.rerun()
            else:
                 st.info("ℹ️ Train models to enable evaluation.")


elif st.session_state.app_mode == "Evaluate Models":
    if st.session_state.model_results is None or st.session_state.X_test is None:
        st.warning("⚠️ Please train models first on the 'Train Models' step.")
        if st.button("Go to Model Training"):
            st.session_state.app_mode = "Train Models"
            st.rerun()
    else:
        st.header("📈 Step 5: Evaluate Model Performance")
        
        results_df = st.session_state.model_results
        predictions = st.session_state.predictions
        trained_models = st.session_state.trained_models
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        problem_type = st.session_state.problem_type
        
        # Display performance summary table again
        st.subheader("Performance Summary")
        st.dataframe(results_df.style.format(precision=4))
        st.markdown(get_table_download_link(results_df, filename="model_performance_summary.csv", link_text="Download Performance Summary CSV"), unsafe_allow_html=True)
        
        # Plot overall model performance comparison
        with st_exception_handler("Plotting Model Performance"):
             plot_model_performance(results_df, problem_type)
        
        st.markdown("---")
        
        # --- Detailed Model Analysis ---
        st.subheader("🔍 Detailed Model Analysis")
        
        # Get list of successfully trained models
        successful_models = results_df[results_df['Status'] == 'Success']['Model'].tolist()
        
        if not successful_models:
             st.warning("No models were trained successfully. Cannot show detailed analysis.")
        else:
             selected_model_name = st.selectbox(
                 "Select a model for detailed evaluation:",
                 successful_models,
                 key="eval_select_model"
             )
             
             if selected_model_name:
                  # Retrieve the specific model's data
                  if selected_model_name in trained_models and selected_model_name in predictions:
                       selected_trained_model = trained_models[selected_model_name]
                       selected_predictions = predictions[selected_model_name]
                       
                       # Show detailed plots and metrics
                       with st_exception_handler(f"Detailed Evaluation for {selected_model_name}"):
                           show_model_details(
                               selected_model_name, 
                               selected_trained_model, 
                               X_test, 
                               y_test, 
                               selected_predictions, 
                               problem_type
                           )
                           
                       # --- Download Predictions for Selected Model ---
                       st.markdown("---")
                       st.subheader(f"Download Predictions for {selected_model_name}")
                       try:
                            preds_df = X_test.copy()
                            preds_df['actual_target'] = y_test
                            preds_df[f'predicted_{selected_model_name}'] = selected_predictions
                            st.markdown(get_table_download_link(preds_df, filename=f"predictions_{selected_model_name}.csv", link_text=f"Download Test Set with Predictions ({selected_model_name})"), unsafe_allow_html=True)
                       except Exception as e_dl:
                            logger.error(f"Failed to create prediction download link: {e_dl}")
                            st.warning("Could not generate download link for predictions.")
                            
                  else:
                       st.error(f"Could not find trained model or predictions for '{selected_model_name}'.")

        # Navigation button
        if st.button("← Back to Model Training", key="eval_back"):
            st.session_state.app_mode = "Train Models"
            st.rerun()

# --- Footer ---
st.markdown("---")
st.caption("AutoML SaaS Platform v1.0")

