import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, StackingClassifier, StackingRegressor
from xgboost import XGBClassifier, XGBRegressor

# Caching data loading functions for efficiency
@st.cache_data
def load_sample_data(name: str) -> pd.DataFrame:
    """Load a sample dataset by name from Seaborn's repository."""
    return sns.load_dataset(name)

@st.cache_data
def load_csv_data(file) -> pd.DataFrame:
    """Load a CSV file from an uploaded file object into a DataFrame."""
    return pd.read_csv(file)

def get_column_types(df: pd.DataFrame):
    """
    Separate column names into numeric and categorical lists.
    Numeric includes int/float types, categorical includes object, category, bool.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    return num_cols, cat_cols

# Initialize session state for persistent variables
if "trained" not in st.session_state:
    st.session_state.trained = False
    st.session_state.df = None
    st.session_state.last_data_source = None
    st.session_state.last_data_name = None

st.title("Interactive ML Model Trainer")
st.markdown("Configure the dataset and model in the sidebar, then click **Train** to train the model and view results.")

# Sidebar: Dataset selection
st.sidebar.header("1. Dataset Selection")
data_source = st.sidebar.radio("Choose data source:", ["Sample dataset", "Upload CSV"])
df = None  # DataFrame to hold the loaded data
data_name = None

if data_source == "Sample dataset":
    # Let user select one of Seaborn's built-in sample datasets
    sample_list = sns.get_dataset_names()  # list of available dataset names
    dataset_name = st.sidebar.selectbox("Select a sample dataset", sample_list)
    if dataset_name:
        df = load_sample_data(dataset_name)
        data_name = dataset_name
elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = load_csv_data(uploaded_file)
        data_name = uploaded_file.name

# If a new dataset is loaded, reset the trained flag and related state
if data_name is not None:
    if (st.session_state.last_data_source != data_source) or (st.session_state.last_data_name != data_name):
        st.session_state.trained = False  # new data, so require re-training
        st.session_state.df = None  # clear old data
    st.session_state.last_data_source = data_source
    st.session_state.last_data_name = data_name

# Store or update the dataset in session state
if df is not None:
    st.session_state.df = df

# Main panel: show a preview of the dataset (if loaded)
if st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Dataset Preview")
    st.write(f"**Dataset:** {data_name} —  {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10))
else:
    st.info("👈 Please select a dataset to get started.")
    st.stop()  # halt app until a dataset is provided

# Determine numeric and categorical columns in the dataset
num_cols, cat_cols = get_column_types(df)

# Sidebar: Model configuration form (uses st.form to batch inputs)
st.sidebar.header("2. Model Configuration")
with st.sidebar.form(key="config_form"):
    task_type = st.selectbox("Task type:", ["Classification", "Regression"])
    is_classification = (task_type == "Classification")
    target_col = st.selectbox("Target variable:", options=df.columns)
    # Feature selection: separate multiselect for numeric and categorical features (exclude target)
    default_num = [col for col in num_cols if col != target_col]
    default_cat = [col for col in cat_cols if col != target_col]
    selected_num = st.multiselect("Numeric features:", options=num_cols, default=default_num)
    selected_cat = st.multiselect("Categorical features:", options=cat_cols, default=default_cat)
    model_choice = st.selectbox("Model:", ["Random Forest", "XGBoost", "Stacking (RF + XGB)"])
    n_estimators = st.number_input("Number of trees (n_estimators)", min_value=10, max_value=500, value=100)
    max_depth = st.number_input("Max depth of trees (0 for none)", min_value=0, max_value=20, value=0)
    learning_rate = None
    if model_choice in ["XGBoost", "Stacking (RF + XGB)"]:
        learning_rate = st.number_input("Learning rate (for XGBoost)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    test_size_percent = st.slider("Test set size (%)", min_value=5, max_value=50, value=20)
    submit_train = st.form_submit_button(label="Train")

# When the Train button is clicked, execute training and evaluation
if submit_train:
    features = selected_num + selected_cat
    if target_col in features:
        features.remove(target_col)
    if len(features) == 0:
        st.error("Please select at least one feature for training.")
        st.stop()
    data = df[features + [target_col]].dropna()
    if len(data) < len(df):
        st.warning(f"Dropped {len(df) - len(data)} rows due to missing values.")
    X_all = data[features]
    y_all = data[target_col]
    # One-hot encode categorical features
    X_all_encoded = pd.get_dummies(X_all, columns=selected_cat)
    # Encode target for classification
    if is_classification:
        le = LabelEncoder()
        y_all_encoded = le.fit_transform(y_all.astype(str))
        class_names = le.classes_
    else:
        y_all_encoded = y_all.values
        class_names = None
    test_size = test_size_percent / 100.0
    stratify_param = y_all_encoded if is_classification and len(np.unique(y_all_encoded)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
    X_all_encoded, y_all_encoded, test_size=test_size, stratify=stratify_param, random_state=42)
    # Define model based on selection
    model = None
    if model_choice == "Random Forest":
        if is_classification:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth), random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth), random_state=42)
    elif model_choice == "XGBoost":
        if is_classification:
            model = XGBClassifier(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth),
                                  learning_rate=(learning_rate or 0.1), use_label_encoder=False, eval_metric='logloss', random_state=42)
        else:
            model = XGBRegressor(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth),
                                 learning_rate=(learning_rate or 0.1), random_state=42)
    elif model_choice == "Stacking (RF + XGB)":
        if is_classification:
            base_estimators = [
                ("rf", RandomForestClassifier(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth), random_state=42)),
                ("xgb", XGBClassifier(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth),
                                      learning_rate=(learning_rate or 0.1), use_label_encoder=False, eval_metric='logloss', random_state=42))
            ]
            model = StackingClassifier(estimators=base_estimators, final_estimator=None, passthrough=False)
        else:
            base_estimators = [
                ("rf", RandomForestRegressor(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth), random_state=42)),
                ("xgb", XGBRegressor(n_estimators=n_estimators, max_depth=(None if max_depth == 0 else max_depth),
                                     learning_rate=(learning_rate or 0.1), random_state=42))
            ]
            model = StackingRegressor(estimators=base_estimators, final_estimator=None, passthrough=False)
    model.fit(X_train, y_train)
    st.session_state.trained = True
    st.session_state.is_classification = is_classification

    # Evaluate model performance
    if is_classification:
        y_pred = model.predict(X_test)
        if class_names is not None:
            y_test_labels = [class_names[i] for i in y_test]
            y_pred_labels = [class_names[i] for i in y_pred]
        else:
            y_test_labels = y_test
            y_pred_labels = y_pred
        accuracy_val = accuracy_score(y_test, y_pred)
        st.session_state.accuracy = accuracy_val
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm,
                    xticklabels=class_names if class_names is not None else "auto",
                    yticklabels=class_names if class_names is not None else "auto")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title("Confusion Matrix")
        st.session_state.confusion_fig = fig_cm
        roc_fig = None
        roc_auc_val = None
        if len(np.unique(y_test)) == 2:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
            roc_auc_val = auc(fpr, tpr)
            roc_fig, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.2f}")
            ax_roc.plot([0, 1], [0, 1], "--", color="gray")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend(loc="lower right")
            st.session_state.roc_fig = roc_fig
            st.session_state.roc_auc = roc_auc_val
        else:
            st.session_state.roc_fig = None
            st.session_state.roc_auc = None
    else:
        y_pred = model.predict(X_test)
        mse_val = mean_squared_error(y_test, y_pred)
        rmse_val = np.sqrt(mse_val)   
        r2_val = r2_score(y_test, y_pred)
        st.session_state.rmse = rmse_val
        st.session_state.r2 = r2_val
        residuals = y_test - y_pred
        fig_res, ax_res = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax_res, color='teal')
        ax_res.axvline(0, color="red", linestyle="--")
        ax_res.set_xlabel("Residual (Actual – Predicted)")
        ax_res.set_title("Residuals Distribution")
        st.session_state.residual_fig = fig_res

    # Feature importance (if applicable)
    st.session_state.feature_figs = []  # list of (label, figure)
    if model_choice == "Random Forest":
        importances = model.feature_importances_
        feat_names = X_train.columns
        imp_order = np.argsort(importances)[::-1]
        fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
        ax_imp.barh(feat_names[imp_order], importances[imp_order], color="steelblue")
        ax_imp.invert_yaxis()
        ax_imp.set_xlabel("Importance")
        ax_imp.set_title("Feature Importance")
        st.session_state.feature_figs.append(("Random Forest", fig_imp))
    elif model_choice == "XGBoost":
        importances = model.feature_importances_
        feat_names = X_train.columns
        imp_order = np.argsort(importances)[::-1]
        fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
        ax_imp.barh(feat_names[imp_order], importances[imp_order], color="#FAA43A")
        ax_imp.invert_yaxis()
        ax_imp.set_xlabel("Importance")
        ax_imp.set_title("Feature Importance")
        st.session_state.feature_figs.append(("XGBoost", fig_imp))
    elif model_choice == "Stacking (RF + XGB)":
        rf_model = model.named_estimators_["rf"]
        xgb_model = model.named_estimators_["xgb"]
        if hasattr(rf_model, "feature_importances_"):
            importances = rf_model.feature_importances_
            feat_names = X_train.columns
            imp_order = np.argsort(importances)[::-1]
            fig_rf, ax_rf = plt.subplots(figsize=(6, 4))
            ax_rf.barh(feat_names[imp_order], importances[imp_order], color="steelblue")
            ax_rf.invert_yaxis()
            ax_rf.set_xlabel("Importance")
            ax_rf.set_title("Feature Importance (RF)")
            st.session_state.feature_figs.append(("Random Forest", fig_rf))
        if hasattr(xgb_model, "feature_importances_"):
            importances = xgb_model.feature_importances_
            feat_names = X_train.columns
            imp_order = np.argsort(importances)[::-1]
            fig_xgb, ax_xgb = plt.subplots(figsize=(6, 4))
            ax_xgb.barh(feat_names[imp_order], importances[imp_order], color="#FAA43A")
            ax_xgb.invert_yaxis()
            ax_xgb.set_xlabel("Importance")
            ax_xgb.set_title("Feature Importance (XGB)")
            st.session_state.feature_figs.append(("XGBoost", fig_xgb))

# Fragment: Display results independently
@st.fragment
def display_results():
    if not st.session_state.get("trained", False):
        st.info("Configure parameters and click **Train** to train a model.")
        return
    if st.session_state.is_classification:
        st.subheader("Classification Performance")
        st.write(f"**Accuracy:** {st.session_state.accuracy:.3f}")
        st.write("**Confusion Matrix:**")
        st.pyplot(st.session_state.confusion_fig)
        if st.session_state.roc_fig is not None:
            st.write("**ROC Curve:**")
            st.pyplot(st.session_state.roc_fig)
    else:
        st.subheader("Regression Performance")
        st.write(f"**RMSE:** {st.session_state.rmse:.3f}")
        st.write(f"**R²:** {st.session_state.r2:.3f}")
        st.write("**Residuals Distribution:**")
        st.pyplot(st.session_state.residual_fig)
    if st.session_state.feature_figs:
        st.subheader("Feature Importance")
        if len(st.session_state.feature_figs) > 1:
            cols = st.columns(len(st.session_state.feature_figs))
            for idx, (label, fig) in enumerate(st.session_state.feature_figs):
                with cols[idx]:
                    st.pyplot(fig)
                    st.caption(f"{label}")
        else:
            label, fig = st.session_state.feature_figs[0]
            st.pyplot(fig)
            st.caption(f"{label}")

# Display results fragment on main page
display_results()
