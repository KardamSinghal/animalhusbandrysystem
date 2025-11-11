import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix

warnings.filterwarnings('ignore')

# ========================
# Page Configuration
# ========================
st.set_page_config(
    page_title="Animal Husbandry ML System",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Light theme CSS
st.markdown(
    """
    <style>
    .main { background-color: #f5f7fa; color: #2c3e50; }
    .stApp { background-color: #f5f7fa; }
    h1, h2, h3, h4, h5, h6 { color: #2c3e50 !important; }
    .stMetric { background: #ffffff; border-radius: 10px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    .stButton>button { background:#2c7be5; color:#fff; border:none; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ========================
# Navigation jump handler (fixes nav set error)
# Must run BEFORE sidebar is created
# ========================
if "pending_nav" in st.session_state and st.session_state["pending_nav"]:
    st.session_state["nav"] = st.session_state["pending_nav"]
    st.session_state["pending_nav"] = ""

# ========================
# Helpers
# ========================
def coerce_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s_clean = s.str.replace(r'[^0-9\.\-]', '', regex=True)
    num = pd.to_numeric(s_clean, errors='coerce')
    return num

def maybe_parse_datetime_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    out = pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
    return out

def datetime_to_float(s: pd.Series) -> pd.Series:
    if not np.issubdtype(s.dtype, np.datetime64):
        s = pd.to_datetime(s, errors='coerce')
    vals = s.view('int64').astype('float64')  # ns -> float
    vals[s.isna()] = np.nan
    return vals / 1e9  # seconds

def make_numeric_for_ml(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.datetime64):
            X[col] = datetime_to_float(X[col])
    return X

def corr_numeric(df: pd.DataFrame):
    tmp = pd.DataFrame(index=df.index)
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            tmp[col] = df[col]
        elif np.issubdtype(df[col].dtype, np.datetime64):
            tmp[col] = datetime_to_float(df[col])
    tmp = tmp.select_dtypes(include=[np.number])
    tmp = tmp.loc[:, tmp.nunique(dropna=True) > 1]
    if tmp.shape[1] >= 2:
        return tmp.corr()
    return None

# ========================
# Data Loading
# ========================
@st.cache_data(ttl=3600, show_spinner=False)
def load_dataset(filename, max_rows=15000):
    """
    Load CSV safely:
    - Read everything as string first
    - Convert date-like columns by name (month/date/time/year)
    - Convert others to numeric if possible (robust cleaning)
    - Special handling for known milk dataset (Production + Month)
    """
    filepath = f'data/{filename}'
    try:
        if not os.path.exists(filepath):
            return None, f"File not found: {filepath}"

        df = pd.read_csv(filepath, nrows=max_rows, low_memory=False, dtype=str)
        df.columns = df.columns.str.strip()

        # Rename for milk dataset
        for col in list(df.columns):
            cl = col.lower()
            if 'month' in cl and 'production' not in cl and 'month' not in df.columns:
                df = df.rename(columns={col: 'Month'})
            if ('milk' in cl or 'production' in cl or 'pounds' in cl) and 'production' not in df.columns:
                df = df.rename(columns={col: 'Production'})

        # Type conversion
        for col in df.columns:
            cl = col.lower()
            s = df[col].astype(str).str.strip()

            if any(k in cl for k in ['date', 'month', 'time', 'year']):
                dt = maybe_parse_datetime_series(s)
                if dt.notna().sum() >= max(3, int(0.3 * len(dt))):
                    df[col] = dt
                else:
                    num = coerce_numeric_series(s)
                    if num.notna().sum() >= max(3, int(0.3 * len(num))):
                        df[col] = num
                    else:
                        df[col] = s
            else:
                num = coerce_numeric_series(s)
                if num.notna().sum() >= max(3, int(0.3 * len(num))):
                    df[col] = num
                else:
                    df[col] = s

        # Enforce milk dataset fields if possible
        if 'milk' in filename.lower():
            if 'Production' in df.columns and not pd.api.types.is_numeric_dtype(df['Production']):
                df['Production'] = coerce_numeric_series(df['Production'])
            if 'Production' not in df.columns:
                for col in df.columns:
                    if any(k in col.lower() for k in ['milk', 'production', 'pounds']):
                        df['Production'] = coerce_numeric_series(df[col])
                        break
            if 'Month' in df.columns and not np.issubdtype(df['Month'].dtype, np.datetime64):
                df['Month'] = maybe_parse_datetime_series(df['Month'])

        return df, None
    except Exception as e:
        return None, f"Error loading {filename}: {str(e)}"

@st.cache_data(ttl=3600, show_spinner=False)
def check_available_datasets():
    datasets = {
        'animal_behavior.csv': 'Animal Behavior',
        'animal_disease.csv': 'Animal Disease',
        'milk_production.csv': 'Milk Production',
        'livestock_disease.csv': 'Livestock Disease',
        'dairy_prices.csv': 'Dairy Prices',
        'meat_dairy.csv': 'Meat & Dairy'
    }
    available = {}
    for filename, name in datasets.items():
        exists = os.path.exists(f'data/{filename}')
        available[name] = {
            'filename': filename,
            'loaded': exists,
            'df': None,
            'error': None if exists else f"File not found"
        }
    return available

# ========================
# ML Training
# ========================
@st.cache_resource(show_spinner=False)
def train_classification_model(df, target_col, feature_cols):
    try:
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        X = make_numeric_for_ml(X)
        X = X.fillna(X.mean(numeric_only=True))

        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        target_encoder = None
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index)
        elif isinstance(y, np.ndarray):
            y = pd.Series(y)

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        class_counts = y.value_counts()
        valid_classes = class_counts[class_counts >= 2].index.tolist()
        if len(valid_classes) < 2:
            raise ValueError("Need at least 2 classes with 2+ samples each for classification")
        mask = y.isin(valid_classes)
        X, y = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)

        can_stratify = (y.value_counts() >= 2).all() and y.nunique() >= 2

        if can_stratify:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        model = RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=42, n_jobs=-1, max_features='sqrt'
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classes = target_encoder.classes_ if target_encoder is not None else np.unique(y)

        return {
            'model': model,
            'accuracy': accuracy,
            'feature_cols': feature_cols,
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'classes': classes
        }
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def train_regression_model(df, target_col, feature_cols):
    try:
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        X = make_numeric_for_ml(X)
        if np.issubdtype(y.dtype, np.datetime64):
            y = datetime_to_float(y)

        X = X.fillna(X.mean(numeric_only=True))
        y = y.fillna(y.mean())

        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = GradientBoostingRegressor(
            n_estimators=50, max_depth=4, random_state=42, learning_rate=0.1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = float(np.mean(np.abs(y_test - y_pred)))

        return {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'feature_cols': feature_cols,
            'label_encoders': label_encoders,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    except Exception as e:
        st.error(f"Error training regression model: {str(e)}")
        return None

# ========================
# Visualization
# ========================
def create_optimized_plot(df, plot_type, x=None, y=None, **kwargs):
    MAX_POINTS = 1000
    if len(df) > MAX_POINTS:
        df = df.sample(n=MAX_POINTS, random_state=42)
        if x and x in df.columns:
            try:
                df = df.sort_values(x)
            except:
                pass
    try:
        if plot_type == 'histogram':
            return px.histogram(df, x=x, nbins=30, **kwargs)
        elif plot_type == 'scatter':
            return px.scatter(df, x=x, y=y, **kwargs)
        elif plot_type == 'line':
            return px.line(df, x=x, y=y, **kwargs)
        elif plot_type == 'box':
            return px.box(df, y=y, **kwargs)
        elif plot_type == 'bar':
            return px.bar(df, x=x, y=y, **kwargs)
        else:
            return px.scatter(df, x=x, y=y, **kwargs)
    except Exception as e:
        st.error(f"Plot error: {str(e)}")
        return None

# ========================
# Sidebar (Light theme, Clean Status)
# ========================
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center">
            <img src="https://img.icons8.com/color/96/000000/cow.png" width="80"/>
            <h3 style="margin-bottom:0">Smart Farm ML</h3>
            <span style="color:#7f8c8d">Animal Husbandry AI Dashboard</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    datasets = check_available_datasets()
    total_ds = len(datasets)
    loaded_ds = sum(1 for d in datasets.values() if d["loaded"])
    st.metric("Datasets Loaded", f"{loaded_ds}/{total_ds}")
    st.progress(loaded_ds / total_ds if total_ds else 0.0)

    st.caption("Status")
    c1, c2 = st.columns(2)
    for i, (name, info) in enumerate(datasets.items()):
        icon = "✅" if info["loaded"] else "❌"
        bg = "#d4edda" if info["loaded"] else "#fdecea"
        fg = "#155724" if info["loaded"] else "#611a15"
        pill_html = f"""
            <div style="
                display:flex;justify-content:space-between;align-items:center;
                padding:6px 10px;margin-bottom:6px;border-radius:8px;
                background:{bg}; color:{fg}; border:1px solid rgba(0,0,0,0.05);
                font-size: 13px;
            ">
                <span>{name}</span>
                <span>{icon}</span>
            </div>
        """
        with (c1 if i % 2 == 0 else c2):
            st.markdown(pill_html, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("⚡ Quick Actions")
    a1, a2 = st.columns(2)
    with a1:
        if st.button("♻️ Clear cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.toast("Cache cleared", icon="✅")
            st.rerun()
    with a2:
        if st.button("🔁 Refresh app", use_container_width=True):
            st.rerun()

    with st.expander("📥 Upload/Replace Dataset"):
        file_map = {
            "Animal Behavior": "animal_behavior.csv",
            "Animal Disease": "animal_disease.csv",
            "Milk Production": "milk_production.csv",
            "Dairy Prices": "dairy_prices.csv",
            "Livestock Disease": "livestock_disease.csv",
            "Meat & Dairy": "meat_dairy.csv",
        }
        choice = st.selectbox("Select dataset slot", list(file_map.keys()))
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            os.makedirs("data", exist_ok=True)
            target = os.path.join("data", file_map[choice])
            with open(target, "wb") as f:
                f.write(up.read())
            st.success(f"Saved to data/{file_map[choice]}")
            st.toast("Upload complete. Reloading...", icon="✅")
            st.rerun()

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Data Explorer",
            "🏥 Health Analysis",
            "🦠 Disease Prediction",
            "🥛 Milk Production",
            "📈 Advanced Analytics",
            "📄 Reports",
        ],
        key="nav",
    )

    st.markdown("---")
    st.caption("🐄 Animal Husbandry ML System")

# ========================
# HOME PAGE (Quick Start with st.rerun)
# ========================
if page == "🏠 Home":
    st.title("🐄 Animal Husbandry ML Management System")
    st.caption("Using real Kaggle datasets for health, disease, and milk forecasting")

    ds = datasets
    loaded = {name: info for name, info in ds.items() if info["loaded"]}
    total_ds = len(ds)
    loaded_ds = len(loaded)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Datasets Loaded", f"{loaded_ds}/{total_ds}")
    with c2: st.metric("Modules", "Health • Disease • Milk")
    with c3: st.metric("Cache", "Active")
    with c4: st.metric("Status", "Ready" if loaded_ds else "Waiting")

    st.markdown("---")

    st.subheader("🚀 Quick Start")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("📊 Explore Data", use_container_width=True):
            st.session_state["pending_nav"] = "📊 Data Explorer"
            st.rerun()
    with q2:
        if st.button("🏥 Health ML", use_container_width=True):
            st.session_state["pending_nav"] = "🏥 Health Analysis"
            st.rerun()
    with q3:
        if st.button("🦠 Disease ML", use_container_width=True):
            st.session_state["pending_nav"] = "🦠 Disease Prediction"
            st.rerun()
    with q4:
        if st.button("🥛 Milk Forecast", use_container_width=True):
            st.session_state["pending_nav"] = "🥛 Milk Production"
            st.rerun()

    st.markdown("---")

    st.subheader("📦 Dataset Status")
    grid_cols = st.columns(3)
    idx = 0
    for name, info in ds.items():
        with grid_cols[idx % 3]:
            box = st.container(border=True)
            with box:
                st.markdown(f"**{name}**")
                if info["loaded"]:
                    st.markdown("<div style='color:#155724;background:#d4edda;border-radius:8px;padding:6px 10px;margin-bottom:6px;'>Available ✅</div>", unsafe_allow_html=True)
                    with st.expander("Preview 5 rows"):
                        dfp, _ = load_dataset(info["filename"], max_rows=2000)
                        if dfp is not None:
                            st.write(f"Rows: {len(dfp)} • Cols: {len(dfp.columns)}")
                            st.dataframe(dfp.head(5), use_container_width=True)
                else:
                    st.markdown("<div style='color:#611a15;background:#fdecea;border-radius:8px;padding:6px 10px;margin-bottom:6px;'>Missing ❌</div>", unsafe_allow_html=True)
        idx += 1

# ========================
# DATA EXPLORER PAGE
# ========================
elif page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    available = {n: i for n, i in datasets.items() if i['loaded']}
    if not available:
        st.error("No datasets loaded")
        st.stop()

    choice = st.selectbox("Select Dataset", list(available.keys()))
    df, _ = load_dataset(available[choice]['filename'])
    if df is None:
        st.error("Failed to load dataset")
        st.stop()

    st.info(f"📊 {df.shape[0]} rows × {df.shape[1]} cols")
    tabs = st.tabs(["📋 Data", "📊 Visualizations", "🔍 Quality"])

    with tabs[0]:
        n = st.slider("Rows to show", 5, 100, 20)
        st.dataframe(df.head(n), use_container_width=True)
        st.markdown("### Statistics")
        st.dataframe(df.describe(include='all'), use_container_width=True)

    with tabs[1]:
        corr = corr_numeric(df)
        if corr is not None:
            st.markdown("#### Correlation Heatmap")
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric variability to compute a correlation matrix.")

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            missing = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing': missing.values,
                'Percentage': (missing / len(df) * 100).round(2)
            }).sort_values('Missing', ascending=False)
            st.dataframe(missing_df, use_container_width=True)
        with c2:
            st.metric("Duplicates", df.duplicated().sum())
            unique = df.nunique()
            st.dataframe(pd.DataFrame({'Column': unique.index, 'Unique': unique.values}).head(10), use_container_width=True)

# ========================
# HEALTH ANALYSIS PAGE
# ========================
elif page == "🏥 Health Analysis":
    st.title("🏥 Health Analysis")
    if not datasets['Animal Behavior']['loaded']:
        st.error("animal_behavior.csv not found")
        st.stop()

    df, _ = load_dataset('animal_behavior.csv')
    if df is None:
        st.error("Failed to load")
        st.stop()

    st.success(f"✅ {len(df)} records loaded")

    all_cols = df.columns.tolist()
    st.dataframe(df.head(10), use_container_width=True)

    target = st.selectbox("Target", all_cols)
    features = st.multiselect("Features", [c for c in all_cols if c != target],
                              default=[c for c in all_cols if c != target][:5])

    if features and st.button("🚀 Train Model", type="primary"):
        train_df = df[[target] + features].dropna(subset=[target])
        if len(train_df) < 10:
            st.error("Not enough data")
            st.stop()

        is_clf = df[target].nunique() < 20 or df[target].dtype == 'object'
        result = train_classification_model(train_df, target, features) if is_clf \
                 else train_regression_model(train_df, target, features)
        if result:
            st.success("✅ Model trained!")
            if 'accuracy' in result:
                st.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
                imp = pd.DataFrame({'Feature': features, 'Importance': result['model'].feature_importances_}).sort_values('Importance', ascending=False)
                fig = px.bar(imp, x='Importance', y='Feature', orientation='h')
                st.plotly_chart(fig, use_container_width=True)
                cm = confusion_matrix(result['y_test'], result['y_pred'])
                labels = result['target_encoder'].classes_ if result['target_encoder'] else result['classes']
                fig = px.imshow(cm, x=labels, y=labels, text_auto=True, labels=dict(x="Predicted", y="Actual"), color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
            else:
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("R²", f"{result['r2']:.3f}")
                with c2: st.metric("RMSE", f"{result['rmse']:.2f}")
                with c3: st.metric("MAE", f"{result['mae']:.2f}")

# ========================
# DISEASE PREDICTION PAGE
# ========================
elif page == "🦠 Disease Prediction":
    st.title("🦠 Disease Prediction")
    if not datasets['Animal Disease']['loaded']:
        st.error("animal_disease.csv not found")
        st.stop()

    df, _ = load_dataset('animal_disease.csv')
    if df is None:
        st.error("Failed to load")
        st.stop()

    st.success(f"✅ {len(df)} records loaded")

    all_cols = df.columns.tolist()
    st.dataframe(df.head(10), use_container_width=True)

    target = st.selectbox("Target Column", all_cols)
    features = st.multiselect("Feature Columns", [c for c in all_cols if c != target],
                              default=[c for c in all_cols if c != target][:8])

    if features and st.button("🚀 Train Model", type="primary"):
        train_df = df[[target] + features].dropna(subset=[target])
        if len(train_df) < 10:
            st.error("Not enough data")
            st.stop()
        result = train_classification_model(train_df, target, features)
        if result:
            st.success(f"✅ Accuracy: {result['accuracy']*100:.2f}%")
            imp = pd.DataFrame({'Feature': features, 'Importance': result['model'].feature_importances_}).sort_values('Importance', ascending=False)[:10]
            fig = px.bar(imp, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig, use_container_width=True)

# ========================
# MILK PRODUCTION PAGE
# ========================
elif page == "🥛 Milk Production":
    st.title("🥛 Milk Production Analysis & Forecasting")
    if not datasets['Milk Production']['loaded']:
        st.error("milk_production.csv not found")
        st.stop()

    df, _ = load_dataset('milk_production.csv')
    if df is None:
        st.error("Failed to load dataset")
        st.stop()

    st.success(f"✅ {len(df)} records loaded")

    prod_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ['production', 'milk', 'pounds']) and pd.api.types.is_numeric_dtype(df[c]):
            prod_col = c; break
    if prod_col is None:
        num_cols_temp = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols_temp:
            prod_col = num_cols_temp[0]

    date_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ['month', 'date', 'time', 'year']) and np.issubdtype(df[c].dtype, np.datetime64):
            date_col = c; break
    if date_col is None:
        for c in df.columns:
            if 'month' in c.lower():
                maybe = maybe_parse_datetime_series(df[c])
                if maybe.notna().sum() >= max(3, int(0.3 * len(maybe))):
                    df[c] = maybe; date_col = c; break

    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['Year'] = df[date_col].dt.year
            df['Month_Num'] = df[date_col].dt.month
            df['Day_of_Year'] = df[date_col].dt.dayofyear
        except:
            pass

    st.markdown("#### Dataset Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': [str(df[c].dtype) for c in df.columns],
        'Non-Null': [int(df[c].notna().sum()) for c in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)

    tabs = st.tabs(["📈 Analysis", "🔮 Forecast"])

    with tabs[0]:
        st.dataframe(df.head(12), use_container_width=True)
        if prod_col and pd.api.types.is_numeric_dtype(df[prod_col]):
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Total", f"{df[prod_col].sum():,.0f}")
            with c2: st.metric("Average", f"{df[prod_col].mean():,.0f}")
            with c3: st.metric("Max", f"{df[prod_col].max():,.0f}")
            with c4: st.metric("Min", f"{df[prod_col].min():,.0f}")

            if date_col:
                fig = px.line(df.dropna(subset=[date_col, prod_col]).sort_values(date_col), x=date_col, y=prod_col)
                st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.histogram(df, x=prod_col, nbins=30)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.box(df, y=prod_col)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Could not find a numeric Production column. Please verify the CSV.")

    with tabs[1]:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            df['Index'] = np.arange(len(df))
            num_cols = ['Index']
        elif prod_col and prod_col in num_cols and len(num_cols) == 1:
            df['Index'] = np.arange(len(df))
            num_cols.append('Index')

        default_target = prod_col if prod_col in num_cols else num_cols[0]
        target = st.selectbox("Target Variable", num_cols, index=num_cols.index(default_target) if default_target in num_cols else 0)
        features = st.multiselect(
            "Feature Variables",
            [c for c in num_cols if c != target],
            default=[c for c in ['Year', 'Month_Num', 'Day_of_Year', 'Index'] if c in num_cols] or
                    [c for c in num_cols if c != target][:min(3, max(1, len(num_cols)-1))]
        )

        if not features:
            st.warning("Select at least one feature")
            st.stop()

        if st.button("🚀 Train Forecasting Model", type="primary"):
            train_df = df[[target] + features].dropna()
            if len(train_df) < 10:
                st.error("Not enough rows after dropping missing values")
                st.stop()

            result = train_regression_model(train_df, target, features)
            if result:
                st.success("✅ Model trained!")
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("R²", f"{result['r2']:.3f}")
                with c2: st.metric("RMSE", f"{result['rmse']:.2f}")
                with c3: st.metric("MAE", f"{result['mae']:.2f}")
                imp = pd.DataFrame({'Feature': features, 'Importance': result['model'].feature_importances_}).sort_values('Importance', ascending=False)
                fig = px.bar(imp, x='Importance', y='Feature', orientation='h', title='Feature Importance')
                st.plotly_chart(fig, use_container_width=True)

# ========================
# ADVANCED ANALYTICS PAGE
# ========================
elif page == "📈 Advanced Analytics":
    st.title("📈 Advanced Analytics")
    available = {n: i for n, i in datasets.items() if i['loaded']}
    if not available:
        st.error("No datasets")
        st.stop()
    choice = st.selectbox("Dataset", list(available.keys()))
    df, _ = load_dataset(available[choice]['filename'])
    if df is None:
        st.error("Failed to load")
        st.stop()

    corr = corr_numeric(df)
    if corr is not None:
        st.markdown("### Correlation Matrix")
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric variability to compute correlation.")

    st.markdown("### Statistics")
    st.dataframe(df.describe(include='all'), use_container_width=True)

# ========================
# REPORTS PAGE
# ========================
elif page == "📄 Reports":
    st.title("📄 Reports & Export")
    available = check_available_datasets()

    tabs = st.tabs(["📊 Generate", "📥 Export"])
    with tabs[0]:
        if st.button("📄 Generate Report", type="primary"):
            report = f"""
{'='*60}
ANIMAL HUSBANDRY ML SYSTEM REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASETS:
"""
            for name, info in available.items():
                if info['loaded']:
                    d, _ = load_dataset(info['filename'])
                    if d is not None:
                        report += f"\n{name}: Rows={len(d)}, Cols={len(d.columns)}, Missing={int(d.isnull().sum().sum())}"
            if 'health_model' in st.session_state:
                report += f"\n\nHEALTH MODEL: Accuracy={st.session_state['health_model']['accuracy']*100:.2f}%"
            if 'disease_model' in st.session_state:
                report += f"\nDISEASE MODEL: Accuracy={st.session_state['disease_model']['accuracy']*100:.2f}%"
            if 'milk_model' in st.session_state:
                report += f"\nMILK MODEL: R²={st.session_state['milk_model']['r2']:.4f}, RMSE={st.session_state['milk_model']['rmse']:.2f}"
            report += f"\n{'='*60}\n"
            st.text_area("Report Preview", report, height=400)
            st.download_button("📥 Download Report", report, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")

    with tabs[1]:
        st.markdown("### Export Datasets")
        for name, info in available.items():
            if info['loaded']:
                d, _ = load_dataset(info['filename'])
                if d is not None:
                    c1, c2 = st.columns([3, 1])
                    with c1: st.write(f"**{name}** - {len(d)} rows × {len(d.columns)} cols")
                    with c2:
                        st.download_button("📥 CSV", d.to_csv(index=False), f"{info['filename']}", key=f"dl_{name}")

# Footer
st.markdown("---")
st.caption("🐄 Animal Husbandry ML System | © 2025")