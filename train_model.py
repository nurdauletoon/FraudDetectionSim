# train_model.py
# Fraud Detection Dashboard - исправленная версия с вкладкой "📈 Обучение модели"
# Автор: восстановление и правки по запросу пользователя

import os
import io
import re
import json
import toml
import random
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

# ----------------------------
# Helpers for secrets / session df
# ----------------------------
def load_secrets(path: str = "secrets.toml") -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    for enc in ("utf-8", "utf-8-sig", "utf-16", "latin1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return toml.load(f)
        except Exception:
            continue
    raise ValueError("Не удалось загрузить secrets.toml — пересохраните файл в UTF-8 без BOM или проверьте кодировку.")


def get_session_df() -> pd.DataFrame | None:
    df = st.session_state.get('df')
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    df_raw = st.session_state.get('df_raw')
    if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
        return df_raw
    return None

# ----------------------------
# Load secrets safely
# ----------------------------
SECRETS_FILE = "secrets.toml"
try:
    secrets = load_secrets(SECRETS_FILE)
except Exception as e:
    try:
        st.warning(f"Не удалось загрузить {SECRETS_FILE}: {e}")
    except Exception:
        print(f"Warning: Не удалось загрузить {SECRETS_FILE}: {e}")
    secrets = {}

GEMINI_API_KEY = secrets.get("GEMINI_API_KEY") or secrets.get("GEMINI") or secrets.get("GEMINI_KEY")

# ----------------------------
# Optional: try import Gemini lib (not critical for training)
# ----------------------------
GENAI_AVAILABLE = False
try:
    import google.generativeai as genai  # type: ignore
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False
    genai = None

# ----------------------------
# Streamlit page config + styles
# ----------------------------
st.set_page_config(
    page_title="Fraud Detection AI - Trainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern dark theme, glassmorphism, animations
st.markdown(
    """
    <style>
    /* Base */
    :root{
        --bg:#071021;
        --panel:#0b1622;
        --muted:#9AA6B2;
        --accent1: linear-gradient(90deg, #06b6d4 0%, #7c3aed 100%);
        --accent:#06b6d4;
        --glass: rgba(255,255,255,0.03);
        --card-shadow: 0 6px 18px rgba(2,6,23,0.6);
    }
    html, body, #root, .streamlit-container {
        background: radial-gradient(1200px 600px at 10% 10%, rgba(124,58,237,0.08), transparent),
                    radial-gradient(1000px 500px at 90% 90%, rgba(6,182,212,0.04), transparent),
                    var(--bg) !important;
        color: #E6EEF3;
        font-family: "Inter", "Segoe UI", Roboto, sans-serif;
    }
    /* Sidebar */
    .css-1d391kg { padding-top: 1rem; } /* small fix for Streamlit internal classes */
    .stSidebar .css-1d391kg, .stSidebar .css-1d391kg * {
        color: #E6EEF3;
    }
    .stSidebar .stButton>button {
        background: linear-gradient(90deg,#06b6d4,#7c3aed);
        border: none;
        color: #001219;
        font-weight: 600;
        padding: 10px 14px;
        box-shadow: 0 6px 18px rgba(124,58,237,0.12);
        border-radius: 10px;
    }
    /* Header */
    .header-row { display:flex; align-items:center; gap:12px; }
    .app-title { font-size:20px; font-weight:700; letter-spacing:0.2px; }
    .app-sub { color:var(--muted); font-size:12px; margin-top:-6px; }

    /* Glass metric cards */
    .metric-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
        border-radius:14px;
        padding:14px;
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(255,255,255,0.04);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .metric-card:hover { transform: translateY(-6px); box-shadow: 0 14px 30px rgba(2,6,23,0.65); }

    .metric-title { font-size:13px; color:var(--muted); margin:0; }
    .metric-value { font-size:22px; font-weight:700; margin:6px 0 0 0; color:#E6EEF3; }

    /* Buttons primary */
    .btn-primary {
        background: linear-gradient(90deg,#06b6d4,#7c3aed);
        color:#001219 !important;
        border:none;
        padding:8px 12px;
        border-radius:10px;
        font-weight:700;
        box-shadow: 0 8px 20px rgba(7,22,42,0.6);
    }
    .secondary {
        background: transparent;
        border: 1px solid rgba(255,255,255,0.06);
        color: var(--muted);
        padding:8px 12px;
        border-radius:10px;
    }

    /* Small text */
    .small { font-size:12px; color:var(--muted); }

    /* Fancy table */
    .stDataFrame table {
        background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005));
        border-radius:8px;
        overflow:hidden;
    }

    /* Upload box */
    .upload-box {
        border: 1px dashed rgba(255,255,255,0.06);
        padding: 10px;
        border-radius: 10px;
        background: linear-gradient(180deg, rgba(255,255,255,0.01), transparent);
    }

    /* Animated badge */
    .badge {
        display:inline-block;
        padding:6px 10px;
        border-radius:999px;
        background: linear-gradient(90deg,#7c3aed,#06b6d4);
        color:#001219;
        font-weight:700;
        font-size:12px;
        box-shadow: 0 8px 24px rgba(124,58,237,0.12);
    }

    /* small responsive fixes */
    @media (max-width: 900px) {
        .metric-value { font-size:18px; }
        .app-title { font-size:18px; }
    }

    /* subtle fade-in animations for main content */
    .fade-in { animation: fadeInEase .45s ease-in; }
    @keyframes fadeInEase { from { opacity:0; transform: translateY(6px);} to {opacity:1; transform: translateY(0);} }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------
# Utility functions
# ----------------------------
def safe_json_extract(text: str):
    if not text:
        return None
    try:
        cleaned = re.sub(r"```(?:json|JSON)?", "", text)
        cleaned = cleaned.replace("```", "").strip()
        m = re.search(r"(\[.*\])", cleaned, re.S)
        if m:
            return json.loads(m.group(1))
        m2 = re.search(r"(\{.*\})", cleaned, re.S)
        if m2:
            return json.loads(m2.group(1))
        return json.loads(cleaned)
    except Exception:
        return None

def ensure_columns_upper(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [c.strip() for c in df2.columns]
    return df2

def try_parse_datetime_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s)
    except Exception:
        return pd.to_datetime(s, errors='coerce')

# ----------------------------
# Model training helpers
# ----------------------------
def build_supervised_pipeline(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = None
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
    if categorical_transformer:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
    else:
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols)], remainder='drop')

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs'))
    ])
    return pipeline

def train_supervised(df: pd.DataFrame, target_col='Class'):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    df = df[~df[target_col].isna()].copy()
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    # drop id/time from categoricals
    for c in list(categorical_cols):
        if c.lower() in ['id', 'time', 'timestamp', 'date']:
            categorical_cols.remove(c)

    pipeline = build_supervised_pipeline(numeric_cols, categorical_cols)
    stratify = y if (y.nunique() > 1 and len(y) > 20) else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    try:
        report = classification_report(y_test, y_pred, output_dict=True)
    except Exception:
        report = {"accuracy": float((y_pred == y_test).mean())}

    # Feature importances (coeffs from logistic)
    fi = pd.DataFrame(columns=['Feature', 'Importance'])
    try:
        classifier = pipeline.named_steps['classifier']
        pre = pipeline.named_steps['preprocessor']
        feature_names = []
        if 'num' in pre.named_transformers_:
            feature_names.extend(numeric_cols)
        if 'cat' in pre.named_transformers_:
            ohe = pre.named_transformers_['cat'].named_steps['onehot']
            cat_cols = categorical_cols
            ohe_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
            feature_names.extend(ohe_feature_names)
        coefs = classifier.coef_[0]
        ln = min(len(coefs), len(feature_names))
        fi = pd.DataFrame({'Feature': feature_names[:ln], 'Importance': coefs[:ln]})
        fi['AbsImportance'] = fi['Importance'].abs()
        fi = fi.sort_values(by='AbsImportance', ascending=False).drop(columns=['AbsImportance']).reset_index(drop=True)
    except Exception:
        fi = pd.DataFrame(columns=['Feature', 'Importance'])

    return pipeline, report, fi, df

def train_unsupervised(df: pd.DataFrame, n_estimators=100, contamination=0.05):
    # We'll use numeric features only for IsolationForest by default
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number]).copy()
    if numeric.empty:
        raise ValueError("Нет числовых признаков для обучения аномалий.")
    # impute
    numeric = numeric.fillna(numeric.median())
    iso = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    iso.fit(numeric)
    scores = -iso.decision_function(numeric)  # higher -> more anomalous
    pred = iso.predict(numeric)  # -1 anomaly, 1 normal
    df['anomaly_score'] = scores
    df['pred_anomaly'] = np.where(pred == -1, 1, 0)
    report = {"unsupervised": True, "contamination": contamination, "anomalies": int(df['pred_anomaly'].sum())}
    fi = pd.DataFrame(columns=['Feature', 'Importance'])
    return iso, report, fi, df

# ----------------------------
# Demo data generator
# ----------------------------
def generate_demo_transactions(n=500):
    countries = ['KZ', 'RU', 'US', 'CN', 'DE', 'GB', 'FR', 'IN', 'BR']
    merchants = ['Kaspi', 'Halyk', 'Amazon', 'Apple', 'Sber', 'Visa', 'Mastercard']
    data = []
    for i in range(n):
        is_fraud = random.random() < 0.05
        amount = float(np.random.lognormal(4, 1.2)) if not is_fraud else float(np.random.lognormal(6, 1.5))
        ts = datetime.now() - timedelta(hours=random.randint(0, 720))
        data.append({
            'id': f'TXN_{i:06d}',
            'Amount': round(amount, 2),
            'Country': random.choice(countries),
            'Merchant': random.choice(merchants),
            'Time': ts.strftime("%Y-%m-%d %H:%M:%S"),
            'Class': 1 if is_fraud else 0,
            'fraud_probability': round(random.uniform(0.8, 0.99), 4) if is_fraud else round(random.uniform(0.01, 0.3), 4)
        })
    return pd.DataFrame(data)

# ----------------------------
# Gemini search stub (keeps old behaviour; not required for training)
# ----------------------------
def gemini_search_videos(query: str, max_results: int = 6, model_name: str = "gemini-2.5-flash"):
    # If genai not available, return helpful message
    if not GENAI_AVAILABLE:
        return [{"title": "Gemini unavailable", "url": "", "description": "Install google-generativeai or skip video search."}]
    # In production you'd call Gemini here. For now return empty.
    return []

# ----------------------------
# Dashboard class
# ----------------------------
class FraudDetectionDashboard:
    def __init__(self):
        self.demo_transactions = generate_demo_transactions(500)
        # init session keys safely
        if 'model_trained' not in st.session_state:
            st.session_state['model_trained'] = False
            st.session_state['model'] = None
            st.session_state['report'] = None
            st.session_state['feat_importance'] = None
            st.session_state['df'] = None
            st.session_state['df_raw'] = None
            st.session_state['current_file_name'] = None
            st.session_state['model_type'] = None
            st.session_state['model_file'] = "fraud_model.pkl"

        self.render_sidebar()
        self.route_page()

    def render_sidebar(self):
        with st.sidebar:
            st.markdown("# ⚙️ Управление")
            st.markdown("Загрузите CSV файл с транзакциями (если есть колонка `Class` — будет супервизорное обучение).")
            uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"], key="uploader_main")

            st.markdown("---")
            st.markdown("## Навигация")
            # added training tab here
            self.page = st.radio(
                "Выберите раздел:",
                ["📊 Дашборд", "📋 История транзакций", "🤖 Анализ модели", "📂 CSV Анализ", "🏭 3D Модель", "🌍 Карта транзакций",  "📈 Обучение модели"]
            )
            st.markdown("---")
            st.markdown("Версия: full-1.1")
            st.markdown("Разработчик: Нурдаулет Урынбайулы")

            if uploaded_file is not None:
                try:
                    file_bytes = uploaded_file.read()
                    # only retrain when a new file uploaded (or explicitly on tab)
                    if st.session_state.get('current_file_name') != uploaded_file.name:
                        with st.spinner("Идёт загрузка CSV и обучение модели..."):
                            df = pd.read_csv(io.BytesIO(file_bytes))
                            df = ensure_columns_upper(df)
                            st.session_state['df_raw'] = df.copy()
                            # auto-train (supervised if Class present, otherwise unsupervised)
                            try:
                                if 'Class' in df.columns:
                                    model, report, fi, cleaned = train_supervised(df, target_col='Class')
                                    st.session_state['model_trained'] = True
                                    st.session_state['model_type'] = 'supervised'
                                else:
                                    model, report, fi, cleaned = train_unsupervised(df)
                                    st.session_state['model_trained'] = True
                                    st.session_state['model_type'] = 'unsupervised'
                                st.session_state['model'] = model
                                st.session_state['report'] = report
                                st.session_state['feat_importance'] = fi
                                st.session_state['df'] = cleaned
                                st.session_state['current_file_name'] = uploaded_file.name
                                # save model
                                try:
                                    joblib.dump(model, st.session_state.get('model_file', 'fraud_model.pkl'))
                                except Exception:
                                    pass
                                st.success("✅ Модель обучена и данные загружены")
                            except Exception as e:
                                st.session_state['model_trained'] = False
                                st.error(f"Ошибка при обучении: {e}")
                                st.session_state['df'] = df
                                st.session_state['current_file_name'] = uploaded_file.name
                    else:
                        st.info("Файл уже загружен.")
                except Exception as e:
                    st.error(f"Ошибка при чтении файла: {e}")

            if st.button("Сбросить модель/файл"):
                st.session_state['model_trained'] = False
                st.session_state['model'] = None
                st.session_state['report'] = None
                st.session_state['feat_importance'] = None
                st.session_state['df'] = None
                st.session_state['df_raw'] = None
                st.session_state['current_file_name'] = None
                st.success("Сессия сброшена.")

    def route_page(self):
        if self.page == "📊 Дашборд":
            self.create_dashboard(real_df=get_session_df(), real_report=st.session_state.get('report'))
        elif self.page == "📋 История транзакций":
            self.create_transaction_history(real_df=get_session_df())
        elif self.page == "🤖 Анализ модели":
            self.analysis_model()
        elif self.page == "📂 CSV Анализ":
            self.analyze_csv(st.session_state.get('model'), st.session_state.get('report'), st.session_state.get('feat_importance'), get_session_df())
        elif self.page == "🏭 3D Модель":
            self.create_anylogic_page()
        elif self.page == "🌍 Карта транзакций":
            self.create_fraud_map(real_df=get_session_df())
        elif self.page == "📈 Обучение модели":
            self.train_page()
        else:
            st.info("Выберите раздел в сайдбаре.")

    # ----------------------------
    # Dashboard
    # ----------------------------
    def create_dashboard(self, real_df=None, real_report=None):
        st.title("🔍 Дашборд транзакций")
        st.markdown("---")
        if real_df is not None and isinstance(real_df, pd.DataFrame) and not real_df.empty:
            df = real_df.copy()
            df.columns = [c.strip() for c in df.columns]
            total_txns = len(df)
            fraud_txns = int(df['Class'].sum()) if 'Class' in df.columns else 0
            accuracy = None
            recall = None
            if real_report and isinstance(real_report, dict):
                accuracy = real_report.get('accuracy') if isinstance(real_report.get('accuracy'), (int, float)) else real_report.get('accuracy')
                if '1' in real_report:
                    recall = real_report['1'].get('recall')
                elif '1.0' in real_report:
                    recall = real_report['1.0'].get('recall')

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"<div class='metric-card'><h4>✅ Всего транзакций</h4><h2 style='color:#00FFFF'>{total_txns:,}</h2></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><h4>🚨 Мошеннических</h4><h2 style='color:#EF4444'>{fraud_txns}</h2></div>", unsafe_allow_html=True)
            with col3:
                acc_text = f'{accuracy:.1%}' if isinstance(accuracy, (float, int)) else '—'
                st.markdown(f"<div class='metric-card'><h4>📈 Accuracy</h4><h2 style='color:#10B981'>{acc_text}</h2></div>", unsafe_allow_html=True)
            with col4:
                rec_text = f'{recall:.1%}' if isinstance(recall, (float, int)) else '—'
                st.markdown(f"<div class='metric-card'><h4>🎯 Recall (class=1)</h4><h2 style='color:#F59E0B'>{rec_text}</h2></div>", unsafe_allow_html=True)

            st.markdown("---")
            # Plots etc.
            colA, colB = st.columns(2)
            with colA:
                st.subheader("📊 Распределение по Amount")
                if 'Amount' in df.columns:
                    sample_df = df.sample(min(10000, len(df)))
                    try:
                        fig = px.histogram(sample_df, x="Amount", color="Class" if 'Class' in df.columns else None, nbins=100, log_y=True)
                        fig.update_layout(template="plotly_dark", height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Не удалось построить гистограмму Amount: {e}")
                else:
                    st.info("Колонка 'Amount' отсутствует в данных.")
            with colB:
                st.subheader("📊 Распределение по Time")
                if 'Time' in df.columns:
                    try:
                        tser = try_parse_datetime_series(df['Time'])
                        plot_df = pd.DataFrame({'Time': tser, 'Class': df['Class'] if 'Class' in df.columns else 0})
                        fig = px.histogram(plot_df, x="Time", color="Class", nbins=100)
                        fig.update_layout(template="plotly_dark", height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Не удалось построить гистограмму Time: {e}")
                else:
                    st.info("Колонка 'Time' отсутствует в данных.")

            st.subheader("🚨 Примеры мошеннических транзакций")
            if 'Class' in df.columns:
                fraud_data = df[df['Class'] == 1].head(20)
                st.dataframe(fraud_data, use_container_width=True)
            else:
                st.info("Нет метки Class в данных — нельзя показать мошенничества.")
        else:
            st.title("🔍 Демо-Дашборд")
            st.info("Загрузите CSV, чтобы увидеть реальные данные. Показаны демо-данные.")
            transactions = self.demo_transactions
            col1, col2, col3, col4 = st.columns(4)
            total_txns = len(transactions)
            suspicious_txns = len(transactions[transactions['fraud_probability'] > 0.7])
            fraud_txns = int(transactions['Class'].sum())
            accuracy = random.uniform(0.92, 0.98)
            with col1:
                st.markdown(f"<div class='metric-card'><h4>✅ Всего</h4><h2 style='color:#00FFFF'>{total_txns}</h2></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><h4>⚠️ Подозрительных</h4><h2 style='color:#F59E0B'>{suspicious_txns}</h2></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-card'><h4>🚨 Мошенничество</h4><h2 style='color:#EF4444'>{fraud_txns}</h2></div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div class='metric-card'><h4>📈 Accuracy (демо)</h4><h2 style='color:#10B981'>{accuracy:.1%}</h2></div>", unsafe_allow_html=True)

    # Transaction history
    def create_transaction_history(self, real_df=None):
        st.title("📋 История транзакций")
        st.markdown("---")
        df = real_df if real_df is not None else self.demo_transactions
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            st.info("Нет данных. Загрузите CSV во вкладке слева.")
            return
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        col1, col2 = st.columns(2)
        with col1:
            class_filter = st.selectbox("Класс транзакции", ["Все", "Легальные (0)", "Мошенничество (1)"])
        with col2:
            if 'Amount' in df.columns:
                try:
                    min_val = float(df['Amount'].min())
                    max_val = float(df['Amount'].max())
                    slider_max = min(max_val, 5000.0)
                    min_amount, max_amount = st.slider("Сумма (Amount)", min_val, slider_max, (min_val, slider_max), key="real_amount_slider")
                except Exception:
                    min_amount, max_amount = 0.0, 5000.0
            else:
                min_amount, max_amount = 0.0, 1.0

        filtered_df = df.copy()
        if class_filter == "Легальные (0)":
            if 'Class' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Class'] == 0]
        elif class_filter == "Мошенничество (1)":
            if 'Class' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['Class'] == 1]
        if 'Amount' in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df['Amount'] >= min_amount) & (filtered_df['Amount'] <= max_amount)]
        st.dataframe(filtered_df.head(200), use_container_width=True)
        st.info(f"Показано {min(200, len(filtered_df))} из {len(filtered_df)} записей.")

    # Model analysis page
    def analysis_model(self):
        st.title("🤖 Анализ обученной модели")
        st.markdown("---")
        if not st.session_state.get('model_trained'):
            st.warning("Пожалуйста, загрузите CSV и обучите модель.")
            return
        report = st.session_state.get('report')
        feat_imp = st.session_state.get('feat_importance')
        st.subheader("Отчёт о производительности")
        if isinstance(report, dict):
            try:
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4))
            except Exception:
                st.json(report)
        else:
            st.info("Отчёт недоступен.")
        st.subheader("Важность признаков")
        if feat_imp is not None and isinstance(feat_imp, pd.DataFrame) and not feat_imp.empty:
            try:
                fig = px.bar(feat_imp.head(30), x='Importance', y='Feature', orientation='h', title="Топ признаков")
                fig.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'}, height=500)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Ошибка при визуализации важности признаков: {e}")
        else:
            st.info("Важность признаков недоступна.")

    # CSV Analysis
    def analyze_csv(self, model, report, feat_importance, df):
        st.title("📂 CSV Анализ")
        st.markdown("Подробный анализ загружённого CSV.")
        if df is None:
            st.info("Загрузите CSV в сайдбаре.")
            return
        st.success(f"Файл: {st.session_state.get('current_file_name')} (строк: {len(df)})")
        st.subheader("Превью данных")
        st.dataframe(df.head(10))
        st.subheader("Общая статистика")
        try:
            st.dataframe(df.describe().transpose())
        except Exception:
            st.info("Не удалось показать describe().")
        st.subheader("Количество мошенничеств")
        if 'Class' in df.columns:
            fraud_cases = int(df['Class'].sum())
            fraud_percent = (fraud_cases / len(df)) * 100
            st.markdown(f"<div class='metric-card'><h4>Всего мошенничеств: {fraud_cases}</h4><p class='small'>{fraud_percent:.4f}% от всех транзакций</p></div>", unsafe_allow_html=True)
        else:
            st.info("Колонка 'Class' отсутствует.")
        st.subheader("Распределение Amount")
        if 'Amount' in df.columns:
            sample_df = df.sample(min(10000, len(df)))
            fig = px.histogram(sample_df, x="Amount", color="Class" if 'Class' in df.columns else None, nbins=100, title="Amount distribution", log_y=True)
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Колонка 'Amount' отсутствует.")

    # AnyLogic page
    def create_anylogic_page(self):
        st.title("🏭 3D модель (AnyLogic)")
        st.markdown("Встроенная модель AnyLogic Cloud.")
        anylogic_model_url = "https://cloud.anylogic.com/model/53d94b35-c5c3-4b07-8672-214851504a84"
        components.iframe(anylogic_model_url, height=650, scrolling=True)
        st.info("Если модель не отображается — проверьте доступность AnyLogic Cloud.")

    # Map page
    def create_fraud_map(self, real_df=None):
        st.title("🌍 Карта мошеннических транзакций")
        st.markdown("Географическая визуализация мошенничества по странам (или координатам, если доступны).")
        df = real_df if real_df is not None else self.demo_transactions
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            st.info("Нет данных для построения карты. Загрузите CSV.")
            return
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        if 'Class' not in df.columns:
            st.warning("В CSV отсутствует колонка 'Class'. Нельзя показать карту мошенничеств по Class.")
            return
        if 'Country' not in df.columns and 'country' in df.columns:
            df['Country'] = df['country']
        if 'Country' not in df.columns:
            st.warning("Добавьте колонку 'Country' в CSV (ISO коды типа KZ, RU, US или полные названия).")
            return
        fraud_df = df[df['Class'] == 1].copy()
        if fraud_df.empty:
            st.info("В данных нет пометок мошенничества (Class==1).")
            return
        country_coords = {
            'KZ': [48.0196, 66.9237], 'RU': [61.5240, 105.3188], 'US': [37.0902, -95.7129],
            'CN': [35.8617, 104.1954], 'DE': [51.1657, 10.4515], 'GB': [55.3781, -3.4360],
            'FR': [46.2276, 2.2137], 'IN': [20.5937, 78.9629], 'BR': [-14.2350, -51.9253],
            'PK': [30.3753, 69.3451]
        }
        fraud_df['lat'] = fraud_df['Country'].map(lambda c: country_coords.get(c, [None, None])[0])
        fraud_df['lon'] = fraud_df['Country'].map(lambda c: country_coords.get(c, [None, None])[1])
        map_df = fraud_df.dropna(subset=['lat', 'lon']).copy()
        if map_df.empty:
            st.warning("Не удалось найти координаты для стран в данных. Добавьте mapping координат.")
            st.dataframe(fraud_df['Country'].value_counts().reset_index().rename(columns={'index':'Country','Country':'Count'}))
            return
        agg = map_df.groupby(['Country','lat','lon']).agg({'id':'count'}).reset_index().rename(columns={'id':'FraudCount'})
        fig = px.scatter_mapbox(agg, lat="lat", lon="lon", size="FraudCount", hover_name="Country", hover_data={"FraudCount":True}, zoom=1, size_max=40)
        fig.update_layout(mapbox_style="open-street-map", template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Таблица: мошенничества по странам")
        st.dataframe(agg.sort_values('FraudCount', ascending=False))


    # ----------------------------
    # New: Training page (tab)
    # ----------------------------
    def train_page(self):
        st.title("📈 Обучение модели")
        st.markdown("Просмотр и запуск обучения модели. (Автоматическое обучение уже происходит при загрузке CSV.)")
        df = get_session_df()
        if df is None:
            st.info("Загрузите CSV в сайдбаре, чтобы начать обучение.")
            return

        st.subheader("Текущий файл")
        st.write(st.session_state.get('current_file_name', '—'))
        st.subheader("Параметры обучения")
        col1, col2 = st.columns(2)
        with col1:
            supervised_exists = 'Class' in df.columns
            st.text(f"Class column present: {supervised_exists}")
            contamination = st.slider("Контаминация (если unsupervised)", min_value=0.001, max_value=0.5, value=0.05, step=0.001)
        with col2:
            save_path = st.text_input("Путь для сохранения модели", value=st.session_state.get('model_file','fraud_model.pkl'))

        st.markdown("---")
        if st.button("🔁 Перетренировать модель (по текущему CSV)"):
            with st.spinner("Запуск обучения..."):
                try:
                    if 'Class' in df.columns:
                        model, report, fi, cleaned = train_supervised(df, target_col='Class')
                        st.session_state['model_type'] = 'supervised'
                    else:
                        model, report, fi, cleaned = train_unsupervised(df, contamination=contamination)
                        st.session_state['model_type'] = 'unsupervised'
                    st.session_state['model'] = model
                    st.session_state['report'] = report
                    st.session_state['feat_importance'] = fi
                    st.session_state['df'] = cleaned
                    st.session_state['model_trained'] = True
                    st.success("✅ Обучение завершено.")
                    try:
                        joblib.dump(model, save_path)
                        st.session_state['model_file'] = save_path
                        st.success(f"Модель сохранена: {save_path}")
                    except Exception as e:
                        st.warning(f"Не удалось сохранить модель: {e}")
                except Exception as e:
                    st.error(f"Ошибка при обучении: {e}")

        st.markdown("---")
        if st.session_state.get('model_trained'):
            st.subheader("Результаты последнего обучения")
            st.write("Тип модели:", st.session_state.get('model_type'))
            report = st.session_state.get('report') or {}
            if st.session_state.get('model_type') == 'supervised':
                st.subheader("Отчёт (classification_report)")
                try:
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(4))
                except Exception:
                    st.json(report)
                fi = st.session_state.get('feat_importance') or pd.DataFrame()
                if not fi.empty:
                    st.subheader("Топ признаков")
                    st.dataframe(fi.head(20))
            else:
                st.subheader("Unsupervised report")
                st.json(report)
                if 'anomaly_score' in st.session_state.get('df', pd.DataFrame()).columns:
                    st.subheader("Распределение anomaly_score")
                    try:
                        tmp = st.session_state['df'][['anomaly_score']].dropna()
                        fig = px.histogram(tmp, x='anomaly_score', nbins=60, title="anomaly_score distribution")
                        fig.update_layout(template="plotly_dark", height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass
            st.markdown("---")
            if st.button("Скачать модель (joblib)"):
                path = st.session_state.get('model_file','fraud_model.pkl')
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                    st.download_button("Скачать файл модели", data, file_name=os.path.basename(path))
                except Exception as e:
                    st.warning(f"Модель не найдена в {path}: {e}")

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    FraudDetectionDashboard()
