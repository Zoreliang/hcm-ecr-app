# ECR_app.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ===================== UI / THEME =====================
st.set_page_config(
    page_title="3-Year Risk Prediction of Adverse Cardiovascular Events in HCM",
    layout="wide"
)
st.markdown("""
<style>
.main { background-color:#ffffff; font-family:'Arial', sans-serif; }
h1, h2, h3 { color:#1f2937; }
.block-container { padding-top: 1.1rem; padding-bottom: 1.6rem; }
/* 单行省略，不再换行 */
.inline-label { 
  font-size:14px; color:#374151; margin-top:.45rem;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.num-input > div > div > input {
  background-color:#F5F9FF !important;
  border:1px solid #D7E3FF !important;
  border-radius:10px !important;
}
hr { margin: .6rem 0 1.1rem 0; }
.result-box {
  background:#EEF8EE; border:1px solid #BFE7BF;
  border-radius:10px; padding:.75rem .9rem;
  color:#0f5132; font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ===================== PATHS =====================
PATHS = {
    "ECR": {
        "train": r"D:\WXKTJY\HCM MACES\data analysis\model construction\ECR\training set1.xlsx",
        "best":  {"hidden":[64], "dropout":0.3205834974859022, "lr":0.0016603175384760594}
    }
}

# ===================== DeepSurv deps =====================
import random
import torch
import torchtuples as tt
from pycox.models import CoxPH as PyCoxPH

RANDOM_STATE = 2025
YEARS = [3.0]  # only 3-year

def _set_seed(seed: int = RANDOM_STATE):
    """尽量保证可复现：Python/Numpy/Torch 随机性固定 & 关闭 cudnn 的随机性。"""
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 尽量走确定性算法（某些环境可能不支持，忽略异常）
        try:
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:
        pass

# ===================== Data IO (consistent with offline) =====================
def read_ecr_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training file not found: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    if "number" in df.columns:
        df = df.drop(columns=["number"])
    ren = {}
    if "Follow-up Time" in df.columns: ren["Follow-up Time"] = "time"
    if "label" in df.columns: ren["label"] = "event"
    df = df.rename(columns=ren)
    if ("time" not in df.columns) or ("event" not in df.columns):
        raise ValueError("Dataset must contain 'Follow-up Time' and 'label' (renamed to 'time' and 'event').")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    df["event"] = pd.to_numeric(df["event"], errors="coerce").astype(int).astype(bool)
    df["time"]  = pd.to_numeric(df["time"],  errors="coerce").astype(float)
    X = df.drop(columns=["time", "event"]).apply(pd.to_numeric, errors="coerce")
    return X, df["time"].values.astype("float32"), df["event"].values.astype("bool"), df

def fit_ecr_deepsurv(X: pd.DataFrame, time: np.ndarray, event: np.ndarray,
                     hidden=[64], dropout=0.32, lr=0.00166):
    """训练 DeepSurv；训练完强制 eval 并计算基线风险，避免推理时受 dropout 影响。"""
    from sklearn.preprocessing import StandardScaler
    _set_seed(RANDOM_STATE)
    scaler = StandardScaler().fit(X.values.astype("float32"))
    Xs = scaler.transform(X.values.astype("float32"))

    net = tt.practical.MLPVanilla(X.shape[1], hidden, 1, torch.nn.ReLU, dropout)
    model = PyCoxPH(net, tt.optim.Adam(lr=lr))

    dtrain = (Xs, (time.astype("float32"), event.astype("float32")))
    model.fit(dtrain[0], dtrain[1],
              batch_size=64, epochs=200, verbose=False,
              callbacks=[tt.callbacks.EarlyStopping(patience=20)],
              val_data=dtrain)

    # 🔒 推理固定 eval，避免 dropout/BN 带来的不稳定
    model.net.eval()
    with torch.no_grad():
        model.compute_baseline_hazards()

    return {"net": model, "scaler": scaler, "feat_cols": X.columns.tolist()}

def predict_surv_df(model_pack, X_df: pd.DataFrame):
    """推理前再次确保 eval，并在 no_grad 下预测，保证稳定"""
    scaler = model_pack["scaler"]; net = model_pack["net"]
    X_aligned = X_df[model_pack["feat_cols"]].copy()
    Xs = scaler.transform(X_aligned.values.astype("float32"))
    with torch.no_grad():
        net.net.eval()
        sdf = net.predict_surv_df(Xs.astype("float32"))
    return sdf

def risk_3y(surv_df):
    t = 36.0  # months
    S = np.array([np.interp(t, surv_df.index.values, surv_df.iloc[:, i].values)
                  for i in range(surv_df.shape[1])])
    return 1.0 - S

# ===================== Cache =====================
@st.cache_data(show_spinner=True)
def load_training():
    return read_ecr_dataset(PATHS["ECR"]["train"])

@st.cache_resource(show_spinner=True)
def train_model_cached(X, time, event, params):
    return fit_ecr_deepsurv(X, time, event, **params)

# ===================== Load & Train =====================
try:
    Xtr, ttr, etr, df_tr = load_training()
except Exception as e:
    st.error(f"Failed to load training data: {e}")
    st.stop()

model_pack = train_model_cached(Xtr, ttr, etr, PATHS["ECR"]["best"])

# ===================== Header =====================
st.markdown(
    "<h1 style='text-align:center;'>3-Year Risk Prediction of Adverse Cardiovascular Events in Hypertrophic Cardiomyopathy</h1>",
    unsafe_allow_html=True
)
st.markdown("<hr/>", unsafe_allow_html=True)

# ===================== Single-patient input（左 1/3，右 2/3） =====================
left_col, right_col = st.columns([1, 2])
INT_FEATURES = {"age", "syncope", "atrial fibrillation"}

with st.form("single_input_form"):
    user_vals = {}

    with left_col:
        for col in model_pack["feat_cols"]:
            series = pd.to_numeric(Xtr[col], errors="coerce")
            default_val = float(np.nanmedian(series.values)) if series.notna().any() else 0.0
            name_norm = col.strip().lower()
            is_int = name_norm in INT_FEATURES

            lab_col, inp_col = st.columns([1, 1.4])
            with lab_col:
                st.markdown(f"<div class='inline-label' title='{col}'>{col}</div>", unsafe_allow_html=True)
            with inp_col:
                if is_int:
                    if name_norm in {"syncope", "atrial fibrillation"}:
                        user_vals[col] = st.number_input(
                            label="", key=f"int_{col}", value=int(round(default_val)),
                            step=1, min_value=0, max_value=1, label_visibility="collapsed", help=None
                        )
                    else:
                        user_vals[col] = st.number_input(
                            label="", key=f"int_{col}", value=int(round(default_val)),
                            step=1, label_visibility="collapsed", help=None
                        )
                else:
                    user_vals[col] = st.number_input(
                        label="", key=f"flt_{col}", value=float(round(default_val, 2)),
                        step=0.01, format="%.2f", label_visibility="collapsed", help=None
                    )

    submitted = st.form_submit_button("Predict", use_container_width=False)

# ===================== Predict & Plot（左结果 / 右曲线） =====================
if submitted:
    one = pd.DataFrame([user_vals], columns=model_pack["feat_cols"])
    try:
        sdf = predict_surv_df(model_pack, one)
        p3 = float(risk_3y(sdf)[0])

        with left_col:
            st.markdown(f"<div class='result-box'>Predicted 3-year risk: <b>{p3*100:.1f}%</b></div>",
                        unsafe_allow_html=True)

        with right_col:
            mask = (sdf.index >= 0) & (sdf.index <= 50)
            if mask.any():
                x = sdf.index[mask]
                y = sdf.iloc[:, 0].values[mask]
            else:
                x = sdf.index
                y = sdf.iloc[:, 0].values

            fig, ax = plt.subplots(figsize=(9.5, 5.3))
            ax.plot(x, y, lw=2.2, label="Predicted survival S(t)")
            ax.axvline(36.0, ls="--", lw=1.1, c="gray", label="36 months")
            ax.set_xlabel("Follow-up time (months)")
            ax.set_ylabel("Survival probability")
            ax.set_ylim(0, 1.02)
            ax.set_xlim(0, 50)
            ax.legend(frameon=False, loc="best")
            st.pyplot(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ===================== Batch inference (optional) =====================
st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Batch inference (optional)")
st.caption("Upload an Excel/CSV containing the same feature columns as the training set (no time/event/number).")

up = st.file_uploader("Upload .xlsx / .csv", type=["xlsx", "csv"])
if up is not None:
    try:
        if up.name.lower().endswith(".xlsx"):
            df_in = pd.read_excel(up)
        else:
            df_in = pd.read_csv(up)

        missing = [c for c in model_pack["feat_cols"] if c not in df_in.columns]
        if missing:
            st.error(f"Missing feature columns: {missing}")
        else:
            X_in = df_in[model_pack["feat_cols"]].copy()
            for c in X_in.columns:
                X_in[c] = pd.to_numeric(X_in[c], errors="coerce")
            if X_in.isna().any().any():
                fill_map = {c: float(np.nanmedian(pd.to_numeric(Xtr[c], errors='coerce'))) for c in X_in.columns}
                X_in = X_in.fillna(fill_map)

            sdf = predict_surv_df(model_pack, X_in)
            p3_all = risk_3y(sdf)
            out = df_in.copy()
            out["Risk_3y"] = np.round(p3_all, 4)
            out["Risk_3y(%)"] = np.round(p3_all*100, 1)

            st.success(f"Computed {len(out)} rows.")
            st.dataframe(out.head(20), use_container_width=True)

            csv = out.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("Download results (.csv)", data=csv, file_name="HCM_ACE_3y_predictions.csv")
    except Exception as e:
        st.error(f"Failed to process the uploaded file: {e}")

st.caption("Risk is computed as 1 − S(36 months).")
