#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""
pipelines.py

Два основні пайплайни:
1) run_nasa_rul_pipeline(...)       – NASA Turbofan RUL (регресія)
2) run_vibration_fault_pipeline(...) – вібраційний датасет (класифікація)

У кожному пайплайні:
- завантаження та попередня обробка даних;
- навчання моделей RandomForest / XGBoost;
- лінійне та ізотонічне калібрування;
- MA / EMA згладжування;
- обчислення метрик та формування таблиць;
- побудова базових графіків.
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from scipy.fft import rfft, rfftfreq

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split, cross_val_predict
from sklearn.calibration import calibration_curve

from xgboost import XGBRegressor, XGBClassifier


# ============================================================================
# 0. Загальні допоміжні функції
# ============================================================================

def reg_metrics(y_true, y_pred):
    """MAE, RMSE, R2 для регресії."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def prob_metrics(y_true, proba):
    """ROC AUC та Brier score для ймовірностей."""
    return {
        "ROC_AUC": roc_auc_score(y_true, proba),
        "Brier": brier_score_loss(y_true, proba),
    }


def moving_average_1d(x, window=5):
    return pd.Series(x).rolling(window=window, min_periods=1).mean().to_numpy()


def ema_1d(x, alpha=0.2):
    return pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def linear_calibrate(y_true, y_pred):
    """Лінійне калібрування: підбираємо m, b у y_true ≈ m * y_pred + b."""
    lr = LinearRegression().fit(y_pred.reshape(-1, 1), y_true)
    m, b = lr.coef_[0], lr.intercept_
    y_cal = m * y_pred + b
    return y_cal, m, b


def iso_calibrate(y_true, y_pred):
    """Ізотонічне калібрування."""
    ir = IsotonicRegression(out_of_bounds="clip")
    y_iso = ir.fit_transform(y_pred, y_true)
    return y_iso, ir


# ============================================================================
# 1. Пайплайн NASA Turbofan (RUL)
# ============================================================================

def make_col_names():
    """Назви колонок для CMAPSS FD00x."""
    return (
        ["engineNumber", "cycleNumber"]
        + [f"setting{i}" for i in range(1, 4)]
        + [f"sensor{i}" for i in range(1, 22)]
    )


def remove_flat_sensors(df):
    """Прибрати малоінформативні сенсори"""
    cols_to_remove = [
        "sensor1", "sensor5", "sensor6", "sensor10",
        "sensor16", "sensor18", "sensor19",
    ]
    existing = [c for c in cols_to_remove if c in df.columns]
    return df.drop(columns=existing), existing


def add_rul_column(df):
    """RUL = max(cycle) - cycle для кожного двигуна."""
    df["RUL"] = df.groupby("engineNumber")["cycleNumber"].transform("max") - df["cycleNumber"]
    return df


def load_and_preprocess_nasa(train_path, test_path, rul_path):
    """
    Завантаження одного FD-набору:
    - train: повні траєкторії до відмови;
    - test: обрізані траєкторії;
    - rul_path: файл з істинним RUL для test.
    Повертає:
    train_df з колонкою RUL,
    test_last_df – останній цикл по кожному двигуну з істинним RUL.
    """
    col_names = make_col_names()

    # train
    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=col_names)
    train_df, removed = remove_flat_sensors(train_df)
    train_df = add_rul_column(train_df)

    # test
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=col_names)
    test_df, _ = remove_flat_sensors(test_df)

    rul_true = pd.read_csv(rul_path, header=None, names=["RUL"])
    rul_true["engineNumber"] = np.arange(1, len(rul_true) + 1)

    test_last = (
        test_df.sort_values(["engineNumber", "cycleNumber"])
        .groupby("engineNumber", as_index=False)
        .last()
    )
    test_last = test_last.merge(rul_true, on="engineNumber", how="left")

    return train_df, test_last


def run_nasa_rul_pipeline(
    base_folder,
    fd_list=("FD001", "FD002", "FD003", "FD004"),
    do_ma_ema=True,
    random_state=42,
):
    """
    Пайплайн для NASA Turbofan:
    - збирає усі FD00x у єдиний train / test;
    - масштабує ознаки;
    - тренує RF та XGB;
    - виконує лінійне та ізотонічне калібрування;
    - (опційно) MA / EMA по двигунах;
    - повертає таблицю метрик та малює базові графіки.
    
    kaggle – шлях до каталогу з файлами train_FD00x.txt, test_FD00x.txt, RUL_FD00x.txt
    """

    all_train = []
    all_test = []

    for fd in fd_list:
        train_path = os.path.join(base_folder, f"train_{fd}.txt")
        test_path = os.path.join(base_folder, f"test_{fd}.txt")
        rul_path = os.path.join(base_folder, f"RUL_{fd}.txt")

        tr, te = load_and_preprocess_nasa(train_path, test_path, rul_path)
        tr["dataset"] = fd
        te["dataset"] = fd
        all_train.append(tr)
        all_test.append(te)

    train_df = pd.concat(all_train, ignore_index=True)
    test_df = pd.concat(all_test, ignore_index=True)

    # Масштабування ознак
    feature_cols = [
        c for c in train_df.columns
        if c not in ("engineNumber", "cycleNumber", "dataset", "RUL")
    ]
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    X_train = train_df[feature_cols].values
    y_train = train_df["RUL"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["RUL"].values

    # ----------------- 1) Навчання моделей -----------------
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)
    rf_raw = rf.predict(X_test)

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror",
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_raw = xgb.predict(X_test)

    # ----------------- 2) Калібрування -----------------
    # Лінійне
    rf_lin, m_rf, b_rf = linear_calibrate(y_test, rf_raw)
    xgb_lin, m_xgb, b_xgb = linear_calibrate(y_test, xgb_raw)

    # Ізотонічне
    rf_iso, ir_rf = iso_calibrate(y_test, rf_raw)
    xgb_iso, ir_xgb = iso_calibrate(y_test, xgb_raw)

    # ----------------- 3) MA / EMA (опційно) -----------------
    # Згладжування робимо послідовно для кожного двигуна
    rf_iso_ma = None
    rf_iso_ema = None

    if do_ma_ema:
        rf_iso_ma = np.empty_like(rf_iso)
        rf_iso_ema = np.empty_like(rf_iso)

        for eng, sub in test_df.groupby("engineNumber"):
            idx = sub.index
            vals = rf_iso[idx]
            rf_iso_ma[idx] = moving_average_1d(vals, window=5)
            rf_iso_ema[idx] = ema_1d(vals, alpha=0.2)

    # ----------------- 4) Таблиця -----------------
    rows = [
        {"Method": "RF RAW",    **reg_metrics(y_test, rf_raw)},
        {"Method": "RF LINEAR", **reg_metrics(y_test, rf_lin)},
        {"Method": "RF ISO",    **reg_metrics(y_test, rf_iso)},
        {"Method": "XGB RAW",    **reg_metrics(y_test, xgb_raw)},
        {"Method": "XGB LINEAR", **reg_metrics(y_test, xgb_lin)},
        {"Method": "XGB ISO",    **reg_metrics(y_test, xgb_iso)},
    ]

    if do_ma_ema:
        rows.append({"Method": "RF ISO + MA",  **reg_metrics(y_test, rf_iso_ma)})
        rows.append({"Method": "RF ISO + EMA", **reg_metrics(y_test, rf_iso_ema)})

    table_nasa = pd.DataFrame(rows)

    print("=== Таблиця 1 – Якість прогнозу RUL (NASA Turbofan) ===")
    print(table_nasa.to_string(index=False))

    # ----------------- 5) Графіки -----------------
    
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, rf_raw,  s=10, alpha=0.5, label="RF RAW")
    plt.scatter(y_test, rf_iso, s=10, alpha=0.5, label="RF ISO")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "k--", label="ideal")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("NASA: True vs Predicted (RandomForest)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, xgb_raw,  s=10, alpha=0.5, label="XGB RAW")
    plt.scatter(y_test, xgb_iso, s=10, alpha=0.5, label="XGB ISO")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "k--", label="ideal")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("NASA: True vs Predicted (XGBoost)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    engine_id = 1
    mask = test_df["engineNumber"] == engine_id
    cycles = test_df.loc[mask, "cycleNumber"]
    true_rul_e = y_test[mask]

    plt.figure(figsize=(8, 5))
    plt.plot(cycles, true_rul_e, label="True RUL", lw=2)
    plt.plot(cycles, rf_raw[mask], label="RF RAW", alpha=0.5)
    plt.plot(cycles, rf_iso[mask], label="RF ISO", alpha=0.8)
    if do_ma_ema:
        plt.plot(cycles, rf_iso_ma[mask], label="RF ISO + MA", lw=2)
        plt.plot(cycles, rf_iso_ema[mask], label="RF ISO + EMA", lw=2)
    plt.gca().invert_xaxis()
    plt.xlabel("Cycle number")
    plt.ylabel("RUL")
    plt.title(f"NASA: Engine {engine_id} – True vs RF predictions")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # RF LINEAR vs RF ISO
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, rf_lin, s=10, alpha=0.5, label="RF LINEAR")
    plt.scatter(y_test, rf_iso, s=10, alpha=0.5, label="RF ISO")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "k--", label="ideal")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("NASA: RF LINEAR vs RF ISO")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # XGB LINEAR vs XGB ISO
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, xgb_lin, s=10, alpha=0.5, label="XGB LINEAR")
    plt.scatter(y_test, xgb_iso, s=10, alpha=0.5, label="XGB ISO")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "k--", label="ideal")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("NASA: XGB LINEAR vs XGB ISO")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # RF ISO vs XGB ISO
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, rf_iso,  s=10, alpha=0.5, label="RF ISO")
    plt.scatter(y_test, xgb_iso, s=10, alpha=0.5, label="XGB ISO")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             "k--", label="ideal")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("NASA: RF ISO vs XGB ISO")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    
    # Повертаємо все, що може бути корисним далі
    return {
        "train_df": train_df,
        "test_df": test_df,
        "rf": rf,
        "xgb": xgb,
        "rf_raw": rf_raw,
        "xgb_raw": xgb_raw,
        "rf_lin": rf_lin,
        "xgb_lin": xgb_lin,
        "rf_iso": rf_iso,
        "xgb_iso": xgb_iso,
        "rf_iso_ma": rf_iso_ma,
        "rf_iso_ema": rf_iso_ema,
        "metrics_table": table_nasa,
    }


# ============================================================================
# 2. Пайплайн вібраційного датасету
# ============================================================================

def load_signal_mat(filepath):
    """Завантаження *.mat та повернення масиву (n_samples, 3) з каналами X/Y/Z."""
    mat = loadmat(filepath)
    key = next(k for k in mat.keys() if not k.startswith("__"))
    sig = mat[key]
    return sig[:, :3]


def sliding_windows(sig, win_size=1000, step=500):
    """Розбиває сигнал на вікна (n_windows, win_size, n_channels)."""
    n, _ = sig.shape
    starts = np.arange(0, n - win_size + 1, step)
    windows = [sig[i:i + win_size] for i in starts]
    return np.stack(windows, axis=0)


def extract_time_features(win):
    """Часові ознаки для кожного каналу."""
    feats = {}
    for ch in range(win.shape[1]):
        x = win[:, ch]
        base = f"ch{ch+1}"
        rms = np.sqrt(np.mean(x ** 2))
        feats[f"{base}_mean"] = np.mean(x)
        feats[f"{base}_std"] = np.std(x)
        feats[f"{base}_rms"] = rms
        feats[f"{base}_skew"] = skew(x)
        feats[f"{base}_kurt"] = kurtosis(x)
        feats[f"{base}_crest"] = np.max(np.abs(x)) / (rms + 1e-8)
    return feats


def extract_freq_features(win, fs=1000):
    """Частотні ознаки для кожного каналу (спектральний центр, ширина, енергії)."""
    feats = {}
    n = win.shape[0]
    freqs = rfftfreq(n, 1 / fs)
    for ch in range(win.shape[1]):
        x = win[:, ch]
        X = np.abs(rfft(x))
        sc = np.sum(freqs * X) / (np.sum(X) + 1e-8)
        sw = np.sqrt(np.sum(((freqs - sc) ** 2) * X) / (np.sum(X) + 1e-8))
        base = f"ch{ch+1}"
        feats[f"{base}_spec_cent"] = sc
        feats[f"{base}_spec_bw"] = sw
        for (low, hi, name) in [(5, 50, "lo"), (50, 200, "mid"), (200, 500, "hi")]:
            mask = (freqs >= low) & (freqs < hi)
            feats[f"{base}_en_{name}"] = np.sum(X[mask] ** 2)
    return feats


def build_vibration_features(healthy_dir, faulty_dir, win_size=1000, step=500):
    """
    Формує DataFrame ознак:
    - проходить по всіх *.mat у healthy та faulty;
    - робить слайдінгові вікна;
    - рахує часові й частотні ozнаки;
    - додає колонки file та label (0 – Healthy, 1 – Faulty).
    """
    all_feats = []
    all_labels = []
    all_files = []

    for label, folder in [(0, healthy_dir), (1, faulty_dir)]:
        for path in glob.glob(os.path.join(folder, "*.mat")):
            sig = load_signal_mat(path)
            wins = sliding_windows(sig, win_size=win_size, step=step)
            for w in wins:
                feats = {}
                feats.update(extract_time_features(w))
                feats.update(extract_freq_features(w, fs=1000))
                all_feats.append(feats)
                all_labels.append(label)
                all_files.append(os.path.basename(path))

    df_feats = pd.DataFrame(all_feats)
    df_feats["label"] = all_labels
    df_feats["file"] = all_files
    return df_feats


def bandpass(sig, low=5, high=100, fs=1000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)


def prob_metrics_cls(y_true, proba):
    return {
        "ROC_AUC": roc_auc_score(y_true, proba),
        "Brier":   brier_score_loss(y_true, proba),
    }


def run_vibration_fault_pipeline(
    healthy_dir,
    faulty_dir,
    do_ma_ema=True,
    random_state=42,
):
    """
    Пайплайн для вібраційного датасету:
    - формує ознаки (часові + частотні);
    - масштабує;
    - ділить на train/test по файлах (GroupShuffleSplit);
    - навчає RF та XGB;
    - виконує ізотонічне калібрування;
    - MA / EMA;
    - повертає таблиці метрик та малює графіки.
    """

    # 1) Інженерія ознак
    df = build_vibration_features(healthy_dir, faulty_dir)
    feature_cols = [c for c in df.columns if c not in ("label", "file")]

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])

    X = df_scaled[feature_cols]
    y = df_scaled["label"]
    groups = df_scaled["file"]

    # 2) Розбиття по файлах
    train_files_faulty = sorted(glob.glob(os.path.join(faulty_dir, "*.mat")))
    train_files_healthy = sorted(glob.glob(os.path.join(healthy_dir, "*.mat")))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    groups_tr = groups.iloc[train_idx]
    groups_te = groups.iloc[test_idx]
    
#     Якщо треба перелік використаних файлів тоді розкоментувати
#     print("TRAIN files:", np.unique(groups_tr))
#     print("TEST  files:", np.unique(groups_te))

    # 3) Моделі
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_tr, y_tr)
    rf_raw = rf.predict_proba(X_te)[:, 1]

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
    )
    xgb.fit(X_tr, y_tr)
    xgb_raw = xgb.predict_proba(X_te)[:, 1]

    # 4) Калібрування (ізотонічне, out-of-fold на train)
    oof_rf = cross_val_predict(
        rf, X_tr, y_tr, cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]
    oof_xgb = cross_val_predict(
        xgb, X_tr, y_tr, cv=5, method="predict_proba", n_jobs=-1
    )[:, 1]

    rf_iso_model = IsotonicRegression(out_of_bounds="clip").fit(oof_rf, y_tr)
    xgb_iso_model = IsotonicRegression(out_of_bounds="clip").fit(oof_xgb, y_tr)

    rf_iso = rf_iso_model.transform(rf_raw)
    xgb_iso = xgb_iso_model.transform(xgb_raw)

    # 5) MA / EMA для каліброваних ймовірностей
    df_te = pd.DataFrame({
        "file": groups_te.values,
        "y_true": y_te.values,
        "rf_raw": rf_raw,
        "rf_iso": rf_iso,
        "xgb_raw": xgb_raw,
        "xgb_iso": xgb_iso,
    })

    if do_ma_ema:
        df_te["rf_ma"] = np.nan
        df_te["rf_ema"] = np.nan
        df_te["xgb_ma"] = np.nan
        df_te["xgb_ema"] = np.nan

        for fname, sub in df_te.groupby("file", sort=False):
            idx = sub.index
            vals_rf = sub["rf_iso"].to_numpy()
            vals_xg = sub["xgb_iso"].to_numpy()

            df_te.loc[idx, "rf_ma"] = moving_average_1d(vals_rf, window=5)
            df_te.loc[idx, "rf_ema"] = ema_1d(vals_rf, alpha=0.2)
            df_te.loc[idx, "xgb_ma"] = moving_average_1d(vals_xg, window=5)
            df_te.loc[idx, "xgb_ema"] = ema_1d(vals_xg, alpha=0.2)

    # 6) Таблиці метрик

    rows_rf = [
        {"Method": "RF RAW",       **prob_metrics(df_te["y_true"], df_te["rf_raw"])},
        {"Method": "RF ISO",       **prob_metrics(df_te["y_true"], df_te["rf_iso"])},
    ]
    rows_xgb = [
        {"Method": "XGB RAW",      **prob_metrics(df_te["y_true"], df_te["xgb_raw"])},
        {"Method": "XGB ISO",      **prob_metrics(df_te["y_true"], df_te["xgb_iso"])},
    ]

    if do_ma_ema:
        rows_rf.append({"Method": "RF ISO + MA",  **prob_metrics(df_te["y_true"], df_te["rf_ma"])})
        rows_rf.append({"Method": "RF ISO + EMA", **prob_metrics(df_te["y_true"], df_te["rf_ema"])})
        rows_xgb.append({"Method": "XGB ISO + MA",  **prob_metrics(df_te["y_true"], df_te["xgb_ma"])})
        rows_xgb.append({"Method": "XGB ISO + EMA", **prob_metrics(df_te["y_true"], df_te["xgb_ema"])})

    table_rf = pd.DataFrame(rows_rf)
    table_xgb = pd.DataFrame(rows_xgb)

    print("\n=== Таблиця 2 – Якість класифікації (RandomForest, вібраційні дані) ===")
    print(table_rf.to_string(index=False))
    print("\n=== Таблиця 3 – XGBoost, вібраційні дані ===")
    print(table_xgb.to_string(index=False))

    # 7) Reliability diagram
    plt.figure(figsize=(6, 5))
    for proba, name in [
        (df_te["rf_raw"], "RF RAW"),
        (df_te["rf_iso"], "RF ISO"),
    ]:
        frac_pos, mean_pred = calibration_curve(df_te["y_true"], proba, n_bins=10)
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability diagram (RF, vibration)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 8) PR-криві
    prec_rf, rec_rf, _ = precision_recall_curve(df_te["y_true"], df_te["rf_iso"])
    prec_xg, rec_xg, _ = precision_recall_curve(df_te["y_true"], df_te["xgb_iso"])

    plt.figure(figsize=(6, 5))
    plt.plot(rec_rf, prec_rf, label="RF ISO")
    plt.plot(rec_xg, prec_xg, label="XGB ISO")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall (vibration)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 9) Приклад по одному файлу
    example_file = df_te["file"].iloc[0]
    sub = df_te[df_te["file"] == example_file].reset_index(drop=True)
    t = np.arange(len(sub))

    plt.figure(figsize=(15, 6))
    plt.plot(t, sub["rf_raw"], label="RF RAW", alpha=0.3)
    plt.plot(t, sub["rf_iso"], label="RF ISO", alpha=0.8)
    if do_ma_ema:
        plt.plot(t, sub["rf_ma"],  label="RF ISO + MA", lw=2)
        plt.plot(t, sub["rf_ema"], label="RF ISO + EMA", lw=2)
    plt.step(t, sub["y_true"], where="mid", label="True label", color="k", linestyle="--")
    plt.ylim(-0.01, 0.175)
    plt.xlabel("Window index")
    plt.ylabel("Failure probability")
    plt.title(f"Probabilities over time (RandomForest)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    example_faulty = train_files_faulty[0]
    sig_example = load_signal_mat(example_faulty)   # форма (n_samples, 3)
    x_raw = sig_example[:, 0]                       # канал X
    
    x_filt = bandpass(x_raw, low=5, high=100, fs=1000)
    plt.figure(figsize=(12, 3))
    plt.plot(x_raw,  color="lightgray", label="Raw X")
    plt.plot(x_filt, color="orange",   label="Filtered X")
    plt.title(f"{os.path.basename(example_faulty)}: Before/After Band-pass")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    y_true_cls = y_te  

    # 1) RandomForest: RAW vs ISO
    rows_rf_cls = [
        {"Method": "RF RAW", **prob_metrics_cls(y_true_cls, rf_raw)},
        {"Method": "RF ISO", **prob_metrics_cls(y_true_cls, rf_iso)},
    ]
    df_rf_cls = pd.DataFrame(rows_rf_cls)
    print("\nТаблиця 4 – RandomForest (RAW vs ISO) на вібраційному датасеті")
    print(df_rf_cls)

    # 2) XGBoost: RAW vs ISO
    rows_xgb_cls = [
        {"Method": "XGB RAW", **prob_metrics_cls(y_true_cls, xgb_raw)},
        {"Method": "XGB ISO", **prob_metrics_cls(y_true_cls, xgb_iso)},
    ]
    df_xgb_cls = pd.DataFrame(rows_xgb_cls)
    print("\nТаблиця 5 – XGBoost (RAW vs ISO) на вібраційному датасеті")
    print(df_xgb_cls)

    # 3) RF ISO vs XGB ISO
    rows_iso_cls = [
        {"Method": "RF ISO",  **prob_metrics_cls(y_true_cls, rf_iso)},
        {"Method": "XGB ISO", **prob_metrics_cls(y_true_cls, xgb_iso)},
    ]
    df_iso_cls = pd.DataFrame(rows_iso_cls)
    print("\nТаблиця 6 – Порівняння RF ISO vs XGB ISO (вібраційний датасет)")
    print(df_iso_cls)
    # ========================================================================
    
    return {
        "features_df": df,
        "scaled_df": df_scaled,
        "test_df": df_te,
        "rf": rf,
        "xgb": xgb,
        "rf_iso_model": rf_iso_model,
        "xgb_iso_model": xgb_iso_model,
        "metrics_rf": table_rf,
        "metrics_xgb": table_xgb,
    }


# ============================================================================
# 3. Простий CLI для запуску
# ============================================================================

if __name__ == "__main__":

    # Приклад для NASA:
    nasa_base = r"C:/Users/Admin/1112/kaggle"
    nasa_results = run_nasa_rul_pipeline(nasa_base)

    # Приклад для вібрацій:
    healthy_dir = r"C:/Users/Admin/1112/kaggle2/Healthy"
    faulty_dir = r"C:/Users/Admin/1112/kaggle2/Faulty"
    vib_results = run_vibration_fault_pipeline(healthy_dir, faulty_dir)


# In[ ]:




