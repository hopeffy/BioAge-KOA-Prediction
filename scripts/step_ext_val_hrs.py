"""
STEP EXT-VAL-HRS: External Validation on RAND HRS Longitudinal File
=====================================================================
Applies the trained CHARLS Random Forest (raw17 / raw16) model to the
RAND HRS 1992-2022 dataset for replication-cohort external validation.

Design decisions
----------------
- Primary feature set: raw16  (raw17 minus Biological Age)
  Reason: RAND fat file does not contain a pre-computed Biological Age.
  The HRS Venous Blood Study biomarkers (separate file) can be used to
  compute KDM-BA, but that requires an additional data linkage step.
  raw16 sensitivity analysis is therefore the PRIMARY HRS analysis.

- KOA proxy outcome:
  arthritis_proxy = r{wave}arthrhr == 1   (physician-diagnosed arthritis)
  pain_proxy      = r{wave}pain == 1      (chronic pain; wave-specific)
  KOA_HRS = arthritis_proxy AND pain_proxy
  Caveat: HRS lacks a knee-specific pain localisation module in the fat
  file; the pain proxy is less specific than CHARLS position_knees.
  This is documented and reported as a limitation.

- Wave selection: wave 13 (2016) by default. Change WAVE constant below.
  Wave 13 overlaps chronologically with CHARLS waves 3-4 (2015-2018).
  Participants must have responded in the selected wave (r{w}iwstat == 1).

- Model: RF retrained on full CHARLS Dataset A internal partition
  (n = 9,863, from step_01_data_prep/dataset_A_internal_train.csv)
  using raw16 columns. Preprocessing (SimpleImputer + StandardScaler)
  fitted on CHARLS train, applied to HRS without re-fitting.

Input files
-----------
  randhrs1992_2022v1_STATA/randhrs1992_2022v1.dta  (RAND HRS fat file)
  step_01_data_prep/dataset_A_internal_train.csv    (CHARLS train)

Output folder
-------------
  step_ext_val_hrs/
    hrs_analysis_cohort.csv
    ext_val_hrs_metrics.csv
    ext_val_hrs_calibration_table.csv
    ext_val_hrs_dca.csv
    ext_val_hrs_report.txt
    reliability_diagram_hrs.png
    dca_curve_hrs.png
"""

import os
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration — change WAVE to use a different HRS survey year
# ---------------------------------------------------------------------------
WAVE = 13           # 13 = 2016  |  12 = 2014  |  11 = 2012  |  10 = 2010
N_TREES = 300
RANDOM_STATE = 42
MIN_AGE = 45        # align with CHARLS eligibility

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HRS_FILE = os.path.join(
    BASE_DIR, "..", "randhrs1992_2022v1_STATA", "randhrs1992_2022v1.dta"
)
CHARLS_TRAIN = os.path.join(BASE_DIR, "step_01_data_prep", "dataset_A_internal_train.csv")
OUT_DIR = os.path.join(BASE_DIR, "step_ext_val_hrs")

# ---------------------------------------------------------------------------
# RAND HRS variable name helpers
# ---------------------------------------------------------------------------

def w(var: str) -> str:
    """Return wave-prefixed variable name, e.g. w('agey_b') -> 'r13agey_b'."""
    return f"r{WAVE}{var}"


# ---------------------------------------------------------------------------
# Column specification (only the columns we need from the 1.7 GB file)
# ---------------------------------------------------------------------------

TIME_INVARIANT = [
    "hhid",          # household ID
    "pn",            # person number
    "ragender",      # gender: 1=male, 2=female
    "raedyrs",       # years of education (time-invariant)
]

WAVE_VARS = [
    w("iwstat"),     # interview status (1 = responded)
    w("agey_b"),     # age at interview
    w("mstat"),      # marital status
    w("hibpe"),      # hypertension ever diagnosed
    w("diabe"),      # diabetes ever diagnosed
    w("cancre"),     # cancer ever diagnosed
    w("hearte"),     # heart disease ever diagnosed
    w("stroke"),     # stroke ever diagnosed
    w("arthr"),      # current-wave arthritis report (RAND: r{w}arthr)
    w("arthre"),     # arthritis ever diagnosed (fallback)
    w("stoopa"),     # any difficulty stooping/kneeling/crouching  (KOA-relevant ADL)
    w("smokev"),     # ever smoked regularly
    w("drinkd"),     # drink at all
    w("bmi"),        # BMI (may be missing in some waves)
]

ALL_COLS = TIME_INVARIANT + WAVE_VARS


# ---------------------------------------------------------------------------
# Load HRS data (column-selection for memory efficiency)
# ---------------------------------------------------------------------------

def load_hrs() -> pd.DataFrame:
    print(f"Loading RAND HRS fat file (wave {WAVE} columns only)...")
    print(f"File: {HRS_FILE}")

    if not os.path.exists(HRS_FILE):
        raise FileNotFoundError(
            f"RAND HRS file not found at:\n  {HRS_FILE}\n"
            "Check that the randhrs1992_2022v1_STATA folder is in the project root."
        )

    # Try pyreadstat first (faster column selection on large STATA files)
    try:
        import pyreadstat  # type: ignore
        df, meta = pyreadstat.read_dta(HRS_FILE, usecols=ALL_COLS)
        print(f"  Read with pyreadstat: {df.shape}")
        return df
    except ImportError:
        print("  pyreadstat not available; falling back to pandas.read_stata (slower)...")
    except Exception as e:
        print(f"  pyreadstat failed ({e}); falling back to pandas...")

    # Fallback: pandas read_stata with columns parameter
    df = pd.read_stata(HRS_FILE, columns=ALL_COLS, convert_categoricals=False)
    print(f"  Read with pandas: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Build analysis cohort
# ---------------------------------------------------------------------------

def build_cohort(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("COHORT CONSTRUCTION")
    print("=" * 60)
    print(f"  Rows in full RAND file: {len(df):,}")

    # 1. Keep only wave respondents
    responded = df[w("iwstat")] == 1
    df = df[responded].copy()
    print(f"  After wave {WAVE} respondents only: {len(df):,}")

    # 2. Age filter (>= MIN_AGE, aligns with CHARLS)
    age_ok = df[w("agey_b")] >= MIN_AGE
    df = df[age_ok].copy()
    print(f"  After age >= {MIN_AGE}: {len(df):,}")

    # 3. KOA proxy outcome
    # Strategy: current-wave arthritis (r{w}arthr) AND difficulty stooping/kneeling (r{w}stoopa)
    # This is more KOA-specific than arthritis-ever alone:
    #   - r{w}arthr: physician-reported arthritis THIS wave (not lifetime)
    #   - r{w}stoopa: any difficulty stooping/kneeling/crouching (knee-relevant ADL)
    # Rationale: OA is a current disease + functional impairment; RA does not primarily limit kneeling

    # Current arthritis (prefer r{w}arthr; fallback to r{w}arthre if needed)
    if w("arthr") in df.columns and df[w("arthr")].notna().sum() > 100:
        arthr_col = w("arthr")
    else:
        arthr_col = w("arthre")
        print(f"  Using arthritis-ever ({arthr_col}) as fallback.")
    arthritis = (df[arthr_col] == 1).astype(int)

    # Knee-relevant functional limitation (stooping/kneeling/crouching)
    if w("stoopa") in df.columns and df[w("stoopa")].notna().sum() > 100:
        stoop_limitation = (df[w("stoopa")] == 1).astype(int)
        proxy_desc = "current arthritis + difficulty stooping/kneeling"
    else:
        # Final fallback: arthritis-only (broad, expect inflated prevalence)
        stoop_limitation = pd.Series(1, index=df.index)
        proxy_desc = "current arthritis only (stoopa not available)"
        print(f"  WARNING: {w('stoopa')} not available; arthritis-only proxy used.")

    koa = (arthritis == 1) & (stoop_limitation == 1)
    df["KOA_HRS"] = koa.astype(int)

    # 4. Drop rows where KOA cannot be determined (arthritis missing)
    arthritis_missing = df[arthr_col].isna()
    df = df[~arthritis_missing].copy()
    print(f"  After dropping missing arthritis: {len(df):,}")
    print(f"  KOA proxy: {proxy_desc}")
    print(f"  KOA proxy prevalence: {df['KOA_HRS'].mean()*100:.1f}%  "
          f"(n_pos={df['KOA_HRS'].sum():,})")
    print(f"\n  NOTE: KOA proxy is less specific than CHARLS (arthritis + knee-specific pain).")
    print(f"  Outcome-definition noise is a documented cross-cultural limitation.")

    return df


# ---------------------------------------------------------------------------
# Feature engineering — map RAND HRS → raw16
# ---------------------------------------------------------------------------

RAW16 = [
    # wave + time identifiers
    "wave_id",
    # demographics
    "Gender", "Age_New", "Marital", "Education", "Residence_proxy",
    # comorbidities
    "Hypertension", "Dyslipidemia_proxy", "Diabetes", "Cancer", "CVD",
    # lifestyle
    "Smoke", "Drink",
    # physical
    "BMI", "BMI_New",
    # NOTE: Biological Age EXCLUDED (not available in fat file)
    # Add "Biological_Age" here when VBS linkage is done
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Wave identifier (analogous to CHARLS 'wave' and 'Time')
    out["wave_id"] = WAVE

    # Gender: RAND 1=male, 2=female  →  same encoding as CHARLS
    out["Gender"] = df["ragender"].map({1: 1, 2: 2})

    # Age bucket: 1 = 45-59, 2 = ≥ 60
    out["Age_New"] = (df[w("agey_b")] >= 60).astype(int) + 1

    # Marital: 1=married/partnered, 2=other
    # RAND r{w}mstat: 1=married, 2=married absent, 3=partnered, 4=separated,
    #                 5=divorced, 6=widowed, 7=never married
    married_codes = {1, 2, 3}
    out["Marital"] = df[w("mstat")].apply(
        lambda x: 1 if x in married_codes else 2
    )

    # Education: years → low/medium/high
    # <9 years → 1 (low), 9-12 → 2 (medium), >12 → 3 (high)
    def edu_bucket(yrs):
        if pd.isna(yrs):
            return np.nan
        if yrs < 9:
            return 1
        if yrs <= 12:
            return 2
        return 3
    out["Education"] = df["raedyrs"].apply(edu_bucket)

    # Residence: urban/rural — RAND HRS doesn't have this in fat file.
    # Use a constant proxy (US is ~80% urban) and flag as approximation.
    # Set to 1 (urban) for all; sensitivity analysis excludes this feature.
    out["Residence_proxy"] = 1
    print("\n  NOTE: Residence set to urban (1) for all HRS participants.")
    print("  US rural/urban classification requires geographic restricted file.")
    print("  Consider excluding Residence in sensitivity analysis.")

    # Hypertension (0/1)
    out["Hypertension"] = (df[w("hibpe")] == 1).astype(float)
    out.loc[df[w("hibpe")].isna(), "Hypertension"] = np.nan

    # Dyslipidemia proxy: HRS has cholesterol medication item in some waves
    # Use r{w}cholst (1=takes cholesterol-lowering meds) as proxy
    cholst_col = w("cholst")
    if cholst_col in df.columns:
        out["Dyslipidemia_proxy"] = (df[cholst_col] == 1).astype(float)
        out.loc[df[cholst_col].isna(), "Dyslipidemia_proxy"] = np.nan
    else:
        print(f"\n  NOTE: {cholst_col} not found; Dyslipidemia_proxy set to NaN.")
        out["Dyslipidemia_proxy"] = np.nan

    # Diabetes (0/1)
    out["Diabetes"] = (df[w("diabe")] == 1).astype(float)
    out.loc[df[w("diabe")].isna(), "Diabetes"] = np.nan

    # Cancer (0/1)
    out["Cancer"] = (df[w("cancre")] == 1).astype(float)
    out.loc[df[w("cancre")].isna(), "Cancer"] = np.nan

    # CVD: composite of heart disease OR stroke
    heart = (df[w("hearte")] == 1)
    stroke = (df[w("stroke")] == 1)
    cvd_missing = df[w("hearte")].isna() & df[w("stroke")].isna()
    out["CVD"] = (heart | stroke).astype(float)
    out.loc[cvd_missing, "CVD"] = np.nan

    # Smoke: ever smoked (1) vs never (0)
    out["Smoke"] = (df[w("smokev")] == 1).astype(float)
    out.loc[df[w("smokev")].isna(), "Smoke"] = np.nan

    # Drink: drink at all (1) vs abstainer (0)
    out["Drink"] = (df[w("drinkd")] == 1).astype(float)
    out.loc[df[w("drinkd")].isna(), "Drink"] = np.nan

    # BMI (continuous)
    out["BMI"] = pd.to_numeric(df[w("bmi")], errors="coerce")

    # BMI category: 1=<25, 2=25-29.9, 3=≥30
    def bmi_cat(b):
        if pd.isna(b):
            return np.nan
        if b < 25:
            return 1
        if b < 30:
            return 2
        return 3
    out["BMI_New"] = out["BMI"].apply(bmi_cat)

    out["KOA"] = df["KOA_HRS"].values

    # Drop rows where KOA target is NaN (shouldn't happen, but defensive)
    out = out[out["KOA"].notna()].copy()

    return out


# ---------------------------------------------------------------------------
# Train CHARLS model (raw16) and apply to HRS
# ---------------------------------------------------------------------------

def load_charls_train() -> pd.DataFrame:
    if not os.path.exists(CHARLS_TRAIN):
        raise FileNotFoundError(
            f"CHARLS training data not found at: {CHARLS_TRAIN}\n"
            "Run step_01_data_prep.py first."
        )
    return pd.read_csv(CHARLS_TRAIN)


def get_feature_cols(charls_df: pd.DataFrame, hrs_df: pd.DataFrame) -> list:
    """Return the intersection of available raw16 columns in both datasets."""
    charls_available = [c for c in RAW16 if c in charls_df.columns]
    hrs_available    = [c for c in RAW16 if c in hrs_df.columns]
    common = [c for c in RAW16 if c in charls_available and c in hrs_available]
    missing_charls = [c for c in RAW16 if c not in charls_df.columns]
    missing_hrs    = [c for c in RAW16 if c not in hrs_df.columns]
    if missing_charls:
        print(f"\n  Columns missing in CHARLS: {missing_charls}")
    if missing_hrs:
        print(f"  Columns missing in HRS: {missing_hrs}")
    print(f"  Using {len(common)} shared feature columns: {common}")
    return common


def preprocess(X_train: np.ndarray, X_ext: np.ndarray):
    """Fit imputer+scaler on CHARLS train, apply to external (no re-fitting)."""
    num_imputer = SimpleImputer(strategy="median")
    X_train_imp = num_imputer.fit_transform(X_train)
    X_ext_imp   = num_imputer.transform(X_ext)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_ext_sc   = scaler.transform(X_ext_imp)

    return X_train_sc, X_ext_sc


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def calibration_slope_intercept(y_true, proba, eps=1e-6):
    p = np.clip(proba, eps, 1.0 - eps)
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(logit_p, y_true)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan


def expected_calibration_error(y_true, proba, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(proba, bins[1:-1], right=True)
    ece, mce, n = 0.0, 0.0, len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        obs  = y_true[mask].mean()
        pred = proba[mask].mean()
        gap  = abs(obs - pred)
        ece += (mask.sum() / n) * gap
        mce = max(mce, gap)
    return ece, mce


def calibration_table(y_true, proba, n_bins=10):
    df = pd.DataFrame({"y": y_true, "p": proba})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    tbl = (
        df.groupby("bin", observed=False)
        .agg(n=("y", "size"), mean_pred=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    tbl["abs_gap"] = (tbl["observed_rate"] - tbl["mean_pred"]).abs()
    return tbl


def wilson_ci(rate, n, z=1.96):
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + z**2 / n
    centre = (rate + z**2 / (2 * n)) / denom
    half   = (z / denom) * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))
    return max(0.0, centre - half), min(1.0, centre + half)


def decision_curve(y_true, proba, thresholds):
    n = len(y_true)
    prev = y_true.mean()
    rows = []
    for t in thresholds:
        pred_pos = proba >= t
        tp = ((pred_pos == 1) & (y_true == 1)).sum()
        fp = ((pred_pos == 1) & (y_true == 0)).sum()
        odds = t / (1.0 - t)
        nb_model = (tp / n) - (fp / n) * odds
        nb_all   = prev - (1.0 - prev) * odds
        rows.append({"threshold": t, "net_benefit_model": nb_model,
                     "net_benefit_all": nb_all, "net_benefit_none": 0.0})
    return pd.DataFrame(rows)


def evaluate(y_true, proba, label="HRS"):
    roc_auc  = roc_auc_score(y_true, proba)
    pr_auc   = average_precision_score(y_true, proba)
    brier    = brier_score_loss(y_true, proba)
    ece, mce = expected_calibration_error(y_true, proba)
    slope, intercept = calibration_slope_intercept(y_true, proba)

    print(f"\n  [{label}]")
    print(f"  ROC-AUC  = {roc_auc:.4f}")
    print(f"  PR-AUC   = {pr_auc:.4f}")
    print(f"  Brier    = {brier:.4f}")
    print(f"  ECE      = {ece:.4f}")
    print(f"  MCE      = {mce:.4f}")
    print(f"  Cal slope= {slope:.3f}  (acceptable: 0.80-1.20)")
    print(f"  Cal int  = {intercept:.3f}")
    if slope < 0.80 or slope > 1.20:
        print(f"  WARNING: slope outside [0.80, 1.20] -> logistic recalibration recommended")

    return {
        "roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier,
        "ece": ece, "mce": mce,
        "calibration_slope": slope, "calibration_intercept": intercept,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reliability(y_true, proba, title, out_path):
    cal_tbl = calibration_table(y_true, proba)
    xs = cal_tbl["mean_pred"].values
    ys = cal_tbl["observed_rate"].values
    ns = cal_tbl["n"].values
    lows  = np.array([wilson_ci(y, int(n))[0] for y, n in zip(ys, ns)])
    highs = np.array([wilson_ci(y, int(n))[1] for y, n in zip(ys, ns)])
    yerr  = np.vstack([ys - lows, highs - ys])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.8,
                markersize=6, capsize=3, label="HRS (raw16)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (decile)")
    ax.set_ylabel("Observed KOA rate (decile)")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_dca(dca_df, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dca_df["threshold"], dca_df["net_benefit_model"],
            linewidth=2, label="Model (raw16)")
    ax.plot(dca_df["threshold"], dca_df["net_benefit_all"],
            "--", linewidth=1.5, label="Treat all")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.5, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP EXT-VAL-HRS: RAND HRS External Validation (raw16 feature set)")
    print("=" * 70)
    print(f"Date:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Wave:  {WAVE}  (change WAVE constant to switch year)")
    print(f"Model: Random Forest, n_estimators={N_TREES}, seed={RANDOM_STATE}")
    print(f"Feat:  raw16 (raw17 minus Biological Age)")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load HRS ──────────────────────────────────────────────────────
    hrs_raw = load_hrs()

    # Check which wave pain column is available; add to wave list if missing
    pain_col = w("pain")
    if pain_col not in hrs_raw.columns:
        print(f"  {pain_col} not in dataset; trying r{WAVE}painr...")
        alt = f"r{WAVE}painr"
        if alt in hrs_raw.columns:
            hrs_raw[pain_col] = hrs_raw[alt]
        else:
            print(f"  Pain column not found; arthritis-only KOA proxy will be used.")
            hrs_raw[pain_col] = np.nan

    # ── 2. Build cohort ───────────────────────────────────────────────────
    hrs_cohort = build_cohort(hrs_raw)

    # ── 3. Feature engineering ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    hrs_features = build_features(hrs_cohort)
    hrs_features.to_csv(
        os.path.join(OUT_DIR, "hrs_analysis_cohort.csv"), index=False
    )
    print(f"\n  Cohort saved ({len(hrs_features):,} rows): hrs_analysis_cohort.csv")

    # Descriptive
    print(f"\n  Final cohort:  n = {len(hrs_features):,}")
    print(f"  KOA_HRS prevalence: {hrs_features['KOA'].mean()*100:.1f}%")
    for col in RAW16:
        if col in hrs_features.columns:
            miss = hrs_features[col].isna().mean() * 100
            if miss > 0:
                print(f"  Missing {col}: {miss:.1f}%")

    # ── 4. Load CHARLS train ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CHARLS MODEL TRAINING (raw16)")
    print("=" * 60)
    charls_df = load_charls_train()
    print(f"  CHARLS train: n = {len(charls_df):,}, "
          f"KOA prev = {charls_df['KOA'].mean()*100:.1f}%")

    # Align feature columns
    feature_cols = get_feature_cols(charls_df, hrs_features)
    if len(feature_cols) < 5:
        raise RuntimeError(
            "Fewer than 5 shared feature columns found. "
            "Check column naming between CHARLS and HRS feature tables."
        )

    X_charls = charls_df[feature_cols].values.astype(float)
    y_charls = charls_df["KOA"].values

    X_hrs = hrs_features[feature_cols].values.astype(float)
    y_hrs = hrs_features["KOA"].values

    X_charls_sc, X_hrs_sc = preprocess(X_charls, X_hrs)

    # Train RF
    rf = RandomForestClassifier(
        n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_charls_sc, y_charls)
    print(f"  RF trained on CHARLS (n={len(y_charls):,})")

    # ── 5. Predict & evaluate ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    proba_hrs = rf.predict_proba(X_hrs_sc)[:, 1]
    metrics   = evaluate(y_hrs, proba_hrs, label="HRS replication (raw16)")

    # DCA
    thresholds = np.arange(0.05, 0.61, 0.01)
    dca_df = decision_curve(y_hrs, proba_hrs, thresholds)
    band   = dca_df[(dca_df["threshold"] >= 0.10) & (dca_df["threshold"] <= 0.30)]
    mean_gain = float((band["net_benefit_model"] - band["net_benefit_all"]).mean())
    avoided   = mean_gain * ((1 - band["threshold"]) / band["threshold"]) * 100
    mean_avoided = float(avoided.mean())
    prev_hrs = float(y_hrs.mean())
    baseline_unnecessary = (1 - prev_hrs) * 100
    rel_reduction = (mean_avoided / baseline_unnecessary * 100) if baseline_unnecessary > 0 else 0

    print(f"\n  DCA (10%-30% band):")
    print(f"    Mean net benefit gain vs treat-all: {mean_gain:.4f}")
    print(f"    Avoided unnecessary interventions/100: {mean_avoided:.1f}")
    print(f"    Relative reduction vs treat-all: {rel_reduction:.1f}%")

    metrics.update({
        "cohort": "HRS_replication",
        "feature_set": "raw16",
        "wave": WAVE,
        "n": len(y_hrs),
        "prevalence": prev_hrs,
        "dca_mean_gain_vs_all_10_30": mean_gain,
        "dca_mean_avoided_per_100": mean_avoided,
        "dca_relative_reduction_pct": rel_reduction,
    })

    # ── 6. Save outputs ───────────────────────────────────────────────────
    pd.DataFrame([metrics]).to_csv(
        os.path.join(OUT_DIR, "ext_val_hrs_metrics.csv"), index=False
    )
    cal_tbl = calibration_table(y_hrs, proba_hrs)
    cal_tbl.to_csv(
        os.path.join(OUT_DIR, "ext_val_hrs_calibration_table.csv"), index=False
    )
    dca_df.to_csv(
        os.path.join(OUT_DIR, "ext_val_hrs_dca.csv"), index=False
    )
    plot_reliability(
        y_hrs, proba_hrs,
        title=f"Reliability Diagram: HRS (wave {WAVE}, raw16)",
        out_path=os.path.join(OUT_DIR, "reliability_diagram_hrs.png"),
    )
    plot_dca(
        dca_df,
        title=f"Decision Curve: HRS (wave {WAVE}, raw16)",
        out_path=os.path.join(OUT_DIR, "dca_curve_hrs.png"),
    )

    # ── 7. Text report ────────────────────────────────────────────────────
    slope = metrics["calibration_slope"]
    slope_flag = (
        "ACCEPTABLE (within [0.80, 1.20])"
        if 0.80 <= slope <= 1.20
        else "OUTSIDE RANGE -> logistic recalibration-in-the-large recommended"
    )

    report_lines = [
        "=" * 70,
        "STEP EXT-VAL-HRS REPORT",
        "=" * 70,
        f"Date:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"HRS wave:     {WAVE}",
        f"Feature set:  raw16 (raw17 minus Biological Age)",
        f"CHARLS train: {CHARLS_TRAIN}",
        "",
        "COHORT",
        "-" * 40,
        f"  n                  = {metrics['n']:,}",
        f"  KOA prevalence     = {metrics['prevalence']*100:.1f}%",
        "",
        "PRIMARY METRICS",
        "-" * 40,
        f"  ROC-AUC            = {metrics['roc_auc']:.4f}",
        f"  PR-AUC             = {metrics['pr_auc']:.4f}",
        f"  Brier score        = {metrics['brier']:.4f}",
        f"  ECE                = {metrics['ece']:.4f}",
        f"  MCE                = {metrics['mce']:.4f}",
        f"  Cal. slope         = {metrics['calibration_slope']:.3f}  [{slope_flag}]",
        f"  Cal. intercept     = {metrics['calibration_intercept']:.3f}",
        "",
        "DCA (10%-30% threshold band)",
        "-" * 40,
        f"  Mean gain vs treat-all = {metrics['dca_mean_gain_vs_all_10_30']:.4f}",
        f"  Avoided per 100 pts    = {metrics['dca_mean_avoided_per_100']:.1f}",
        f"  Relative reduction     = {metrics['dca_relative_reduction_pct']:.1f}%",
        "",
        "COMPARISON WITH CHARLS INTERNAL RESULTS",
        "-" * 40,
        "  Setting              ROC-AUC    PR-AUC    Brier    ECE",
        "  Internal OOF         0.847      0.688     0.070    0.045",
        "  Unseen holdout       0.907      0.791     0.053    0.056",
        f"  HRS (raw16)          {metrics['roc_auc']:.3f}      "
        f"{metrics['pr_auc']:.3f}     {metrics['brier']:.3f}    {metrics['ece']:.3f}",
        "",
        "IMPORTANT LIMITATIONS",
        "-" * 40,
        "  1. Biological Age excluded (not in RAND fat file).",
        "     For full raw17 analysis: link HRS Venous Blood Study (VBS)",
        "     biomarker file and compute KDM-BA via step_03_kdm_ba.py.",
        "  2. KOA proxy (arthritis-only) less specific than CHARLS",
        "     (arthritis + knee-localised pain). Outcome noise expected.",
        "  3. Residence set to urban=1 for all participants.",
        "     True urban/rural requires HRS geographic restricted file.",
        "  4. Dyslipidemia proxy (cholesterol medication) may differ from",
        "     CHARLS self-reported physician diagnosis.",
        "  5. HRS is predominantly white US population; CHARLS is Chinese.",
        "     Cross-cultural performance attenuation is expected.",
        "",
        "NEXT STEPS",
        "-" * 40,
        "  A. Obtain HRS VBS biomarker file -> compute KDM-BA -> run raw17.",
        "  B. Apply logistic recalibration-in-the-large if slope outside [0.80, 1.20].",
        "  C. Proceed to ELSA for primary external validation (full raw17).",
        "  D. Fill TBD placeholders in draft_external_validation.tex.",
        "",
        "OUTPUT FILES",
        "-" * 40,
        f"  {OUT_DIR}/",
        "    hrs_analysis_cohort.csv",
        "    ext_val_hrs_metrics.csv",
        "    ext_val_hrs_calibration_table.csv",
        "    ext_val_hrs_dca.csv",
        "    reliability_diagram_hrs.png",
        "    dca_curve_hrs.png",
        "    ext_val_hrs_report.txt  (this file)",
    ]

    report_path = os.path.join(OUT_DIR, "ext_val_hrs_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved: {report_path}")

    print("\n" + "=" * 70)
    print("STEP EXT-VAL-HRS COMPLETED")
    print("=" * 70)
    print(f"  ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"  Cal slope: {metrics['calibration_slope']:.3f}  [{slope_flag}]")
    print(f"  Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
