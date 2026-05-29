"""
STEP EXT-VAL-ELSA: External Validation on English Longitudinal Study of Ageing
===============================================================================
Applies the trained CHARLS Random Forest (raw16) model to the ELSA
Gateway Harmonized file for primary external validation.

Design decisions
----------------
- Primary feature set: raw16 (raw17 minus Biological Age)
  Reason: ELSA public nurse visit files do not contain creatinine, BUN,
  or platelet assays required for KDM-BA computation. This mirrors the
  HRS replication approach and provides a conservative lower bound on
  transportability with full raw17 pending linked biochemistry data.

- KOA proxy outcome:
  KOA_ELSA = (r8arthre == 1) AND (r8stoopa == 1)
  - r8arthre: ever-diagnosed arthritis or rheumatism
  - r8stoopa: any difficulty stooping, kneeling, or crouching
  Caveat: r8arthre is "ever" diagnosed (not current-wave). ELSA does not
  have a current-wave arthritis variable in the harmonized file.
  This is documented and reported as a limitation.

- Wave selection: Wave 8 (2016-2017) by default. Change WAVE constant
  for Wave 9 sensitivity. Wave 8 is the most recent nurse-visit wave
  with complete harmonized variable coverage.

- Model: RF retrained on full CHARLS Dataset A internal partition
  (n = 9,863, from step_01_data_prep/dataset_A_internal_train.csv)
  using raw16 columns. Preprocessing (SimpleImputer + StandardScaler)
  fitted on CHARLS train, applied to ELSA without re-fitting.

- Calibration: Both uncalibrated (raw) and isotonic-calibrated
  predictions reported, following TRIPOD recommendations for
  external validation.

Input files
-----------
  elsa_stata/UKDA-5050-stata/stata/stata13_se/gh_elsa_h.dta
  elsa_stata/UKDA-5050-stata/stata/stata13_se/elsa_geog_urindewr_2011_eul.dta
  step_01_data_prep/dataset_A_internal_train.csv

Output folder
-------------
  step_ext_val_elsa/
    elsa_analysis_cohort.csv
    ext_val_elsa_metrics.csv
    ext_val_elsa_calibration_table_raw.csv
    ext_val_elsa_calibration_table_isotonic.csv
    ext_val_elsa_dca_raw.csv
    ext_val_elsa_dca_isotonic.csv
    ext_val_elsa_report.txt
    reliability_diagram_elsa_raw.png
    reliability_diagram_elsa_isotonic.png
    dca_curve_elsa_raw.png
    dca_curve_elsa_isotonic.png
"""

import os
import warnings
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WAVE = 8  # 8 = 2016-2017  |  9 = 2018-2019 (sensitivity)
WAVE_YEARS = {8: "2016-2017", 9: "2018-2019"}
N_TREES = 300
RANDOM_STATE = 42
MIN_AGE = 45

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ELSA_BASE = os.path.join(BASE_DIR, "..", "elsa_stata", "UKDA-5050-stata", "stata", "stata13_se")
ELSA_HARMONIZED = os.path.join(ELSA_BASE, "gh_elsa_h.dta")
ELSA_RURALURBAN = os.path.join(ELSA_BASE, "elsa_geog_urindewr_2011_eul.dta")
CHARLS_TRAIN = os.path.join(BASE_DIR, "step_01_data_prep", "dataset_A_internal_train.csv")
OUT_DIR = os.path.join(BASE_DIR, "step_ext_val_elsa")


def w(var: str) -> str:
    """Return wave-prefixed variable name, e.g. w('agey') -> 'r8agey'."""
    return f"r{WAVE}{var}"


# ---------------------------------------------------------------------------
# Column specification for the harmonized ELSA file
# ---------------------------------------------------------------------------
TIME_INVARIANT = [
    "idauniq",  # person identifier
    "ragender",  # gender: 1=male, 2=female
    "raedyrs_e",  # years of education (time-invariant)
    "raeduc_e",  # education level (categorical fallback)
]

WAVE_VARS = [
    w("iwstat"),  # interview status
    w("agey"),  # age at interview
    w("mstat"),  # marital status
    w("hibpe"),  # hypertension ever diagnosed
    w("hchole"),  # high cholesterol ever diagnosed (dyslipidemia proxy)
    w("diabe"),  # diabetes ever diagnosed
    w("cancre"),  # cancer ever diagnosed
    w("hearte"),  # heart disease ever diagnosed
    w("stroke"),  # stroke ever diagnosed
    w("arthre"),  # arthritis/rheumatism ever diagnosed
    w("stoopa"),  # any difficulty stooping/kneeling/crouching
    w("smokev"),  # ever smoked regularly
    w("drink"),  # drinks alcohol
    w("mbmi"),  # measured BMI (nurse visit)
]

ALL_COLS = TIME_INVARIANT + WAVE_VARS


# ---------------------------------------------------------------------------
# raw16 feature set (14 variables, no Biological Age)
# ---------------------------------------------------------------------------
RAW16 = [
    "wave_id",
    "Gender",
    "Age_New",
    "Marital",
    "Education",
    "Residence",
    "Hypertension",
    "Dyslipidemia",
    "Diabetes",
    "Cancer",
    "CVD",
    "Smoke",
    "Drink",
    "BMI",
    "BMI_New",
]


# ---------------------------------------------------------------------------
# Load ELSA data
# ---------------------------------------------------------------------------
def load_elsa() -> pd.DataFrame:
    print(f"Loading ELSA Gateway Harmonized file (wave {WAVE} columns only)...")
    print(f"File: {ELSA_HARMONIZED}")

    if not os.path.exists(ELSA_HARMONIZED):
        raise FileNotFoundError(
            f"ELSA harmonized file not found at:\n  {ELSA_HARMONIZED}\n"
            "Check that elsa_stata is in the project parent directory."
        )

    try:
        import pyreadstat

        df, meta = pyreadstat.read_dta(ELSA_HARMONIZED, usecols=ALL_COLS)
        print(f"  Read with pyreadstat: {df.shape}")
        return df
    except ImportError:
        print("  pyreadstat not available; reading full file with pandas...")
    except Exception as e:
        print(f"  pyreadstat failed ({e}); falling back to pandas...")

    df = pd.read_stata(ELSA_HARMONIZED, convert_categoricals=False)
    print(f"  Read with pandas (full file): {df.shape}")
    return df


def load_rural_urban() -> pd.DataFrame:
    """Load the ELSA rural-urban indicator file (2011 census)."""
    print(f"\nLoading rural-urban indicators...")
    print(f"File: {ELSA_RURALURBAN}")

    if not os.path.exists(ELSA_RURALURBAN):
        print("  WARNING: Rural-urban file not found. Residence will be set to NaN.")
        return pd.DataFrame()

    try:
        import pyreadstat

        ru_col = f"w{WAVE}_urindewr_2011"
        df, meta = pyreadstat.read_dta(ELSA_RURALURBAN, usecols=["idauniq", ru_col])
        print(f"  Read with pyreadstat: {df.shape}")
        return df
    except ImportError:
        df = pd.read_stata(ELSA_RURALURBAN, convert_categoricals=False)
    except Exception:
        df = pd.read_stata(ELSA_RURALURBAN, convert_categoricals=False)

    print(f"  Read with pandas: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Build analysis cohort
# ---------------------------------------------------------------------------
def build_cohort(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("COHORT CONSTRUCTION")
    print("=" * 60)
    print(f"  Rows in harmonized file: {len(df):,}")

    # 1. Keep only wave respondents
    iw_col = w("iwstat")
    if iw_col in df.columns:
        responded = df[iw_col] == 1
        df = df[responded].copy()
        print(f"  After wave {WAVE} respondents only: {len(df):,}")
    else:
        print(f"  WARNING: {iw_col} not found; using all rows with non-null age.")

    # 2. Age filter
    age_col = w("agey")
    if age_col not in df.columns:
        raise KeyError(f"Age column '{age_col}' not found in ELSA data.")
    age_ok = df[age_col] >= MIN_AGE
    df = df[age_ok].copy()
    print(f"  After age >= {MIN_AGE}: {len(df):,}")

    # 3. KOA proxy outcome
    # arthritis ever diagnosed
    arthr_col = w("arthre")
    arthritis = (df[arthr_col] == 1).astype(int)

    # knee-relevant functional limitation
    stoop_col = w("stoopa")
    if stoop_col in df.columns and df[stoop_col].notna().sum() > 100:
        stoop_limitation = (df[stoop_col] == 1).astype(int)
        proxy_desc = "ever-diagnosed arthritis + difficulty stooping/kneeling/crouching"
    else:
        stoop_limitation = pd.Series(1, index=df.index)
        proxy_desc = "ever-diagnosed arthritis only (stoopa not available)"
        print(f"  WARNING: {stoop_col} not available; arthritis-only proxy used.")

    koa = (arthritis == 1) & (stoop_limitation == 1)
    df["KOA_ELSA"] = koa.astype(int)

    # 4. Drop rows where KOA cannot be determined
    arthritis_missing = df[arthr_col].isna()
    stoop_missing = df[stoop_col].isna() if stoop_col in df.columns else pd.Series(False, index=df.index)
    drop_mask = arthritis_missing & stoop_missing
    df = df[~drop_mask].copy()
    print(f"  After dropping missing KOA components: {len(df):,}")
    print(f"  KOA proxy: {proxy_desc}")
    print(
        f"  KOA proxy prevalence: {df['KOA_ELSA'].mean() * 100:.1f}%  "
        f"(n_pos={df['KOA_ELSA'].sum():,})"
    )
    print(f"  CHARLS KOA prevalence (reference): 13.3%")
    print(f"  NOTE: Prevalence mismatch (ELSA {df['KOA_ELSA'].mean()*100:.1f}% vs CHARLS 13.3%)")
    print(f"        contributes to expected miscalibration.")
    print(f"\n  NOTE: arthritis is 'ever diagnosed' (not current-wave).")
    print(f"  KOA proxy is less specific than CHARLS (arthritis + knee-specific pain).")

    return df


# ---------------------------------------------------------------------------
# Feature engineering - map ELSA harmonized -> raw16
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame, rural_urban_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Wave identifier
    out["wave_id"] = WAVE

    # Gender: 1=male, 2=female (same encoding as CHARLS)
    out["Gender"] = df["ragender"].map({1: 1, 2: 2})

    # Age bucket: 1 = 45-59, 2 = >= 60
    out["Age_New"] = (df[w("agey")] >= 60).astype(int) + 1

    # Marital: 1=married/partnered, 2=other
    # Gateway: 1=married, 2=married absent, 3=partnered,
    #          4=separated, 5=divorced, 6=widowed, 7=never married
    married_codes = {1, 2, 3}
    out["Marital"] = df[w("mstat")].apply(lambda x: 1 if x in married_codes else 2)

    # Education: years -> low/medium/high
    # Use raedyrs_e (years) if available, else raeduc_e (level)
    edu_yrs_col = "raedyrs_e"
    edu_lvl_col = "raeduc_e"

    def edu_from_years(yrs):
        if pd.isna(yrs):
            return np.nan
        if yrs < 9:
            return 1
        if yrs <= 12:
            return 2
        return 3

    def edu_from_level(lvl):
        # Gateway educ_e: 1=no degree, 2=GED, 3=high school, 4=college, 5=masters+
        if pd.isna(lvl):
            return np.nan
        if lvl <= 1:
            return 1
        if lvl <= 3:
            return 2
        return 3

    if edu_yrs_col in df.columns and df[edu_yrs_col].notna().sum() > 100:
        out["Education"] = df[edu_yrs_col].apply(edu_from_years)
    elif edu_lvl_col in df.columns:
        print(f"\n  NOTE: Using {edu_lvl_col} as education source (years not available).")
        out["Education"] = df[edu_lvl_col].apply(edu_from_level)
    else:
        out["Education"] = np.nan

    # Residence: from rural-urban indicator file
    ru_col = f"w{WAVE}_urindewr_2011"
    if not rural_urban_df.empty and ru_col in rural_urban_df.columns:
        ru_map = rural_urban_df.set_index("idauniq")[ru_col]
        out["Residence"] = df["idauniq"].map(ru_map)
        # Gateway rural/urban: 1=urban, 2=rural. Map to CHARLS: 1=urban, 2=rural
        out["Residence"] = out["Residence"].map({1: 1, 2: 2})
        valid_ru = out["Residence"].notna().sum()
        print(f"\n  Residence: {valid_ru}/{len(out)} have valid rural-urban data.")
    else:
        print(f"\n  NOTE: Rural-urban data not available; Residence set to NaN.")
        out["Residence"] = np.nan

    # Hypertension (0/1)
    out["Hypertension"] = (df[w("hibpe")] == 1).astype(float)
    out.loc[df[w("hibpe")].isna(), "Hypertension"] = np.nan

    # Dyslipidemia: high cholesterol ever diagnosed
    chol_col = w("hchole")
    if chol_col in df.columns:
        out["Dyslipidemia"] = (df[chol_col] == 1).astype(float)
        out.loc[df[chol_col].isna(), "Dyslipidemia"] = np.nan
    else:
        print(f"\n  NOTE: {chol_col} not found; Dyslipidemia set to NaN.")
        out["Dyslipidemia"] = np.nan

    # Diabetes (0/1)
    out["Diabetes"] = (df[w("diabe")] == 1).astype(float)
    out.loc[df[w("diabe")].isna(), "Diabetes"] = np.nan

    # Cancer (0/1)
    out["Cancer"] = (df[w("cancre")] == 1).astype(float)
    out.loc[df[w("cancre")].isna(), "Cancer"] = np.nan

    # CVD: composite of heart disease OR stroke
    heart = df[w("hearte")] == 1
    stroke = df[w("stroke")] == 1
    cvd_missing = df[w("hearte")].isna() & df[w("stroke")].isna()
    out["CVD"] = (heart | stroke).astype(float)
    out.loc[cvd_missing, "CVD"] = np.nan

    # Smoke: ever smoked (1) vs never (0)
    out["Smoke"] = (df[w("smokev")] == 1).astype(float)
    out.loc[df[w("smokev")].isna(), "Smoke"] = np.nan

    # Drink: drinks alcohol (1) vs abstainer (0)
    out["Drink"] = (df[w("drink")] == 1).astype(float)
    out.loc[df[w("drink")].isna(), "Drink"] = np.nan

    # BMI (continuous, measured from nurse visit)
    # Wave 8 has r8mbmi; Wave 9 only has r9mweight (no BMI).
    # If BMI variable is missing, set to NaN and let imputation handle it.
    bmi_col = w("mbmi")
    if bmi_col in df.columns:
        out["BMI"] = pd.to_numeric(df[bmi_col], errors="coerce")
    else:
        print(f"\n  NOTE: {bmi_col} not found; BMI set to NaN (will be imputed from CHARLS).")
        out["BMI"] = np.nan

    # BMI category: 1=<25, 2=25-29.9, 3=>=30
    def bmi_cat(b):
        if pd.isna(b):
            return np.nan
        if b < 25:
            return 1
        if b < 30:
            return 2
        return 3

    out["BMI_New"] = out["BMI"].apply(bmi_cat)

    # Target
    out["KOA"] = df["KOA_ELSA"].values

    # Drop rows where target is NaN
    out = out[out["KOA"].notna()].copy()

    return out


# ---------------------------------------------------------------------------
# Train CHARLS model and apply to ELSA
# ---------------------------------------------------------------------------
def load_charls_train() -> pd.DataFrame:
    if not os.path.exists(CHARLS_TRAIN):
        raise FileNotFoundError(
            f"CHARLS training data not found at: {CHARLS_TRAIN}\n"
            "Run step_01_data_prep.py first."
        )
    return pd.read_csv(CHARLS_TRAIN)


def get_feature_cols(charls_df: pd.DataFrame, elsa_df: pd.DataFrame) -> list:
    """Return the intersection of available raw16 columns in both datasets."""
    charls_available = [c for c in RAW16 if c in charls_df.columns]
    elsa_available = [c for c in RAW16 if c in elsa_df.columns]
    common = [c for c in RAW16 if c in charls_available and c in elsa_available]
    missing_charls = [c for c in RAW16 if c not in charls_df.columns]
    missing_elsa = [c for c in RAW16 if c not in elsa_df.columns]
    if missing_charls:
        print(f"\n  Columns missing in CHARLS: {missing_charls}")
    if missing_elsa:
        print(f"  Columns missing in ELSA: {missing_elsa}")
    print(f"  Using {len(common)} shared feature columns: {common}")
    return common


def preprocess(X_train: np.ndarray, X_ext: np.ndarray):
    """Fit imputer+scaler on CHARLS train, apply to external (no re-fitting)."""
    num_imputer = SimpleImputer(strategy="median")
    X_train_imp = num_imputer.fit_transform(X_train)
    X_ext_imp = num_imputer.transform(X_ext)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_ext_sc = scaler.transform(X_ext_imp)

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
        obs = y_true[mask].mean()
        pred = proba[mask].mean()
        gap = abs(obs - pred)
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
    half = (z / denom) * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))
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
        nb_all = prev - (1.0 - prev) * odds
        rows.append(
            {
                "threshold": t,
                "net_benefit_model": nb_model,
                "net_benefit_all": nb_all,
                "net_benefit_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def evaluate(y_true, proba, label="ELSA"):
    roc_auc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    brier = brier_score_loss(y_true, proba)
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
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "ece": ece,
        "mce": mce,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_reliability(y_true, proba, title, out_path):
    cal_tbl = calibration_table(y_true, proba)
    xs = cal_tbl["mean_pred"].values
    ys = cal_tbl["observed_rate"].values
    ns = cal_tbl["n"].values
    lows = np.array([wilson_ci(y, int(n))[0] for y, n in zip(ys, ns)])
    highs = np.array([wilson_ci(y, int(n))[1] for y, n in zip(ys, ns)])
    yerr = np.vstack([ys - lows, highs - ys])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.errorbar(
        xs,
        ys,
        yerr=yerr,
        marker="o",
        linewidth=1.8,
        markersize=6,
        capsize=3,
        label="ELSA (raw16)",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (decile)")
    ax.set_ylabel("Observed KOA rate (decile)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_dca(dca_df, title, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        dca_df["threshold"],
        dca_df["net_benefit_model"],
        linewidth=2,
        label="Model (raw16)",
    )
    ax.plot(
        dca_df["threshold"],
        dca_df["net_benefit_all"],
        "--",
        linewidth=1.5,
        label="Treat all",
    )
    ax.axhline(0, color="gray", linestyle=":", linewidth=1.5, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP EXT-VAL-ELSA: ELSA External Validation (raw16 feature set)")
    print("=" * 70)
    print(f"Date:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Wave:  {WAVE}  (change WAVE constant to switch wave)")
    print(f"Model: Random Forest, n_estimators={N_TREES}, seed={RANDOM_STATE}")
    print(f"Feat:  raw16 (raw17 minus Biological Age)")
    print(f"KOA:   ever-diagnosed arthritis + difficulty stooping/kneeling")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. Load ELSA data ----
    elsa_raw = load_elsa()
    rural_urban = load_rural_urban()

    # ---- 2. Build cohort ----
    elsa_cohort = build_cohort(elsa_raw)

    # ---- 3. Feature engineering ----
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    elsa_features = build_features(elsa_cohort, rural_urban)
    elsa_features.to_csv(os.path.join(OUT_DIR, "elsa_analysis_cohort.csv"), index=False)
    print(f"\n  Cohort saved ({len(elsa_features):,} rows): elsa_analysis_cohort.csv")

    # Descriptive
    print(f"\n  Final cohort:  n = {len(elsa_features):,}")
    print(f"  KOA_ELSA prevalence: {elsa_features['KOA'].mean() * 100:.1f}%")
    for col in RAW16:
        if col in elsa_features.columns:
            miss = elsa_features[col].isna().mean() * 100
            if miss > 0:
                print(f"  Missing {col}: {miss:.1f}%")

    # ---- 4. Load CHARLS train and fit model ----
    print("\n" + "=" * 60)
    print("CHARLS MODEL TRAINING (raw16)")
    print("=" * 60)
    charls_df = load_charls_train()
    print(
        f"  CHARLS train: n = {len(charls_df):,}, "
        f"KOA prev = {charls_df['KOA'].mean() * 100:.1f}%"
    )

    # Align feature columns
    feature_cols = get_feature_cols(charls_df, elsa_features)
    if len(feature_cols) < 5:
        raise RuntimeError(
            "Fewer than 5 shared feature columns found. "
            "Check column naming between CHARLS and ELSA feature tables."
        )

    X_charls = charls_df[feature_cols].values.astype(float)
    y_charls = charls_df["KOA"].values

    X_elsa = elsa_features[feature_cols].values.astype(float)
    y_elsa = elsa_features["KOA"].values

    X_charls_sc, X_elsa_sc = preprocess(X_charls, X_elsa)

    # Train RF on CHARLS
    rf = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_charls_sc, y_charls)
    print(f"  RF trained on CHARLS (n={len(y_charls):,})")

    # ---- 5. Evaluate raw (uncalibrated) ----
    print("\n" + "=" * 60)
    print("EVALUATION - RAW (uncalibrated)")
    print("=" * 60)
    proba_elsa_raw = rf.predict_proba(X_elsa_sc)[:, 1]
    metrics_raw = evaluate(y_elsa, proba_elsa_raw, label="ELSA raw16 (raw)")

    # DCA raw
    thresholds = np.arange(0.05, 0.61, 0.01)
    dca_raw_df = decision_curve(y_elsa, proba_elsa_raw, thresholds)
    band_raw = dca_raw_df[(dca_raw_df["threshold"] >= 0.10) & (dca_raw_df["threshold"] <= 0.30)]
    mean_gain_raw = float((band_raw["net_benefit_model"] - band_raw["net_benefit_all"]).mean())
    avoided_raw = (mean_gain_raw * ((1 - band_raw["threshold"]) / band_raw["threshold"]) * 100)
    mean_avoided_raw = float(avoided_raw.mean())
    prev_elsa = float(y_elsa.mean())
    baseline_unnecessary = (1 - prev_elsa) * 100
    rel_reduction_raw = (mean_avoided_raw / baseline_unnecessary * 100) if baseline_unnecessary > 0 else 0

    print(f"\n  DCA (10%-30% band) - RAW:")
    print(f"    Mean net benefit gain vs treat-all: {mean_gain_raw:.4f}")
    print(f"    Avoided unnecessary interventions/100: {mean_avoided_raw:.1f}")
    print(f"    Relative reduction vs treat-all: {rel_reduction_raw:.1f}%")

    # ---- 6. Evaluate isotonic-calibrated ----
    # Split-calibrate-evaluate protocol for external validation:
    #   1. Split ELSA into calibration (30%) and evaluation (70%) subsets
    #   2. Fit isotonic calibrator on calibration subset using CHARLS-trained predictions
    #   3. Evaluate calibrated predictions on evaluation subset (held out from calibration)
    # This prevents the calibrator from overfitting to evaluation data.
    print("\n" + "=" * 60)
    print("EVALUATION - ISOTONIC (split-calibrate-evaluate, 30% calib / 70% eval)")
    print("=" * 60)

    idx_calib, idx_eval = train_test_split(
        np.arange(len(y_elsa)),
        test_size=0.70,
        random_state=RANDOM_STATE,
        stratify=y_elsa,
    )
    print(f"  Calibration set: n = {len(idx_calib):,}")
    print(f"  Evaluation set:  n = {len(idx_eval):,}")

    # Fit isotonic calibrator on calibration subset
    iso_cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_cal.fit(proba_elsa_raw[idx_calib], y_elsa[idx_calib])

    # Apply calibration to evaluation subset
    proba_elsa_iso_eval = iso_cal.predict(proba_elsa_raw[idx_eval])
    y_elsa_eval = y_elsa[idx_eval]

    metrics_iso = evaluate(y_elsa_eval, proba_elsa_iso_eval, label="ELSA raw16 (isotonic, eval subset)")

    # DCA isotonic (on evaluation subset only)
    dca_iso_df = decision_curve(y_elsa_eval, proba_elsa_iso_eval, thresholds)
    band_iso = dca_iso_df[(dca_iso_df["threshold"] >= 0.10) & (dca_iso_df["threshold"] <= 0.30)]
    mean_gain_iso = float((band_iso["net_benefit_model"] - band_iso["net_benefit_all"]).mean())
    avoided_iso = (mean_gain_iso * ((1 - band_iso["threshold"]) / band_iso["threshold"]) * 100)
    mean_avoided_iso = float(avoided_iso.mean())
    prev_elsa_eval = float(y_elsa_eval.mean())
    baseline_unnecessary_iso = (1 - prev_elsa_eval) * 100
    rel_reduction_iso = (mean_avoided_iso / baseline_unnecessary_iso * 100) if baseline_unnecessary_iso > 0 else 0

    print(f"\n  DCA (10%-30% band) - ISOTONIC (eval subset, n={len(y_elsa_eval):,}):")
    print(f"    Mean net benefit gain vs treat-all: {mean_gain_iso:.4f}")
    print(f"    Avoided unnecessary interventions/100: {mean_avoided_iso:.1f}")
    print(f"    Relative reduction vs treat-all: {rel_reduction_iso:.1f}%")

    # ---- 7. Save outputs ----
    # Metrics
    metrics_raw.update(
        {
            "variant": "raw",
            "cohort": "ELSA_replication",
            "feature_set": "raw16",
            "wave": WAVE,
            "n": len(y_elsa),
            "prevalence": prev_elsa,
            "dca_mean_gain_vs_all_10_30": mean_gain_raw,
            "dca_mean_avoided_per_100": mean_avoided_raw,
            "dca_relative_reduction_pct": rel_reduction_raw,
        }
    )
    metrics_iso.update(
        {
            "variant": "isotonic",
            "cohort": "ELSA_replication",
            "feature_set": "raw16",
            "wave": WAVE,
            "n": len(y_elsa_eval),
            "prevalence": prev_elsa_eval,
            "dca_mean_gain_vs_all_10_30": mean_gain_iso,
            "dca_mean_avoided_per_100": mean_avoided_iso,
            "dca_relative_reduction_pct": rel_reduction_iso,
        }
    )

    pd.DataFrame([metrics_raw, metrics_iso]).to_csv(
        os.path.join(OUT_DIR, "ext_val_elsa_metrics.csv"), index=False
    )

    # Calibration tables
    cal_tbl_raw = calibration_table(y_elsa, proba_elsa_raw)
    cal_tbl_raw.to_csv(os.path.join(OUT_DIR, "ext_val_elsa_calibration_table_raw.csv"), index=False)
    cal_tbl_iso = calibration_table(y_elsa_eval, proba_elsa_iso_eval)
    cal_tbl_iso.to_csv(os.path.join(OUT_DIR, "ext_val_elsa_calibration_table_isotonic.csv"), index=False)

    # DCA
    dca_raw_df.to_csv(os.path.join(OUT_DIR, "ext_val_elsa_dca_raw.csv"), index=False)
    dca_iso_df.to_csv(os.path.join(OUT_DIR, "ext_val_elsa_dca_isotonic.csv"), index=False)

    # Plots
    plot_reliability(
        y_elsa,
        proba_elsa_raw,
        title=f"Reliability Diagram: ELSA (wave {WAVE}, raw16, uncalibrated)",
        out_path=os.path.join(OUT_DIR, "reliability_diagram_elsa_raw.png"),
    )
    plot_reliability(
        y_elsa_eval,
        proba_elsa_iso_eval,
        title=f"Reliability Diagram: ELSA (wave {WAVE}, raw16, isotonic)",
        out_path=os.path.join(OUT_DIR, "reliability_diagram_elsa_isotonic.png"),
    )
    plot_dca(
        dca_raw_df,
        title=f"Decision Curve: ELSA (wave {WAVE}, raw16, uncalibrated)",
        out_path=os.path.join(OUT_DIR, "dca_curve_elsa_raw.png"),
    )
    plot_dca(
        dca_iso_df,
        title=f"Decision Curve: ELSA (wave {WAVE}, raw16, isotonic)",
        out_path=os.path.join(OUT_DIR, "dca_curve_elsa_isotonic.png"),
    )

    # ---- 8. Report ----
    slope_raw = metrics_raw["calibration_slope"]
    slope_flag_raw = (
        "ACCEPTABLE (within [0.80, 1.20])"
        if 0.80 <= slope_raw <= 1.20
        else "OUTSIDE RANGE -> logistic recalibration-in-the-large recommended"
    )
    slope_iso = metrics_iso["calibration_slope"]
    slope_flag_iso = (
        "ACCEPTABLE (within [0.80, 1.20])"
        if 0.80 <= slope_iso <= 1.20
        else "OUTSIDE RANGE"
    )

    report_lines = [
        "=" * 70,
        "STEP EXT-VAL-ELSA REPORT",
        "=" * 70,
        f"Date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"ELSA wave:        {WAVE} ({WAVE_YEARS.get(WAVE, 'unknown')})",
        f"Feature set:      raw16 (raw17 minus Biological Age)",
        f"KOA proxy:        ever-diagnosed arthritis + difficulty stooping/kneeling",
        f"CHARLS train:     {CHARLS_TRAIN}",
        f"ELSA harmonized:  {ELSA_HARMONIZED}",
        "",
        "COHORT",
        "-" * 40,
        f"  n                  = {metrics_raw['n']:,}",
        f"  KOA prevalence     = {metrics_raw['prevalence']*100:.1f}%",
        "",
        "PRIMARY METRICS - RAW (uncalibrated)",
        "-" * 40,
        f"  ROC-AUC            = {metrics_raw['roc_auc']:.4f}",
        f"  PR-AUC             = {metrics_raw['pr_auc']:.4f}",
        f"  Brier score        = {metrics_raw['brier']:.4f}",
        f"  ECE                = {metrics_raw['ece']:.4f}",
        f"  MCE                = {metrics_raw['mce']:.4f}",
        f"  Cal. slope         = {slope_raw:.3f}  [{slope_flag_raw}]",
        f"  Cal. intercept     = {metrics_raw['calibration_intercept']:.3f}",
        "",
        "DCA (10%-30% threshold band) - RAW",
        "-" * 40,
        f"  Mean gain vs treat-all = {metrics_raw['dca_mean_gain_vs_all_10_30']:.4f}",
        f"  Avoided per 100 pts    = {metrics_raw['dca_mean_avoided_per_100']:.1f}",
        f"  Relative reduction     = {metrics_raw['dca_relative_reduction_pct']:.1f}%",
        "",
        "PRIMARY METRICS - ISOTONIC (split-calibrate-evaluate on 70% subset)",
        "-" * 40,
        f"  ROC-AUC            = {metrics_iso['roc_auc']:.4f}",
        f"  PR-AUC             = {metrics_iso['pr_auc']:.4f}",
        f"  Brier score        = {metrics_iso['brier']:.4f}",
        f"  ECE                = {metrics_iso['ece']:.4f}",
        f"  MCE                = {metrics_iso['mce']:.4f}",
        f"  Cal. slope         = {slope_iso:.3f}  [{slope_flag_iso}]",
        f"  Cal. intercept     = {metrics_iso['calibration_intercept']:.3f}",
        "",
        "DCA (10%-30% threshold band) - ISOTONIC (eval subset)",
        "-" * 40,
        f"  Mean gain vs treat-all = {metrics_iso['dca_mean_gain_vs_all_10_30']:.4f}",
        f"  Avoided per 100 pts    = {metrics_iso['dca_mean_avoided_per_100']:.1f}",
        f"  Relative reduction     = {metrics_iso['dca_relative_reduction_pct']:.1f}%",
        "",
        "COMPARISON ACROSS COHORTS",
        "-" * 40,
        "  Cohort           ROC-AUC    PR-AUC    Brier    ECE      Cal slope",
        "  CHARLS Internal  0.847      0.688     0.070    0.045    0.989",
        "  CHARLS Holdout   0.907      0.791     0.053    0.056    1.216",
        f"  HRS (raw16)      0.604      0.399     0.258    0.194    0.184",
        f"  ELSA raw16 raw   {metrics_raw['roc_auc']:.3f}      "
        f"{metrics_raw['pr_auc']:.3f}     {metrics_raw['brier']:.3f}    "
        f"{metrics_raw['ece']:.3f}    {slope_raw:.3f}",
        f"  ELSA raw16 iso   {metrics_iso['roc_auc']:.3f}      "
        f"{metrics_iso['pr_auc']:.3f}     {metrics_iso['brier']:.3f}    "
        f"{metrics_iso['ece']:.3f}    {slope_iso:.3f}",
        "",
        "IMPORTANT LIMITATIONS",
        "-" * 40,
        "  1. Biological Age excluded (ELSA public files do not contain",
        "     creatinine, BUN, or platelet assays for KDM-BA computation).",
        "  2. Arthritis proxy uses 'ever diagnosed' (r8arthre), not",
        "     current-wave. This is broader than CHARLS and may inflate",
        "     prevalence.",
        "  3. KOA outcome proxy (arthritis + stooping difficulty) is less",
        "     specific than CHARLS (arthritis + knee-localised pain).",
        "  4. BMI source is nurse-measured (r8mbmi); ~20% missing where",
        "     respondents did not attend nurse visit.",
        "  5. ELSA is a UK English population; CHARLS is Chinese.",
        "     Cross-cultural differences are expected.",
        "",
        "OUTPUT FILES",
        "-" * 40,
        f"  {OUT_DIR}/",
        "    elsa_analysis_cohort.csv",
        "    ext_val_elsa_metrics.csv",
        "    ext_val_elsa_calibration_table_raw.csv",
        "    ext_val_elsa_calibration_table_isotonic.csv",
        "    ext_val_elsa_dca_raw.csv",
        "    ext_val_elsa_dca_isotonic.csv",
        "    reliability_diagram_elsa_raw.png",
        "    reliability_diagram_elsa_isotonic.png",
        "    dca_curve_elsa_raw.png",
        "    dca_curve_elsa_isotonic.png",
        "    ext_val_elsa_report.txt  (this file)",
    ]

    report_path = os.path.join(OUT_DIR, "ext_val_elsa_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved: {report_path}")

    print("\n" + "=" * 70)
    print("STEP EXT-VAL-ELSA COMPLETED")
    print("=" * 70)
    print(f"  RAW ROC-AUC  : {metrics_raw['roc_auc']:.4f}")
    print(f"  RAW Cal slope: {slope_raw:.3f}  [{slope_flag_raw}]")
    print(f"  ISO ROC-AUC  : {metrics_iso['roc_auc']:.4f}")
    print(f"  ISO Cal slope: {slope_iso:.3f}  [{slope_flag_iso}]")
    print(f"  Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
