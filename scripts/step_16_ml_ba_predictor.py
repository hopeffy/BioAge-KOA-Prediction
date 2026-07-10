"""
STEP 16: ML-based Biological Age Predictors
============================================
Train ML regressors in CHARLS to predict two Biological Age targets from the
raw16 features (the sociodemographic/clinical variables available in ELSA and
HRS):

  1. ML_BA      -> predict the existing 'Biological Age' column (Sheet1 BA)
  2. ML_KDM_BA  -> predict the computed KDM-BA (BA_KDM_log)

Both predicted BA variants can then be transported to ELSA/HRS, where measured
biomarkers for KDM-BA are unavailable. We also save both models together so
step 17 can evaluate each separately and as an ensemble of the two predicted
BA scores.

Targets
-------
  Biological Age (continuous, years) -- pre-computed in CHARLS Sheet1
  BA_KDM_log (continuous, years)     -- Klemera-Doubal BA from 8 biomarkers

Features
--------
  raw16 set = raw17 minus {Biological Age, wave, Time} plus a constant
  wave_id marker. These are the features harmonised for ELSA/HRS.

Models evaluated
----------------
  - Random Forest regressor
  - XGBoost regressor
  - LightGBM regressor
  - Ensemble mean of the three

Evaluation
----------
  5-fold stratified (on target-quintiles) CV on the internal training partition,
  plus single-shot evaluation on the early unseen holdout. Metrics: R2, MAE,
  RMSE, Pearson r.

Outputs
-------
  step_16_ml_ba_predictor/
    step16_report.txt                  - narrative report comparing targets
    step16_metrics.csv                 - CV + holdout metrics per target/model
    step16_predictions_<target>.csv    - holdout predictions per target
    step16_predictions_cv_<target>.csv - CV OOF predictions per target
    step16_feature_importance.csv      - per-model feature importance per target
    charls_internal_with_ml_ba.csv     - internal partition with both ML-BAs
    charls_unseen_with_ml_ba.csv       - unseen holdout with both ML-BAs
    charls_all_with_ml_ba.csv          - full Dataset A with both ML-BAs
    ml_ba_ensemble_model.pkl           - fitted models for ML_BA
    ml_kdm_ba_ensemble_model.pkl       - fitted models for ML_KDM_BA
    ml_ba_combo_ensemble_model.pkl     - fitted models for both targets
"""

import os
import warnings
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "step_03_kdm_ba")
SPLIT_DIR = os.path.join(BASE_DIR, "step_01_data_prep")
OUT_DIR = os.path.join(BASE_DIR, "step_16_ml_ba_predictor")

DATA_PATH = os.path.join(DATA_DIR, "dataset_A_with_kdm_ba.csv")
TRAIN_PATH = os.path.join(SPLIT_DIR, "dataset_A_internal_train.csv")
UNSEEN_PATH = os.path.join(SPLIT_DIR, "dataset_A_unseen_test.csv")

TARGETS = {
    "ML_BA": "Biological Age",
    "ML_KDM_BA": "BA_KDM_log",
}
N_FOLDS = 5
RANDOM_STATE = 42
N_TREES = 300

# raw16 feature set: same columns harmonised for HRS/ELSA
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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def pearson_r(y_true, y_pred):
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_metrics(y_true, y_pred):
    """Return a dictionary of regression metrics."""
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson_r": pearson_r(y_true, y_pred),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "std_true": float(np.std(y_true)),
        "std_pred": float(np.std(y_pred)),
        "n": len(y_true),
    }


def build_stratifier(y_values: np.ndarray) -> np.ndarray:
    """
    Build target-quintile labels for stratified CV. This keeps the target
    distribution similar across folds.
    """
    return pd.qcut(y_values, q=5, labels=False, duplicates="drop")


def get_models():
    """Return a dict of candidate regressors."""
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=N_TREES,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=5,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=N_TREES,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=N_TREES,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def fit_models(X_train, y_train):
    """Fit all candidate models and return them."""
    models = get_models()
    fitted = {}
    for name, model in models.items():
        print(f"  Fitting {name}...")
        fitted[name] = model.fit(X_train, y_train)
    return fitted


def ensemble_predict(fitted_models, X):
    """Mean prediction across all fitted models."""
    preds = np.column_stack([m.predict(X) for m in fitted_models.values()])
    return preds.mean(axis=1)


def cross_validate_models(df_train, feature_cols, target_col):
    """
    5-fold stratified CV for each model and ensemble.
    Returns metrics and out-of-fold predictions.
    """
    print(f"\n  Target: {target_col}")
    print("  " + "-" * 58)

    X = df_train[feature_cols].values.astype(float)
    y = df_train[target_col].values

    # Drop rows with missing target; features are imputed later per fold.
    complete_mask = pd.notna(y)
    train_idx_map = np.where(complete_mask)[0]
    if not complete_mask.all():
        n_missing = (~complete_mask).sum()
        print(f"  Dropping {n_missing:,} rows with missing target '{target_col}'")
        X = X[complete_mask]
        y = y[complete_mask]

    stratifier = build_stratifier(y)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = {name: np.full(len(y), np.nan) for name in list(get_models().keys()) + ["Ensemble"]}
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, stratifier), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_tr)
        X_val_imp = imp.transform(X_val)
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_imp)
        X_val_sc = scaler.transform(X_val_imp)

        fitted = fit_models(X_tr_sc, y_tr)
        for name, model in fitted.items():
            oof_preds[name][val_idx] = model.predict(X_val_sc)

        oof_preds["Ensemble"][val_idx] = ensemble_predict(fitted, X_val_sc)

        for name in oof_preds:
            m = regression_metrics(y_val, oof_preds[name][val_idx])
            m.update({"model": name, "fold": fold, "eval": "cv_fold", "target": target_col})
            fold_metrics.append(m)
    # Aggregate CV metrics per model and map OOF predictions back to original rows
    cv_metrics = []
    oof_mapped = {name: np.full(len(df_train), np.nan) for name in oof_preds}
    for name in oof_preds:
        oof_mapped[name][train_idx_map] = oof_preds[name]
        m = regression_metrics(y, oof_preds[name])
        m.update({"model": name, "fold": "all", "eval": "cv_oof", "target": target_col})
        cv_metrics.append(m)

    metrics_df = pd.DataFrame(fold_metrics + cv_metrics)
    oof_df = pd.DataFrame(oof_mapped)
    oof_df[f"true_{target_col}"] = df_train[target_col].values
    oof_df["fold"] = np.nan
    oof_df.loc[train_idx_map, "fold"] = stratifier

    return metrics_df, oof_df


def evaluate_on_holdout(df_train, df_unseen, feature_cols, target_col):
    """
    Fit models on the full internal training partition and evaluate on the
    early unseen holdout.
    """
    print(f"\n  Target: {target_col}")
    print("  " + "-" * 58)

    X_train = df_train[feature_cols].values.astype(float)
    y_train = df_train[target_col].values

    # Drop rows with missing target
    train_complete = pd.notna(y_train)
    if not train_complete.all():
        n_missing = (~train_complete).sum()
        print(f"  Dropping {n_missing:,} train rows with missing target '{target_col}'")
        X_train = X_train[train_complete]
        y_train = y_train[train_complete]

    X_unseen = df_unseen[feature_cols].values.astype(float)
    y_unseen = df_unseen[target_col].values

    # Drop rows with missing target in holdout (don't evaluate on NaN)
    unseen_complete = pd.notna(y_unseen)
    unseen_idx_map = np.where(unseen_complete)[0]
    if not unseen_complete.all():
        n_missing = (~unseen_complete).sum()
        print(f"  Dropping {n_missing:,} holdout rows with missing target '{target_col}'")
        X_unseen = X_unseen[unseen_complete]
        y_unseen = y_unseen[unseen_complete]

    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_unseen_imp = imp.transform(X_unseen)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_imp)
    X_unseen_sc = scaler.transform(X_unseen_imp)

    fitted = fit_models(X_train_sc, y_train)

    holdout_metrics = []
    holdout_preds = pd.DataFrame({target_col: y_unseen})

    for name, model in fitted.items():
        preds = model.predict(X_unseen_sc)
        holdout_preds[f"pred_{name}"] = preds
        m = regression_metrics(y_unseen, preds)
        m.update({"model": name, "fold": "-", "eval": "unseen_holdout", "target": target_col})
        holdout_metrics.append(m)

    # Add ensemble explicitly
    ensemble_preds = ensemble_predict(fitted, X_unseen_sc)
    holdout_preds["pred_Ensemble"] = ensemble_preds
    m = regression_metrics(y_unseen, ensemble_preds)
    m.update({"model": "Ensemble", "fold": "-", "eval": "unseen_holdout", "target": target_col})
    holdout_metrics.append(m)

    metrics_df = pd.DataFrame(holdout_metrics)
    return metrics_df, holdout_preds, fitted, imp, scaler, unseen_idx_map


def extract_importance(fitted_models, feature_cols, target_col):
    """Extract and normalize feature importance per model."""
    rows = []
    for name, model in fitted_models.items():
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        else:
            continue
        imp = imp / (imp.sum() + 1e-12)
        for col, v in zip(feature_cols, imp):
            rows.append({"target": target_col, "model": name, "feature": col, "importance": v})
    return pd.DataFrame(rows)


def plot_predictions(holdout_preds, out_dir, target_col):
    """Scatter plot of predicted vs true BA on the unseen holdout."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    model_cols = [c for c in holdout_preds.columns if c.startswith("pred_")]
    for ax, col in zip(axes, model_cols):
        name = col.replace("pred_", "")
        y_true = holdout_preds[target_col].values
        y_pred = holdout_preds[col].values
        ax.scatter(y_true, y_pred, alpha=0.3, s=5)
        mn, mx = y_true.min(), y_true.max()
        ax.plot([mn, mx], [mn, mx], "r--", lw=1)
        ax.set_xlabel(f"True {target_col}")
        ax.set_ylabel(f"Predicted {target_col}")
        r = pearson_r(y_true, y_pred)
        ax.set_title(f"{name}\nR2={r2_score(y_true, y_pred):.3f}, r={r:.3f}")
    plt.tight_layout()
    safe_name = target_col.replace(" ", "_").replace(".", "_")
    path = os.path.join(out_dir, f"step16_holdout_predictions_{safe_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {path}")


def load_data():
    """Load CHARLS full, train, and unseen partitions and merge extra columns."""
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df_full = pd.read_csv(DATA_PATH)
    print(f"  Full CHARLS (step_03): {df_full.shape}")

    df_train = pd.read_csv(TRAIN_PATH)
    df_unseen = pd.read_csv(UNSEEN_PATH)
    print(f"  Internal train: {df_train.shape}")
    print(f"  Unseen holdout: {df_unseen.shape}")

    # Merge extra columns from df_full because step_01 resets indices.
    shared_cols_train = [c for c in df_train.columns if c in df_full.columns]
    extra_cols_train = [c for c in df_full.columns if c not in df_train.columns]
    df_train = df_train.merge(
        df_full[shared_cols_train + extra_cols_train],
        on=shared_cols_train,
        how="left",
        suffixes=("", "_drop"),
    )
    df_train = df_train.loc[:, ~df_train.columns.str.endswith("_drop")]
    if len(df_train) != len(pd.read_csv(TRAIN_PATH)):
        raise RuntimeError(
            f"Train merge expanded rows: {len(pd.read_csv(TRAIN_PATH))} -> {len(df_train)}"
        )

    shared_cols_unseen = [c for c in df_unseen.columns if c in df_full.columns]
    extra_cols_unseen = [c for c in df_full.columns if c not in df_unseen.columns]
    df_unseen = df_unseen.merge(
        df_full[shared_cols_unseen + extra_cols_unseen],
        on=shared_cols_unseen,
        how="left",
        suffixes=("", "_drop"),
    )
    df_unseen = df_unseen.loc[:, ~df_unseen.columns.str.endswith("_drop")]
    if len(df_unseen) != len(pd.read_csv(UNSEEN_PATH)):
        raise RuntimeError(
            f"Unseen merge expanded rows: {len(pd.read_csv(UNSEEN_PATH))} -> {len(df_unseen)}"
        )

    if "wave_id" not in df_train.columns:
        df_train["wave_id"] = 1
    if "wave_id" not in df_unseen.columns:
        df_unseen["wave_id"] = 1

    feature_cols = [c for c in RAW16 if c in df_full.columns]
    missing = [c for c in RAW16 if c not in df_full.columns]
    if missing:
        print(f"  WARNING: missing raw16 columns: {missing}")
    print(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    return df_full, df_train, df_unseen, feature_cols


def train_target(df_train, df_unseen, feature_cols, target_col, out_dir):
    """Run CV + holdout evaluation for one BA target and save artifacts."""
    print("\n" + "=" * 60)
    print(f"TRAINING TARGET: {target_col}")
    print("=" * 60)

    cv_metrics, oof_df = cross_validate_models(df_train, feature_cols, target_col)
    holdout_metrics, holdout_preds, fitted, imp, scaler, unseen_idx_map = evaluate_on_holdout(
        df_train, df_unseen, feature_cols, target_col
    )

    all_metrics = pd.concat([cv_metrics, holdout_metrics], ignore_index=True)
    safe_name = target_col.replace(" ", "_").replace(".", "_")
    all_metrics.to_csv(os.path.join(out_dir, f"step16_metrics_{safe_name}.csv"), index=False)
    oof_df.to_csv(os.path.join(out_dir, f"step16_predictions_cv_{safe_name}.csv"), index=False)
    holdout_preds.to_csv(os.path.join(out_dir, f"step16_predictions_{safe_name}.csv"), index=False)

    importance_df = extract_importance(fitted, feature_cols, target_col)
    importance_df.to_csv(os.path.join(out_dir, f"step16_feature_importance_{safe_name}.csv"), index=False)

    plot_predictions(holdout_preds, out_dir, target_col)

    # Refit on full Dataset A and save artifact
    df_all = pd.concat([df_train, df_unseen], ignore_index=True)
    X_all = df_all[feature_cols].values.astype(float)
    y_all = df_all[target_col].values

    # Drop rows with missing target
    all_complete = pd.notna(y_all)
    if not all_complete.all():
        n_missing = (~all_complete).sum()
        print(f"  Dropping {n_missing:,} full-dataset rows with missing target '{target_col}'")
        X_all = X_all[all_complete]
        y_all = y_all[all_complete]

    imp_all = SimpleImputer(strategy="median")
    X_all_imp = imp_all.fit_transform(X_all)
    scaler_all = StandardScaler()
    X_all_sc = scaler_all.fit_transform(X_all_imp)

    final_models = fit_models(X_all_sc, y_all)
    artifact = {
        "models": final_models,
        "imputer": imp_all,
        "scaler": scaler_all,
        "feature_cols": feature_cols,
        "target": target_col,
        "random_state": RANDOM_STATE,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    return artifact, all_metrics, oof_df["Ensemble"].values, holdout_preds["pred_Ensemble"].values, unseen_idx_map


def main():
    print("=" * 80)
    print("STEP 16: ML-BASED BIOLOGICAL AGE PREDICTORS")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    ensure_dir(OUT_DIR)

    df_full, df_train, df_unseen, feature_cols = load_data()

    artifacts = {}
    all_metrics_list = []
    oof_predictions = {}
    holdout_predictions = {}
    holdout_index_maps = {}

    for col_name, target_col in TARGETS.items():
        if target_col not in df_full.columns:
            print(f"\n  WARNING: target column '{target_col}' not found. Skipping {col_name}.")
            continue

        artifact, metrics, oof_pred, holdout_pred, unseen_idx_map = train_target(
            df_train, df_unseen, feature_cols, target_col, OUT_DIR
        )
        artifacts[col_name] = artifact
        all_metrics_list.append(metrics)
        oof_predictions[col_name] = oof_pred
        holdout_predictions[col_name] = holdout_pred
        holdout_index_maps[col_name] = unseen_idx_map

        artifact_path = os.path.join(OUT_DIR, f"{col_name.lower()}_ensemble_model.pkl")
        joblib.dump(artifact, artifact_path)
        print(f"  Saved artifact: {artifact_path}")

    # Combined artifact with both targets
    combo_artifact = {
        "targets": {k: v["target"] for k, v in artifacts.items()},
        "artifacts": artifacts,
        "feature_cols": feature_cols,
        "random_state": RANDOM_STATE,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    combo_path = os.path.join(OUT_DIR, "ml_ba_combo_ensemble_model.pkl")
    joblib.dump(combo_artifact, combo_path)
    print(f"\n  Saved combo artifact: {combo_path}")

    # Add both ML-BA predictions to CHARLS partitions
    for col_name, preds in oof_predictions.items():
        df_train[col_name] = preds
        df_train[f"{col_name}_source"] = "cv_oof_ensemble"
    for col_name, preds in holdout_predictions.items():
        # Map back to original df_unseen rows; NaN for dropped missing-target rows
        mapped = np.full(len(df_unseen), np.nan)
        mapped[holdout_index_maps[col_name]] = preds
        df_unseen[col_name] = mapped
        df_unseen[f"{col_name}_source"] = "unseen_holdout_ensemble"
        # For rows dropped due to missing target, mark source as unavailable
        df_unseen.loc[np.isnan(mapped), f"{col_name}_source"] = "missing_target_unavailable"

    df_train.to_csv(os.path.join(OUT_DIR, "charls_internal_with_ml_ba.csv"), index=False)
    df_unseen.to_csv(os.path.join(OUT_DIR, "charls_unseen_with_ml_ba.csv"), index=False)

    df_all = pd.concat([df_train, df_unseen], ignore_index=True)
    # Recompute full-dataset predictions using saved combo artifact for consistency
    for col_name, artifact in artifacts.items():
        X_all = df_all[feature_cols].values.astype(float)
        X_all_imp = artifact["imputer"].transform(X_all)
        X_all_sc = artifact["scaler"].transform(X_all_imp)
        df_all[col_name] = ensemble_predict(artifact["models"], X_all_sc)
        df_all[f"{col_name}_source"] = "full_dataset_ensemble"
    df_all.to_csv(os.path.join(OUT_DIR, "charls_all_with_ml_ba.csv"), index=False)

    # Combine all metrics
    all_metrics = pd.concat(all_metrics_list, ignore_index=True)
    all_metrics.to_csv(os.path.join(OUT_DIR, "step16_metrics.csv"), index=False)

    # Combine feature importance
    importance_files = [
        os.path.join(OUT_DIR, f"step16_feature_importance_{v.replace(' ', '_').replace('.', '_')}.csv")
        for v in TARGETS.values()
    ]
    importance_dfs = [pd.read_csv(f) for f in importance_files if os.path.exists(f)]
    if importance_dfs:
        pd.concat(importance_dfs, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "step16_feature_importance.csv"), index=False
        )

    # Report
    report_lines = [
        "=" * 80,
        "STEP 16: ML-BASED BIOLOGICAL AGE PREDICTORS REPORT",
        "=" * 80,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "OBJECTIVE",
        "-" * 40,
        "Train ML regressors to predict two Biological Age targets from raw16",
        "features only, so the same predictors can be applied to ELSA/HRS.",
        "",
        "DATA",
        "-" * 40,
        f"  Full CHARLS (Dataset A): {len(df_all):,} rows",
        f"  Internal training partition: {len(df_train):,} rows",
        f"  Unseen holdout partition: {len(df_unseen):,} rows",
        f"  Features ({len(feature_cols)}): {', '.join(feature_cols)}",
        "",
        "TARGETS",
        "-" * 40,
    ]
    for col_name, target_col in TARGETS.items():
        report_lines.append(f"  {col_name} -> {target_col}")

    report_lines.extend([
        "",
        "MODELS",
        "-" * 40,
        "  - RandomForestRegressor (n_estimators=300)",
        "  - XGBRegressor",
        "  - LGBMRegressor",
        "  - Ensemble = mean of the three",
        "",
        "INTERNAL 5-FOLD CV RESULTS (R2 / MAE / RMSE / Pearson r)",
        "-" * 40,
    ])

    cv_summary = all_metrics[all_metrics["eval"] == "cv_oof"].sort_values(
        ["target", "r2"], ascending=[True, False]
    )
    for target_col, group in cv_summary.groupby("target"):
        report_lines.append(f"  Target: {target_col}")
        for _, row in group.iterrows():
            report_lines.append(
                f"    {row['model']:<12s} R2={row['r2']:.4f}  MAE={row['mae']:.3f}  "
                f"RMSE={row['rmse']:.3f}  r={row['pearson_r']:.4f}  n={int(row['n'])}"
            )

    report_lines.extend([
        "",
        "UNSEEN HOLDOUT RESULTS (R2 / MAE / RMSE / Pearson r)",
        "-" * 40,
    ])

    holdout_summary = all_metrics[all_metrics["eval"] == "unseen_holdout"].sort_values(
        ["target", "r2"], ascending=[True, False]
    )
    for target_col, group in holdout_summary.groupby("target"):
        report_lines.append(f"  Target: {target_col}")
        for _, row in group.iterrows():
            report_lines.append(
                f"    {row['model']:<12s} R2={row['r2']:.4f}  MAE={row['mae']:.3f}  "
                f"RMSE={row['rmse']:.3f}  r={row['pearson_r']:.4f}  n={int(row['n'])}"
            )

    report_lines.extend([
        "",
        "INTERPRETATION",
        "-" * 40,
        "The ML-BA predictors use only sociodemographic/clinical features",
        "available in HRS and ELSA. They are not substitutes for KDM-BA, but",
        "enable fair 'predicted BA' scenarios in external cohorts.",
        "",
        "NEXT STEPS",
        "-" * 40,
        "  1. Apply saved models to ELSA and HRS raw16 cohorts.",
        "  2. Re-run KOA prediction with raw16 + ML_BA and raw16 + ML_KDM_BA.",
        "  3. Compare four scenarios: no BA, ML_BA, ML_KDM_BA, both ML-BAs.",
        "",
        "OUTPUT FILES",
        "-" * 40,
        f"  {OUT_DIR}/",
        "    step16_metrics.csv",
        "    step16_metrics_<target>.csv",
        "    step16_predictions_<target>.csv",
        "    step16_predictions_cv_<target>.csv",
        "    step16_feature_importance.csv",
        "    charls_internal_with_ml_ba.csv",
        "    charls_unseen_with_ml_ba.csv",
        "    charls_all_with_ml_ba.csv",
        "    ml_ba_ensemble_model.pkl",
        "    ml_kdm_ba_ensemble_model.pkl",
        "    ml_ba_combo_ensemble_model.pkl",
        "    step16_holdout_predictions_<target>.png",
    ])

    report_path = os.path.join(OUT_DIR, "step16_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Saved report: {report_path}")

    print("\n" + "=" * 80)
    print("STEP 16 COMPLETED")
    print("=" * 80)
    for target_col, group in cv_summary.groupby("target"):
        best = group.iloc[0]
        print(f"  Best CV  R2 for {target_col}: {best['r2']:.4f} ({best['model']})")
    for target_col, group in holdout_summary.groupby("target"):
        best = group.iloc[0]
        print(f"  Best Holdout R2 for {target_col}: {best['r2']:.4f} ({best['model']})")
    print(f"  Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
