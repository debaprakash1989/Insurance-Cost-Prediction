"""
Block 3: Machine Learning Modeling for Insurance Cost Prediction
================================================================
Run: python block3_ml_modeling.py
Outputs: trained models, evaluation plots, SHAP plots, model.pkl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import pickle
import os

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠ XGBoost not installed; skipping XGBRegressor.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ SHAP not installed; skipping SHAP plots. Install with: pip install shap")

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2a40",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})
PALETTE = ["#00d4ff","#ff6b9d","#ffd93d","#6bcb77","#c77dff","#ff9a3c","#e0e0e0","#f4845f"]
ACCENT  = "#00d4ff"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(path="Medicalpremium.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    
    # Feature Engineering
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["Age_BMI"] = df["Age"] * df["BMI"]
    df["HealthScore"] = (df["Diabetes"] + df["BloodPressureProblems"] +
                         df["AnyTransplants"] + df["AnyChronicDiseases"] +
                         df["KnownAllergies"] + df["HistoryOfCancerInFamily"] +
                         df["NumberOfMajorSurgeries"])
    df["AgeGroup_num"] = pd.cut(df["Age"], bins=[17,30,40,50,60,70], labels=[1,2,3,4,5]).astype(float)
    df["BMICategory_num"] = pd.cut(df["BMI"], bins=[0,18.5,25,30,100], labels=[1,2,3,4]).astype(float)
    
    features = ["Age","Diabetes","BloodPressureProblems","AnyTransplants",
                 "AnyChronicDiseases","Height","Weight","KnownAllergies",
                 "HistoryOfCancerInFamily","NumberOfMajorSurgeries",
                 "BMI","Age_BMI","HealthScore","AgeGroup_num","BMICategory_num"]
    
    X = df[features].fillna(df[features].median())
    y = df["PremiumPrice"]
    
    return X, y, features, df


def get_models():
    models = {
        "Linear Regression":        LinearRegression(),
        "Ridge Regression":         Ridge(alpha=10),
        "Lasso Regression":         Lasso(alpha=10),
        "Decision Tree":            DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest":            RandomForestRegressor(n_estimators=200, max_depth=10,
                                                          random_state=42, n_jobs=-1),
        "Gradient Boosting":        GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                               max_depth=5, random_state=42),
        "Extra Trees":              ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                               max_depth=5, random_state=42,
                                               eval_metric="rmse", verbosity=0)
    return models


# ─────────────────────────────────────────────────────────────────────────────
# MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_train, X_test, y_train, y_test, scaler=None):
    Xtr = scaler.transform(X_train) if scaler else X_train
    Xte = scaler.transform(X_test)  if scaler else X_test
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}, y_pred


def cross_validate_models(models, X, y, scaler=None):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    X_arr = scaler.transform(X) if scaler else X.values
    
    print("\n── 5-Fold Cross-Validation ──")
    for name, model in models.items():
        scores = cross_val_score(model, X_arr, y, cv=kf,
                                 scoring="neg_root_mean_squared_error", n_jobs=-1)
        cv_results[name] = -scores
        print(f"  {name:25s}: RMSE = {-scores.mean():8,.1f} ± {scores.std():6,.1f}")
    return cv_results


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor="#0f0f1a")
    fig.suptitle("Model Performance Comparison", fontsize=20,
                 fontweight="bold", color=ACCENT, y=0.99)
    
    metrics = ["RMSE","MAE","R2","MAPE"]
    titles  = ["RMSE ↓ (lower is better)","MAE ↓ (lower is better)",
               "R² ↑ (higher is better)","MAPE % ↓ (lower is better)"]
    
    sorted_by = results_df.sort_values("R2", ascending=False)
    
    for ax, metric, title, color in zip(axes.flatten(), metrics, titles, PALETTE):
        vals = sorted_by[metric]
        bars = ax.bar(sorted_by["Model"], vals, color=color,
                      alpha=0.8, edgecolor="#0f0f1a")
        best_idx = vals.argmin() if metric != "R2" else vals.argmax()
        bars[best_idx].set_color("#ffd93d")
        bars[best_idx].set_alpha(1.0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8, color="#e0e0e0")
        ax.set_title(title, fontsize=12, color=color, fontweight="bold")
        ax.tick_params(axis="x", rotation=40)
        ax.set_ylabel(metric, fontsize=10)
    
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: model_comparison.png")


def plot_actual_vs_predicted(models_preds, y_test):
    n = len(models_preds)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*7, rows*6), facecolor="#0f0f1a")
    fig.suptitle("Actual vs Predicted Premium Price", fontsize=20,
                 fontweight="bold", color=ACCENT, y=1.01)
    axes = axes.flatten() if rows > 1 else axes.flatten()
    
    for ax, (name, y_pred), color in zip(axes, models_preds.items(), PALETTE):
        r2 = r2_score(y_test, y_pred)
        ax.scatter(y_test, y_pred, alpha=0.55, s=20, color=color, edgecolors="none")
        lim = [min(y_test.min(), y_pred.min()) - 500,
               max(y_test.max(), y_pred.max()) + 500]
        ax.plot(lim, lim, "w--", linewidth=1.5, alpha=0.6, label="Perfect fit")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Actual ₹", fontsize=10); ax.set_ylabel("Predicted ₹", fontsize=10)
        ax.set_title(f"{name}\nR²={r2:.4f}", fontsize=11, color=color, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.3)
    
    for ax in axes[n:]: ax.set_visible(False)
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: actual_vs_predicted.png")


def plot_residuals(best_name, y_test, y_pred):
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="#0f0f1a")
    fig.suptitle(f"Residual Analysis — {best_name}", fontsize=18,
                 fontweight="bold", color=ACCENT, y=1.01)
    
    # Residual scatter
    axes[0].scatter(y_pred, residuals, alpha=0.55, s=20,
                    c=residuals, cmap="RdYlGn", edgecolors="none")
    axes[0].axhline(0, color="#ffd93d", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted ₹", fontsize=11)
    axes[0].set_ylabel("Residual", fontsize=11)
    axes[0].set_title("Residuals vs Predicted", fontsize=13, color="#00d4ff")
    
    # Distribution
    axes[1].hist(residuals, bins=30, color="#00d4ff", alpha=0.7, edgecolor="#0f0f1a", density=True)
    residuals.plot.kde(ax=axes[1], color="#ffd93d", linewidth=2)
    axes[1].set_xlabel("Residual", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("Residual Distribution", fontsize=13, color="#ff6b9d")
    
    # Q-Q plot
    from scipy import stats as sp_stats
    sp_stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].get_lines()[0].set(color="#00d4ff", markersize=3, alpha=0.6)
    axes[2].get_lines()[1].set(color="#ffd93d", linewidth=2)
    axes[2].set_title("Q-Q Plot (Normal)", fontsize=13, color="#6bcb77")
    
    plt.tight_layout()
    plt.savefig("residual_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: residual_analysis.png")


def plot_feature_importance(best_model, best_name, feature_names, X_test, y_test, scaler=None):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0f0f1a")
    fig.suptitle(f"Feature Importance — {best_name}", fontsize=18,
                 fontweight="bold", color=ACCENT, y=1.01)
    
    Xte = scaler.transform(X_test) if scaler else X_test.values
    
    # Built-in importance (tree-based)
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        idx = np.argsort(imp)
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(idx))]
        axes[0].barh([feature_names[i] for i in idx], imp[idx],
                     color=colors, edgecolor="#0f0f1a")
        axes[0].set_xlabel("Importance Score", fontsize=11)
        axes[0].set_title("Built-in Feature Importance", fontsize=13, color="#00d4ff")
    else:
        axes[0].text(0.5, 0.5, "Not available\nfor this model",
                     ha="center", va="center", transform=axes[0].transAxes,
                     fontsize=12, color="#aaa")
    
    # Permutation importance
    perm = permutation_importance(best_model, Xte, y_test,
                                  n_repeats=10, random_state=42, n_jobs=-1)
    idx = np.argsort(perm.importances_mean)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(idx))]
    axes[1].barh([feature_names[i] for i in idx], perm.importances_mean[idx],
                 xerr=perm.importances_std[idx],
                 color=colors, edgecolor="#0f0f1a", error_kw={"ecolor":"#ffd93d","capsize":3})
    axes[1].set_xlabel("Permutation Importance", fontsize=11)
    axes[1].set_title("Permutation Feature Importance", fontsize=13, color="#ff6b9d")
    
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: feature_importance.png")


def plot_shap(best_model, X_test, feature_names, scaler=None):
    if not SHAP_AVAILABLE:
        return
    try:
        Xte = pd.DataFrame(scaler.transform(X_test), columns=feature_names) \
              if scaler else X_test.reset_index(drop=True)
        explainer = shap.Explainer(best_model, Xte)
        shap_values = explainer(Xte)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor="#0f0f1a")
        fig.suptitle("SHAP Value Analysis", fontsize=18, fontweight="bold",
                     color=ACCENT, y=1.01)
        
        plt.sca(axes[0])
        shap.plots.beeswarm(shap_values, show=False, max_display=15,
                             color=plt.cm.RdBu_r)
        axes[0].set_title("SHAP Beeswarm", fontsize=13, color="#00d4ff")
        
        plt.sca(axes[1])
        shap.plots.bar(shap_values, show=False, max_display=15)
        axes[1].set_title("SHAP Mean |Value|", fontsize=13, color="#ff6b9d")
        
        plt.tight_layout()
        plt.savefig("shap_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
        plt.close()
        print("✔ Saved: shap_analysis.png")
    except Exception as e:
        print(f"⚠ SHAP plot skipped: {e}")


def plot_cv_boxplot(cv_results):
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0f0f1a")
    fig.suptitle("5-Fold Cross-Validation — RMSE Distribution",
                 fontsize=18, fontweight="bold", color=ACCENT, y=1.01)
    
    data   = list(cv_results.values())
    labels = list(cv_results.keys())
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="#ffd93d", linewidth=2.5),
                    whiskerprops=dict(color="#aaa"),
                    capprops=dict(color="#aaa"),
                    flierprops=dict(marker="o", alpha=0.5, markersize=4))
    for box, color in zip(bp["boxes"], PALETTE):
        box.set_facecolor(color + "55")
    ax.set_ylabel("RMSE (₹)", fontsize=12)
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig("cv_boxplot.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: cv_boxplot.png")


def plot_learning_curves(best_model, best_name, X, y, scaler=None):
    from sklearn.model_selection import learning_curve
    X_arr = scaler.transform(X) if scaler else X.values
    
    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_arr, y, cv=5,
        scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#0f0f1a")
    fig.suptitle(f"Learning Curves — {best_name}", fontsize=18,
                 fontweight="bold", color=ACCENT)
    
    train_rmse = -train_scores.mean(axis=1)
    val_rmse   = -val_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_std    = val_scores.std(axis=1)
    
    ax.plot(train_sizes, train_rmse, color="#00d4ff", linewidth=2.5, label="Training RMSE", marker="o")
    ax.plot(train_sizes, val_rmse,   color="#ff6b9d", linewidth=2.5, label="Validation RMSE", marker="s")
    ax.fill_between(train_sizes, train_rmse - train_std, train_rmse + train_std,
                    alpha=0.15, color="#00d4ff")
    ax.fill_between(train_sizes, val_rmse - val_std, val_rmse + val_std,
                    alpha=0.15, color="#ff6b9d")
    ax.set_xlabel("Training Samples", fontsize=12)
    ax.set_ylabel("RMSE (₹)", fontsize=12)
    ax.legend(fontsize=11, framealpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: learning_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE ARTIFACTS FOR BLOCK 4
# ─────────────────────────────────────────────────────────────────────────────
def save_model_artifacts(best_model, scaler, feature_names, best_name):
    artifacts = {
        "model":         best_model,
        "scaler":        scaler,
        "feature_names": feature_names,
        "model_name":    best_name,
    }
    with open("model_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print(f"\n✔ Model artifacts saved: model_artifacts.pkl")
    print(f"  Best model: {best_name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("BLOCK 3 — MACHINE LEARNING MODELING")
    print("="*65)
    
    X, y, feature_names, df = load_and_preprocess()
    
    print(f"\nFeatures ({len(feature_names)}): {feature_names}")
    print(f"Target — PremiumPrice | Range: ₹{y.min():,} – ₹{y.max():,}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Scaler (used for linear models only)
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    models = get_models()
    
    print("\n── Training & Evaluating Models ──")
    results     = []
    preds_dict  = {}
    models_fit  = {}
    
    for name, model in models.items():
        use_scaler = name in ("Linear Regression","Ridge Regression","Lasso Regression")
        sc = scaler if use_scaler else None
        metrics, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test, sc)
        results.append({"Model": name, **metrics})
        preds_dict[name] = y_pred
        models_fit[name] = (model, sc)
        print(f"  {name:25s}: RMSE=₹{metrics['RMSE']:8,.1f}  "
              f"MAE=₹{metrics['MAE']:7,.1f}  R²={metrics['R2']:.4f}  "
              f"MAPE={metrics['MAPE']:.2f}%")
    
    results_df = pd.DataFrame(results)
    
    # Best model
    best_row   = results_df.loc[results_df["R2"].idxmax()]
    best_name  = best_row["Model"]
    best_model, best_scaler = models_fit[best_name]
    best_preds = preds_dict[best_name]
    
    print(f"\n🏆 Best Model: {best_name}")
    print(f"   R² = {best_row['R2']:.4f}  RMSE = ₹{best_row['RMSE']:,.1f}")
    
    # Cross validation
    cv_results = cross_validate_models(models, X, y)
    
    # Plots
    print("\n── Generating Plots ──")
    plot_model_comparison(results_df)
    plot_actual_vs_predicted(preds_dict, y_test)
    plot_residuals(best_name, y_test, best_preds)
    plot_feature_importance(best_model, best_name, feature_names,
                            X_test, y_test, best_scaler)
    plot_shap(best_model, X_test, feature_names, best_scaler)
    plot_cv_boxplot(cv_results)
    plot_learning_curves(best_model, best_name, X, y, best_scaler)
    
    # Confidence / prediction intervals (bootstrap)
    print("\n── Prediction Interval (Bootstrap, 90% CI) ──")
    np.random.seed(42)
    boot_preds = []
    Xte_arr = best_scaler.transform(X_test) if best_scaler else X_test.values
    from sklearn.utils import resample
    for _ in range(100):
        Xb, yb = resample(X_train, y_train)
        Xb_arr = best_scaler.transform(Xb) if best_scaler else Xb.values
        m = type(best_model)(**best_model.get_params())
        m.fit(Xb_arr, yb)
        boot_preds.append(m.predict(Xte_arr))
    boot_preds = np.array(boot_preds)
    ci_low  = np.percentile(boot_preds, 5,  axis=0)
    ci_high = np.percentile(boot_preds, 95, axis=0)
    coverage = np.mean((y_test.values >= ci_low) & (y_test.values <= ci_high))
    print(f"   90% CI Coverage: {coverage*100:.1f}%")
    print(f"   Mean PI Width:   ₹{(ci_high - ci_low).mean():,.0f}")
    
    # Save
    save_model_artifacts(best_model, best_scaler, feature_names, best_name)
    
    # Final results table
    print("\n── Final Results Summary ──")
    print(results_df.sort_values("R2", ascending=False).to_string(index=False))
    print("\n✅ Block 3 complete!")
