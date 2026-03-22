"""
Block 2: EDA and Hypothesis Testing for Insurance Cost Prediction
=================================================================
Run: python block2_eda_hypothesis.py
Outputs all plots as PNG files and prints statistical test results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, pearsonr
import warnings
warnings.filterwarnings("ignore")

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
    "font.family":      "DejaVu Sans",
})
PALETTE = ["#00d4ff", "#ff6b9d", "#ffd93d", "#6bcb77", "#c77dff", "#ff9a3c"]
ACCENT  = "#00d4ff"

# ── Load Data ──────────────────────────────────────────────────────────────────
def load_data(path="Medicalpremium.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    # Derive BMI
    df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)
    df["AgeGroup"] = pd.cut(df["Age"], bins=[17,30,40,50,60,70],
                            labels=["18-30","31-40","41-50","51-60","61+"])
    df["BMICategory"] = pd.cut(df["BMI"],
                               bins=[0,18.5,25,30,100],
                               labels=["Underweight","Normal","Overweight","Obese"])
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 1. DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def plot_distributions(df):
    fig = plt.figure(figsize=(20, 14), facecolor="#0f0f1a")
    fig.suptitle("Distribution Analysis — Insurance Dataset", fontsize=22,
                 fontweight="bold", color=ACCENT, y=0.98)
    
    num_cols = ["Age", "Height", "Weight", "BMI", "PremiumPrice", "NumberOfMajorSurgeries"]
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    
    for i, col in enumerate(num_cols):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        color = PALETTE[i]
        # KDE + histogram
        ax.hist(df[col], bins=25, color=color, alpha=0.35, density=True, edgecolor="none")
        df[col].plot.kde(ax=ax, color=color, linewidth=2.5)
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="#ffd93d", linestyle="--", linewidth=1.5, label=f"Mean {mean_val:.1f}")
        ax.axvline(median_val, color="#ff6b9d", linestyle=":", linewidth=1.5, label=f"Median {median_val:.1f}")
        ax.set_title(col, fontsize=13, fontweight="bold", color=color, pad=8)
        ax.legend(fontsize=8, framealpha=0.3)
        ax.set_ylabel("Density", fontsize=9)
        skew = df[col].skew()
        ax.text(0.97, 0.92, f"Skew: {skew:.2f}", transform=ax.transAxes,
                fontsize=8, ha="right", color="#aaa")
    
    plt.savefig("dist_numerical.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: dist_numerical.png")


def plot_binary_distributions(df):
    binary_cols = ["Diabetes","BloodPressureProblems","AnyTransplants",
                   "AnyChronicDiseases","KnownAllergies","HistoryOfCancerInFamily"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0f0f1a")
    fig.suptitle("Binary Health Condition Distributions", fontsize=20,
                 fontweight="bold", color=ACCENT, y=0.99)
    
    for ax, col, color in zip(axes.flatten(), binary_cols, PALETTE):
        counts = df[col].value_counts()
        wedges, texts, autotexts = ax.pie(
            counts, labels=["No","Yes"], autopct="%1.1f%%",
            colors=[color+"55", color], startangle=90,
            wedgeprops=dict(edgecolor="#0f0f1a", linewidth=2),
            textprops={"color": "#e0e0e0", "fontsize": 11}
        )
        for at in autotexts:
            at.set_fontsize(12); at.set_fontweight("bold")
        ax.set_title(col, fontsize=13, fontweight="bold", color=color, pad=12)
    
    plt.tight_layout()
    plt.savefig("dist_binary.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: dist_binary.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREMIUM vs HEALTH FACTORS
# ─────────────────────────────────────────────────────────────────────────────
def plot_premium_by_factors(df):
    binary_cols = ["Diabetes","BloodPressureProblems","AnyTransplants",
                   "AnyChronicDiseases","KnownAllergies","HistoryOfCancerInFamily"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), facecolor="#0f0f1a")
    fig.suptitle("Premium Price Distribution by Health Conditions", fontsize=20,
                 fontweight="bold", color=ACCENT, y=0.99)
    
    labels = ["No","Yes"]
    for ax, col, color in zip(axes.flatten(), binary_cols, PALETTE):
        data0 = df[df[col] == 0]["PremiumPrice"]
        data1 = df[df[col] == 1]["PremiumPrice"]
        bp = ax.boxplot([data0, data1], labels=labels, patch_artist=True,
                        medianprops=dict(color="#ffd93d", linewidth=2.5),
                        whiskerprops=dict(color="#aaa"),
                        capprops=dict(color="#aaa"),
                        flierprops=dict(marker="o", color=color, alpha=0.5, markersize=4))
        bp["boxes"][0].set_facecolor(color + "33")
        bp["boxes"][1].set_facecolor(color + "99")
        
        # Overlay violin
        parts = ax.violinplot([data0, data1], positions=[1, 2],
                               showmeans=False, showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color); pc.set_alpha(0.15)
        
        ax.set_title(col, fontsize=12, fontweight="bold", color=color)
        ax.set_ylabel("Premium Price (₹)", fontsize=9)
        
        # t-test annotation
        t, p = ttest_ind(data0, data1)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(0.5, 0.96, f"p={p:.4f} {sig}", transform=ax.transAxes,
                ha="center", fontsize=9, color="#ffd93d", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig("premium_by_conditions.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: premium_by_conditions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def plot_correlation_heatmap(df):
    num_df = df[["Age","Height","Weight","BMI","NumberOfMajorSurgeries",
                 "Diabetes","BloodPressureProblems","AnyTransplants",
                 "AnyChronicDiseases","KnownAllergies","HistoryOfCancerInFamily",
                 "PremiumPrice"]]
    corr = num_df.corr()
    
    fig, ax = plt.subplots(figsize=(14, 11), facecolor="#0f0f1a")
    fig.suptitle("Correlation Heatmap — All Features", fontsize=18,
                 fontweight="bold", color=ACCENT, y=0.98)
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, fmt=".2f",
                linewidths=0.5, linecolor="#0f0f1a",
                annot_kws={"size": 9, "color": "#e0e0e0"},
                cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(labelsize=10, colors="#ccc")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SCATTER PLOTS — AGE, BMI vs PREMIUM
# ─────────────────────────────────────────────────────────────────────────────
def plot_scatter_analysis(df):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="#0f0f1a")
    fig.suptitle("Scatter Analysis: Continuous Features vs Premium Price",
                 fontsize=18, fontweight="bold", color=ACCENT, y=1.01)
    
    pairs = [("Age", "#00d4ff"), ("BMI", "#ff6b9d"), ("Weight", "#ffd93d")]
    for ax, (feat, color), in zip(axes, pairs):
        sc = ax.scatter(df[feat], df["PremiumPrice"],
                        c=df["PremiumPrice"], cmap="plasma",
                        alpha=0.65, s=30, edgecolors="none")
        # Regression line
        m, b, r, p, _ = stats.linregress(df[feat], df["PremiumPrice"])
        x_line = np.linspace(df[feat].min(), df[feat].max(), 200)
        ax.plot(x_line, m * x_line + b, color=color, linewidth=2.5,
                label=f"r={r:.3f}, p={p:.4f}")
        ax.set_xlabel(feat, fontsize=12, color="#ccc")
        ax.set_ylabel("Premium Price (₹)", fontsize=12, color="#ccc")
        ax.set_title(f"{feat} vs Premium", fontsize=14, fontweight="bold", color=color)
        ax.legend(fontsize=10, framealpha=0.3)
        plt.colorbar(sc, ax=ax, label="Premium Price", shrink=0.85)
    
    plt.tight_layout()
    plt.savefig("scatter_vs_premium.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: scatter_vs_premium.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. SURGICAL IMPACT
# ─────────────────────────────────────────────────────────────────────────────
def plot_surgery_impact(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0f0f1a")
    fig.suptitle("Impact of Major Surgeries on Premium Price",
                 fontsize=18, fontweight="bold", color=ACCENT, y=1.01)
    
    # Boxplot
    groups = [df[df["NumberOfMajorSurgeries"] == i]["PremiumPrice"] for i in range(4)]
    bp = axes[0].boxplot(groups, labels=[f"{i} surgeries" for i in range(4)],
                         patch_artist=True,
                         medianprops=dict(color="#ffd93d", linewidth=2.5),
                         whiskerprops=dict(color="#aaa"),
                         capprops=dict(color="#aaa"))
    for box, color in zip(bp["boxes"], PALETTE):
        box.set_facecolor(color + "66")
    axes[0].set_ylabel("Premium Price (₹)", fontsize=11)
    axes[0].set_title("Distribution by Surgery Count", fontsize=13, color="#00d4ff")
    
    # Mean bar chart
    mean_premium = df.groupby("NumberOfMajorSurgeries")["PremiumPrice"].mean()
    bars = axes[1].bar(mean_premium.index, mean_premium.values,
                       color=PALETTE[:len(mean_premium)], edgecolor="#0f0f1a", linewidth=1.5)
    for bar, val in zip(bars, mean_premium.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                     f"₹{val:,.0f}", ha="center", va="bottom", fontsize=10,
                     color="#ffd93d", fontweight="bold")
    axes[1].set_xlabel("Number of Major Surgeries", fontsize=11)
    axes[1].set_ylabel("Average Premium Price (₹)", fontsize=11)
    axes[1].set_title("Mean Premium by Surgery Count", fontsize=13, color="#ff6b9d")
    axes[1].set_xticks(range(4))
    
    plt.tight_layout()
    plt.savefig("surgery_impact.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: surgery_impact.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. AGE GROUP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def plot_age_group_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), facecolor="#0f0f1a")
    fig.suptitle("Age Group Analysis", fontsize=18, fontweight="bold", color=ACCENT, y=1.01)
    
    age_premium = df.groupby("AgeGroup", observed=True)["PremiumPrice"].mean()
    bars = axes[0].bar(age_premium.index, age_premium.values,
                       color=PALETTE[:len(age_premium)], edgecolor="#0f0f1a")
    for bar, val in zip(bars, age_premium.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                     f"₹{val:,.0f}", ha="center", va="bottom",
                     fontsize=9, color="#ffd93d", fontweight="bold")
    axes[0].set_xlabel("Age Group", fontsize=11)
    axes[0].set_ylabel("Mean Premium Price (₹)", fontsize=11)
    axes[0].set_title("Average Premium by Age Group", fontsize=13, color="#00d4ff")
    
    # Stacked bar: health conditions by age group
    binary_cols = ["Diabetes","BloodPressureProblems","AnyChronicDiseases","AnyTransplants"]
    age_health = df.groupby("AgeGroup", observed=True)[binary_cols].mean()
    age_health.plot(kind="bar", ax=axes[1], color=PALETTE[:4],
                    edgecolor="#0f0f1a", width=0.75)
    axes[1].set_xlabel("Age Group", fontsize=11)
    axes[1].set_ylabel("Prevalence Rate", fontsize=11)
    axes[1].set_title("Health Condition Prevalence by Age", fontsize=13, color="#ff6b9d")
    axes[1].legend(fontsize=9, framealpha=0.3, loc="upper left")
    axes[1].tick_params(axis="x", rotation=0)
    
    plt.tight_layout()
    plt.savefig("age_group_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: age_group_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. OUTLIER DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def plot_outliers(df):
    num_cols = ["Age","Height","Weight","BMI","PremiumPrice"]
    fig, axes = plt.subplots(1, len(num_cols), figsize=(22, 7), facecolor="#0f0f1a")
    fig.suptitle("Outlier Detection via IQR Method", fontsize=18,
                 fontweight="bold", color=ACCENT, y=1.01)
    
    print("\n" + "="*60)
    print("OUTLIER DETECTION SUMMARY (IQR Method)")
    print("="*60)
    
    for ax, col, color in zip(axes, num_cols, PALETTE):
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        
        bp = ax.boxplot(df[col], vert=True, patch_artist=True,
                        medianprops=dict(color="#ffd93d", linewidth=2.5),
                        whiskerprops=dict(color="#aaa"),
                        capprops=dict(color="#aaa"),
                        flierprops=dict(marker="o", color=color, alpha=0.7, markersize=5))
        bp["boxes"][0].set_facecolor(color + "55")
        ax.set_title(col, fontsize=12, fontweight="bold", color=color, pad=8)
        ax.set_ylabel("Value", fontsize=9)
        ax.text(0.5, 0.02, f"Outliers: {len(outliers)}", transform=ax.transAxes,
                ha="center", fontsize=10, color="#ff6b9d", fontweight="bold")
        
        print(f"{col:30s}: {len(outliers):3d} outliers | "
              f"Range [{lower:.1f}, {upper:.1f}]")
    
    print("="*60)
    plt.tight_layout()
    plt.savefig("outlier_detection.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: outlier_detection.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. HYPOTHESIS TESTING
# ─────────────────────────────────────────────────────────────────────────────
def run_hypothesis_tests(df):
    sep = "="*65
    print(f"\n{sep}")
    print("HYPOTHESIS TESTING RESULTS")
    print(sep)

    # ── H1: Chronic diseases → higher premiums ────────────────────────────
    print("\n📌 H1: Chronic Diseases → Higher Premiums")
    g0 = df[df["AnyChronicDiseases"] == 0]["PremiumPrice"]
    g1 = df[df["AnyChronicDiseases"] == 1]["PremiumPrice"]
    t, p = ttest_ind(g0, g1, equal_var=False)
    cohen_d = (g1.mean() - g0.mean()) / np.sqrt((g0.std()**2 + g1.std()**2) / 2)
    print(f"   H0: No difference in premium for chronic vs non-chronic patients")
    print(f"   Mean (No Chronic): ₹{g0.mean():,.0f}  |  Mean (Chronic): ₹{g1.mean():,.0f}")
    print(f"   T-statistic: {t:.4f} | p-value: {p:.6f}")
    print(f"   Cohen's d (effect size): {cohen_d:.4f}")
    print(f"   {'✅ REJECT H0' if p < 0.05 else '❌ FAIL TO REJECT H0'} (α=0.05) — "
          f"{'Significant difference exists.' if p < 0.05 else 'No significant difference.'}")

    # ── H2: Number of surgeries → ANOVA ──────────────────────────────────
    print("\n📌 H2: Number of Surgeries → Different Premium Means (ANOVA)")
    groups = [df[df["NumberOfMajorSurgeries"] == i]["PremiumPrice"] for i in range(4)]
    f, p = f_oneway(*groups)
    print(f"   H0: Mean premium is equal across all surgery groups")
    for i, g in enumerate(groups):
        print(f"   Group {i} surgeries: n={len(g)}, mean=₹{g.mean():,.0f}")
    print(f"   F-statistic: {f:.4f} | p-value: {p:.6f}")
    print(f"   {'✅ REJECT H0' if p < 0.05 else '❌ FAIL TO REJECT H0'} (α=0.05)")

    # ── H3: Diabetes → higher premiums ───────────────────────────────────
    print("\n📌 H3: Diabetes → Higher Premiums")
    g0 = df[df["Diabetes"] == 0]["PremiumPrice"]
    g1 = df[df["Diabetes"] == 1]["PremiumPrice"]
    t, p = ttest_ind(g0, g1, equal_var=False)
    print(f"   Mean (No Diabetes): ₹{g0.mean():,.0f}  |  Mean (Diabetic): ₹{g1.mean():,.0f}")
    print(f"   T-statistic: {t:.4f} | p-value: {p:.6f}")
    print(f"   {'✅ REJECT H0' if p < 0.05 else '❌ FAIL TO REJECT H0'} (α=0.05)")

    # ── H4: Transplants → higher premiums ────────────────────────────────
    print("\n📌 H4: Transplants → Higher Premiums")
    g0 = df[df["AnyTransplants"] == 0]["PremiumPrice"]
    g1 = df[df["AnyTransplants"] == 1]["PremiumPrice"]
    t, p = ttest_ind(g0, g1, equal_var=False)
    print(f"   Mean (No Transplant): ₹{g0.mean():,.0f}  |  Mean (Transplant): ₹{g1.mean():,.0f}")
    print(f"   T-statistic: {t:.4f} | p-value: {p:.6f}")
    print(f"   {'✅ REJECT H0' if p < 0.05 else '❌ FAIL TO REJECT H0'} (α=0.05)")

    # ── H5: Chi-Square — Chronic Disease ↔ Cancer History ────────────────
    print("\n📌 H5: Chi-Square — Chronic Disease ↔ Cancer Family History")
    ct = pd.crosstab(df["AnyChronicDiseases"], df["HistoryOfCancerInFamily"])
    chi2, p, dof, expected = chi2_contingency(ct)
    print(f"   Contingency Table:\n{ct.to_string()}")
    print(f"   Chi2: {chi2:.4f} | df: {dof} | p-value: {p:.6f}")
    print(f"   {'✅ REJECT H0' if p < 0.05 else '❌ FAIL TO REJECT H0'} — "
          f"{'Association exists.' if p < 0.05 else 'No significant association.'}")

    # ── H6: Pearson Correlation — Age ↔ Premium ──────────────────────────
    print("\n📌 H6: Pearson Correlation — Age ↔ Premium Price")
    r, p = pearsonr(df["Age"], df["PremiumPrice"])
    print(f"   r = {r:.4f} | p-value = {p:.6f}")
    print(f"   {'✅ SIGNIFICANT' if p < 0.05 else '❌ NOT SIGNIFICANT'} linear correlation")

    # ── H7: BMI ↔ Premium ────────────────────────────────────────────────
    print("\n📌 H7: Pearson Correlation — BMI ↔ Premium Price")
    r, p = pearsonr(df["BMI"], df["PremiumPrice"])
    print(f"   r = {r:.4f} | p-value = {p:.6f}")
    print(f"   {'✅ SIGNIFICANT' if p < 0.05 else '❌ NOT SIGNIFICANT'} linear correlation")

    print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 9. HYPOTHESIS VISUAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def plot_hypothesis_summary(df):
    binary_cols = ["Diabetes","BloodPressureProblems","AnyTransplants",
                   "AnyChronicDiseases","KnownAllergies","HistoryOfCancerInFamily"]
    results = []
    for col in binary_cols:
        g0 = df[df[col] == 0]["PremiumPrice"]
        g1 = df[df[col] == 1]["PremiumPrice"]
        t, p = ttest_ind(g0, g1, equal_var=False)
        effect = (g1.mean() - g0.mean()) / g0.mean() * 100
        results.append({"Feature": col, "p_value": p, "PctIncrease": effect, "Sig": p < 0.05})
    
    res_df = pd.DataFrame(results).sort_values("PctIncrease", ascending=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor="#0f0f1a")
    fig.suptitle("Hypothesis Testing — Premium Impact Summary",
                 fontsize=18, fontweight="bold", color=ACCENT, y=1.01)
    
    # P-values
    colors = ["#6bcb77" if s else "#ff6b9d" for s in res_df["Sig"]]
    axes[0].barh(res_df["Feature"], -np.log10(res_df["p_value"].clip(1e-10)),
                 color=colors, edgecolor="#0f0f1a")
    axes[0].axvline(-np.log10(0.05), color="#ffd93d", linestyle="--",
                    linewidth=2, label="α=0.05 threshold")
    axes[0].set_xlabel("-log10(p-value)", fontsize=11)
    axes[0].set_title("Statistical Significance", fontsize=13, color="#00d4ff")
    axes[0].legend(fontsize=10, framealpha=0.3)
    
    # % Increase
    colors2 = [PALETTE[i % len(PALETTE)] for i in range(len(res_df))]
    axes[1].barh(res_df["Feature"], res_df["PctIncrease"],
                 color=colors2, edgecolor="#0f0f1a")
    for i, (idx, row) in enumerate(res_df.iterrows()):
        axes[1].text(row["PctIncrease"] + 0.3, i, f"+{row['PctIncrease']:.1f}%",
                     va="center", fontsize=9, color="#ffd93d", fontweight="bold")
    axes[1].set_xlabel("Premium Increase (%) for Condition=1", fontsize=11)
    axes[1].set_title("Effect Size on Premium", fontsize=13, color="#ff6b9d")
    
    plt.tight_layout()
    plt.savefig("hypothesis_summary.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
    plt.close()
    print("✔ Saved: hypothesis_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*65)
    print("BLOCK 2 — EDA & HYPOTHESIS TESTING")
    print("="*65)
    
    df = load_data()
    print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\n── Basic Info ──")
    print(df.describe().round(2).to_string())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    print("\n── Generating Plots ──")
    plot_distributions(df)
    plot_binary_distributions(df)
    plot_premium_by_factors(df)
    plot_correlation_heatmap(df)
    plot_scatter_analysis(df)
    plot_surgery_impact(df)
    plot_age_group_analysis(df)
    plot_outliers(df)
    plot_hypothesis_summary(df)
    run_hypothesis_tests(df)
    
    print("\n✅ Block 2 complete! All plots saved as PNG files.")
