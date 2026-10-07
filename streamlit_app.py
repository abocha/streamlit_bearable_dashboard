# streamlit_app.py
import os

import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import timedelta, date
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

# ───────────────────────────────────────────────
# 0. Page config & Constants
# ───────────────────────────────────────────────
st.set_page_config(page_title="Bearable Mood & Symptom Dashboard", layout="wide")

# --- Define the Key Symptom List (Derived from 60-day analysis) ---
FINAL_KEY_SYMPTOM_COLS = [
    "Depression", "Fidgeting", "Forgetfulness", "Hyperfocused on the RIGHT thing",
    "Impulsivity", "Irritability", "Mental Restlessness", "Mood swings",
    "Pessimism", "Physical restlessness", "Racing thoughts", "Random energy spike",
    "Seeking stimulation", "Shutdown mode", "Sudden energy drop"
]

# ───────────────────────────────────────────────
# 1. Utility: scrub & feature engineer
# ───────────────────────────────────────────────

# === Nutrition Helpers ===
def parse_amount(detail):
    m = re.search(r'Amount eaten\s*[-–]\s*(\w+)', str(detail))
    return {'Little':1, 'Moderate':2, 'A lot':3}.get(m.group(1), np.nan) if m else np.nan

def count_meals(detail):
    s = str(detail)
    if s.startswith('Meals:'):
        return sum(1 for part in s.replace('Meals:','', 1).split('|') if part.strip())
    return 0

CAL_MAP = {
    'Coffee':5, 'Instant noodles':400, 'Shawarma':600, 'Banana':100, 'Yogurt':150,
    'Banana and yogurt':250, 'Chicken':200, 'Chicken and rice':500,
    'Chicken Salad':300, 'Bánh Mì':550, 'Burrito':500, 'Chips':150,
    'Chocolate':200, 'Chocopie':100, 'Coke':140, 'Egg and sausage':250,
    'KFC':800, 'Nuts':200, 'Oreo':50, 'Pizza':300, 'Smoothie':200,
    'Snickers':250, 'Waffle':250
}

def estimate_cal(detail):
    s = str(detail)
    if s.startswith('Meals:'):
        return sum(CAL_MAP.get(item.strip(), 0) for item in s.replace('Meals:','', 1).split('|') if item.strip())
    return 0
# === End Nutrition Helpers ===

# === Core Load & Clean (Accepts DataFrame) ===
@st.cache_data
def load_and_clean(input_df: pd.DataFrame) -> pd.DataFrame:
    """Cleans and pivots data from an already loaded Bearable DataFrame."""
    df = input_df.copy()
    if "date formatted" not in df.columns or "rating/amount" not in df.columns:
         st.error("Input DataFrame is missing required columns ('date formatted', 'rating/amount').")
         return pd.DataFrame()

    df.columns = [c.strip().replace('"', "") for c in df.columns]
    df["date"] = pd.to_datetime(df["date formatted"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating/amount"], errors="coerce")
    df = df.dropna(subset=["date"])

    unique_dates = pd.to_datetime(df["date"].dt.date.unique())
    if not unique_dates.empty:
        cal = pd.DataFrame(index=unique_dates)
        cal.index.name = "date"
    else:
        st.warning("No valid dates found after initial cleaning.")
        return pd.DataFrame()

    energy_map = {"v. low": 1, "low": 2, "ok": 3, "high": 4, "v. high": 5}
    qual_map = {"poor": 1, "ok": 2, "good": 3, "great": 4}
    sleep_pat = re.compile(
        r"Asleep\s+(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})", re.I
    )

    def hhmm_to_hours(s):
        if pd.isna(s): return np.nan
        m = re.match(r"^(\d{1,2}):(\d{2})$", str(s).strip())
        return np.nan if not m else int(m[1]) + int(m[2]) / 60

    def asleep_hours(detail):
        if pd.isna(detail): return (np.nan, np.nan)
        m = sleep_pat.search(str(detail))
        if not m: return (np.nan, np.nan)
        h1, m1, h2, m2 = map(int, m.groups())
        start = timedelta(hours=h1, minutes=m1)
        end = timedelta(hours=h2, minutes=m2)
        if end < start: end += timedelta(days=1)
        duration = (end - start).total_seconds() / 3600
        bed_hour = h1
        return (duration, bed_hour)

    # Process categories
    for category, process_func in [
        ("Mood", lambda df_cat: df_cat.groupby(df_cat["date"].dt.date)["rating"].mean().rename("average_mood")),
        ("Energy", lambda df_cat: df_cat.assign(val=df_cat.detail.str.lower().map(energy_map).fillna(df_cat["rating"])).groupby(df_cat["date"].dt.date)["val"].mean().rename("average_energy")),
        ("Sleep quality", lambda df_cat: df_cat.assign(qnum=df_cat["rating"].fillna(df_cat.detail.str.lower().map(qual_map))).groupby(df_cat["date"].dt.date)["qnum"].mean().rename("sleep_quality_score"))
    ]:
        if category in df['category'].unique():
            cat_df = df[df.category == category].dropna(subset=["rating"])
            if not cat_df.empty:
                cal = cal.join(process_func(cat_df))
        else:
             if category == "Mood": cal['average_mood'] = np.nan
             if category == "Energy": cal['average_energy'] = np.nan
             if category == "Sleep quality": cal['sleep_quality_score'] = np.nan

    # Sleep duration and bedtime
    if 'Sleep' in df['category'].unique():
        sl_df = df[df.category == 'Sleep'].copy()
        if not sl_df.empty:
            sl_df['hours_num'] = sl_df['rating'].apply(hhmm_to_hours)
            sleep_details = sl_df['detail'].apply(asleep_hours).apply(pd.Series)
            sleep_details.columns = ['hours_det', 'bed_hour']
            sl_df = pd.concat([sl_df, sleep_details], axis=1)
            sl_df['sleep_duration_hours'] = sl_df['hours_num'].fillna(sl_df['hours_det'])
            sl_df['late_bedtime_flag'] = (sl_df['bed_hour'] > 1).astype(int)
            sleep_agg = sl_df.groupby(sl_df['date'].dt.date).agg(
                sleep_duration_hours=('sleep_duration_hours', 'first'),
                late_bedtime_flag=('late_bedtime_flag', 'max'),
                bed_hour=('bed_hour', 'first')
            )
            cal = cal.join(sleep_agg)
    else:
        cal['sleep_duration_hours'] = np.nan
        cal['late_bedtime_flag'] = np.nan
        cal['bed_hour'] = np.nan

    # Symptoms
    if 'Symptom' in df['category'].unique():
        sym_df = df[df.category == "Symptom"].copy()
        if not sym_df.empty:
            sym_df["name"] = sym_df.detail.str.replace(
                r"\s*\((?:Mild|Moderate|Severe|Unbearable|Extreme)\)", "", regex=True
            ).str.strip()
            sym_df["sev"] = pd.to_numeric(sym_df["rating"], errors='coerce').astype("Int64").clip(1, 4)
            piv = sym_df.groupby([sym_df["date"].dt.date, "name"])["sev"].max().unstack()
            cal = cal.join(piv)
            if not piv.empty:
                 cal["TotalSymptomScore"] = piv.fillna(0).sum(axis=1)
            else: cal["TotalSymptomScore"] = 0
        else: cal["TotalSymptomScore"] = 0
    else: cal["TotalSymptomScore"] = 0

    # Final cleanup
    cal = cal.astype(float)
    cal = cal.loc[:, cal.notna().sum() >= 3]
    return cal.sort_index()
# === End Load & Clean ===


# === Feature Adder ===
@st.cache_data
def add_features(df: pd.DataFrame, sleep_thr: float) -> pd.DataFrame:
    """Adds rolling averages, flags, and other derived features."""
    if df.empty: return df
    df2 = df.copy()
    for col in ['average_mood', 'sleep_duration_hours']:
        if col not in df2.columns: df2[col] = np.nan

    df2["mood_7d_ma"] = df2["average_mood"].rolling(7, min_periods=1).mean()
    df2["mood_delta"] = df2["average_mood"] - df2["mood_7d_ma"]
    df2["is_weekend"] = (df2.index.to_series().dt.weekday.isin([5, 6])).astype(int)

    if 'sleep_duration_hours' in df2.columns:
        df2["flag_sleep_predict"] = (
            (df2["sleep_duration_hours"] < sleep_thr).shift(2).fillna(False).astype(int)
        )
    else: df2["flag_sleep_predict"] = 0
    return df2
# === End Feature Adder ===


# ───────────────────────────────────────────────
# 2. Sidebar controls
# ───────────────────────────────────────────────
st.sidebar.header("⚙️ Heuristic Flag Thresholds")
energy_thr = st.sidebar.slider("Energy < threshold triggers flag", 1.0, 5.0, 3.0, 0.5, key="energy_thr_slider")
sleep_thr = st.sidebar.slider("Sleep hours threshold (<) for lagged t+2 comparison", 4.0, 8.0, 6.0, 0.5, key="sleep_thr_slider")
var_thr = st.sidebar.slider(
    "Sleep variability flag (7‑day SD > … hours)", 0.0, 3.0, 1.0, 0.1, key="var_thr_slider"
)
key_symptom_use_std = st.sidebar.checkbox(
    "Flag Key Symptoms using median + 1 SD (else median)", value=True, key="key_symptom_std_check"
)

# ───────────────────────────────────────────────
# 3. Load CSV (Read Once Logic)
# ───────────────────────────────────────────────
st.title("🐰 Exploratory Bearable Data Dashboard")
st.markdown("Explore mood, energy, sleep, nutrition, and symptom patterns from a Bearable CSV export. Flags are user-defined heuristics, not medical predictions.")

export_dir_value = os.environ.get("BEARABLE_EXPORT_DIR")
EXPORT_DIR = Path(export_dir_value).expanduser() if export_dir_value else None
pattern = re.compile(
    r"Bearable App - Data Export\. Generated (\d{2}-\d{2}-\d{4})\.csv"
)
latest_file = None
source_info = None
source_object = None

if EXPORT_DIR and EXPORT_DIR.exists() and EXPORT_DIR.is_dir():
    matches = []
    for file in EXPORT_DIR.glob("*.csv"):
        m = pattern.match(file.name)
        if m:
            try:
                dt = pd.to_datetime(m.group(1), format="%d-%m-%Y", errors='coerce')
                if pd.notna(dt): matches.append((dt, file))
            except ValueError: continue
    if matches:
        latest_file = sorted(matches, key=lambda x: x[0], reverse=True)[0][1]
        source_info = latest_file.name
        source_object = latest_file

if latest_file:
    st.success(f"Auto‑loaded: {source_info}")
else:
    if EXPORT_DIR:
        st.sidebar.info("No matching export found in BEARABLE_EXPORT_DIR. Upload a CSV manually instead.")
    else:
        st.sidebar.info("Set BEARABLE_EXPORT_DIR for optional auto-load, or upload a CSV manually.")
    uploaded_file = st.sidebar.file_uploader("Upload Bearable CSV", type="csv", key="manual_upload")
    if uploaded_file:
        source_info = uploaded_file.name
        source_object = uploaded_file
    else:
        st.warning("No auto‑load file found – please upload manually via the sidebar ⬆️")
        st.stop()

try:
    df_raw_unprocessed = pd.read_csv(source_object)
    df_cleaned = load_and_clean(df_raw_unprocessed)
except Exception as e:
    st.error(f"Failed to load or process data from {source_info}: {e}")
    if isinstance(e, pd.errors.EmptyDataError): st.error("The CSV file appears to be empty.")
    elif isinstance(e, UnicodeDecodeError): st.error("Could not decode the file. Check if it's a valid UTF-8 CSV.")
    st.stop()

if df_cleaned.empty:
    st.error(f"No processable data found after cleaning {source_info}. Potential issues: incorrect format, missing essential columns ('date formatted', 'rating/amount'), or insufficient data per column.")
    st.stop()

# ───────────────────────────────────────────────
# 4. Feature engineering & flags
# ───────────────────────────────────────────────
df_feat = add_features(df_cleaned, sleep_thr)

if 'sleep_duration_hours' in df_feat.columns:
    df_feat['sleep_std_7d'] = df_feat['sleep_duration_hours'].rolling(7, min_periods=1).std().fillna(0)
    df_feat['flag_sleep_var'] = (df_feat['sleep_std_7d'] > var_thr).astype(int)
else:
    df_feat['sleep_std_7d'] = 0
    df_feat['flag_sleep_var'] = 0

if 'average_energy' in df_feat.columns:
    df_feat["flag_low_energy"] = (df_feat["average_energy"] < energy_thr).astype(int)
else: df_feat["flag_low_energy"] = 0

# --- Key Symptom Features ---
st.sidebar.info(f"Using pre-defined Key Symptom group ({len(FINAL_KEY_SYMPTOM_COLS)} symptoms)")
actual_key_symptom_cols = [col for col in FINAL_KEY_SYMPTOM_COLS if col in df_feat.columns]
missing_key_cols = set(FINAL_KEY_SYMPTOM_COLS) - set(actual_key_symptom_cols)
if missing_key_cols:
    st.warning(f"Note: The following Key Symptoms were not found in the current data: {missing_key_cols}")

if actual_key_symptom_cols:
    df_feat["key_symptom_sum"] = df_feat[actual_key_symptom_cols].sum(axis=1)
    key_sum_series = df_feat["key_symptom_sum"]
    thr_key = key_sum_series.median() + (key_sum_series.std() if key_symptom_use_std else 0)
    # Ensure threshold is numeric before comparison
    if pd.notna(thr_key):
        df_feat["flag_key_symptoms"] = (key_sum_series > thr_key).astype(int)
    else:
        st.warning("Could not calculate threshold for Key Symptoms (median/std might be NaN). Flag set to 0.")
        df_feat["flag_key_symptoms"] = 0
else:
     df_feat["key_symptom_sum"] = 0
     df_feat["flag_key_symptoms"] = 0
# --- End Key Symptom Features ---

# --- Nutrition Features ---
df_feat[['daily_calories','evening_num_items']] = 0
if 'Nutrition' in df_raw_unprocessed['category'].unique():
    df_raw_unprocessed['date'] = pd.to_datetime(df_raw_unprocessed['date formatted'], errors='coerce').dt.date
    nut = df_raw_unprocessed[df_raw_unprocessed['category'] == 'Nutrition'].copy()
    if not nut.empty:
        nut['nutrition_amount'] = nut['detail'].apply(parse_amount)
        nut['nutrition_num_items'] = nut['detail'].apply(count_meals)
        nut['calories'] = nut['detail'].apply(estimate_cal)
        if isinstance(df_feat.index, pd.DatetimeIndex): df_feat_index_date = df_feat.index.date
        else: df_feat_index_date = df_feat.index
        agg = nut.groupby('date')[['nutrition_amount','nutrition_num_items']].agg(
            nutrition_amount=('nutrition_amount', 'max'), nutrition_num_items=('nutrition_num_items', 'sum')
        ).reindex(df_feat_index_date)
        df_feat = df_feat.join(agg, how='left')
        daily_cal = nut.groupby('date')['calories'].sum().reindex(df_feat_index_date, fill_value=0)
        df_feat['daily_calories'] = daily_cal
        pm = nut[nut['time of day']=='pm'].copy()
        pm['pm_items'] = pm['detail'].apply(count_meals)
        evening_items = pm.groupby(pm['date'])['pm_items'].sum().reindex(df_feat_index_date, fill_value=0)
        df_feat['evening_num_items'] = evening_items
        nut_cols_to_fill = ['nutrition_amount', 'nutrition_num_items', 'daily_calories', 'evening_num_items']
        for col in nut_cols_to_fill:
            if col in df_feat.columns: df_feat[col] = df_feat[col].fillna(0)
        median_cal = df_feat['daily_calories'].median()
        if pd.notna(median_cal) and median_cal > 0:
            df_feat['flag_low_cal'] = (df_feat['daily_calories'] < median_cal).astype(int)
        else: df_feat['flag_low_cal'] = 0
        df_feat['flag_few_items'] = (df_feat['nutrition_num_items'] < 3).astype(int)
        df_feat['flag_no_pm_snack'] = (df_feat['evening_num_items'] == 0).astype(int)
    else: df_feat[['nutrition_amount', 'nutrition_num_items', 'flag_low_cal', 'flag_few_items', 'flag_no_pm_snack']] = 0
else: df_feat[['nutrition_amount', 'nutrition_num_items', 'flag_low_cal', 'flag_few_items', 'flag_no_pm_snack']] = 0
# --- End Nutrition Features ---

# ───────────────────────────────────────────────
# 5. Today’s alerts
# ───────────────────────────────────────────────
today_dt = pd.to_datetime(date.today())
st.subheader(f"🚩 Today's Heuristic Flags ({today_dt.date()})")

# Define alerts with ENHANCED descriptions
alerts = [
    (
        "🔋 Low Energy", 'flag_low_energy',
        f"Energy < {energy_thr:.1f}. A user-defined marker for reviewing same-day context; it is not a prediction."
    ),
    (
        "💤 Short Sleep (t-2)", 'flag_sleep_predict',
        f"Slept < {sleep_thr:.1f}h two days earlier. Marks days for lagged comparison with later observations; no causal claim is implied."
    ),
    (
        "📈 Sleep Var ↑", 'flag_sleep_var',
        f"7-day sleep-duration SD > {var_thr:.1f}h. Highlights unusually variable sleep for exploratory review."
    ),
    (
        "⚠️ Key Symptoms ↑", 'flag_key_symptoms',
        "Combined key-symptom score is above its historical threshold in this dataset. Treat it as an unusual-pattern marker."
    ),
    (
        "⛽ Low Logged Fuel", 'flag_low_cal',
        "Estimated logged calories are below this dataset's median. Estimates are coarse and depend on a small food lookup table."
    ),
    (
        "🥄 Few Logged Items", 'flag_few_items',
        "< 3 distinct food items logged. This may reflect intake, variety, or simply incomplete logging."
    ),
    (
        "🌙 No PM Food Logged", 'flag_no_pm_snack',
        "No food items were logged in the evening. Treat this as a context/logging marker, not a health recommendation."
    )
]


if today_dt in df_feat.index:
    available_alerts = [(label, flag, desc) for label, flag, desc in alerts if flag in df_feat.columns]
    if available_alerts:
        # Adjust columns based on number of alerts, maybe max 4 per row
        num_cols = min(len(available_alerts), 4)
        cols = st.columns(num_cols)
        today_data = df_feat.loc[today_dt]
        col_idx = 0
        for i, (label, flag, desc) in enumerate(available_alerts):
             with cols[col_idx]:
                is_triggered = bool(today_data[flag])
                st.metric(
                    label,
                    "ALERT" if is_triggered else "OK",
                    delta="Review Below" if is_triggered else " ", # Use delta for prompt?
                    delta_color="inverse" if is_triggered else "off",
                    help=desc # Tooltip shows the detailed insight
                )
             col_idx = (col_idx + 1) % num_cols # Cycle through columns
    else: st.info("No alert flags calculable for today.")
else: st.info(f"No data available for {today_dt.date()} in this export.")

# ───────────────────────────────────────────────
# 6. Charts
# ───────────────────────────────────────────────
st.subheader("Key Relationships & Trends")
c1, c2, c3 = st.columns(3)

with c1:
    st.caption("Energy ➜ Mood (same day)")
    if 'average_energy' in df_feat.columns and 'average_mood' in df_feat.columns:
        plot_data = df_feat[['average_energy', 'average_mood']].dropna()
        if not plot_data.empty:
            fig1, ax1 = plt.subplots()
            sns.regplot(x="average_energy", y="average_mood", data=plot_data, ax=ax1,
                        scatter_kws={"s": 30, "alpha": 0.6}, line_kws={"color": "red"}, ci=None)
            ax1.set_title("Energy vs Mood")
            st.pyplot(fig1)
        else: st.info("Not enough data for Energy vs Mood plot.")
    else: st.info("Energy or Mood data missing for plot.")

with c2:
    st.caption(f"Sleep < {sleep_thr:.1f}h ➜ Mood (t+2)")
    if 'sleep_duration_hours' in df_feat.columns and 'average_mood' in df_feat.columns:
        df_p = df_feat.dropna(subset=["sleep_duration_hours", "average_mood"]).copy()
        if not df_p.empty:
            df_p["sleep_flag"] = np.where(df_p["sleep_duration_hours"] < sleep_thr, f"<{sleep_thr:.0f}h", f"≥{sleep_thr:.0f}h")
            df_p["mood_t2"] = df_p["average_mood"].shift(-2)
            df_p = df_p.dropna(subset=["mood_t2"])
            if not df_p.empty and len(df_p['sleep_flag'].unique()) > 1: # Need both groups for boxplot
                fig2, ax2 = plt.subplots()
                sns.boxplot(x="sleep_flag", y="mood_t2", data=df_p, ax=ax2, order=[f"<{sleep_thr:.0f}h", f"≥{sleep_thr:.0f}h"])
                ax2.set_ylabel("Mood two days later")
                ax2.set_xlabel("Sleep Duration (t)")
                ax2.set_title("Sleep vs Mood (t+2)")
                st.pyplot(fig2)
            else: st.info("Not enough data or variance for Sleep vs Mood (t+2) boxplot.")
        else: st.info("Not enough data for Sleep vs Mood plot.")
    else: st.info("Sleep or Mood data missing for plot.")

# (Corrected code for Section 6, column c3)
with c3:
    st.caption("Key Symptom Score Over Time (with Trend)")
    if 'key_symptom_sum' in df_feat.columns and df_feat['key_symptom_sum'].notna().any():
        chart_data = df_feat[['key_symptom_sum']].reset_index().dropna()

        if not chart_data.empty:
            base = alt.Chart(chart_data).encode(
                x=alt.X('date:T', title='Date')
            )

            raw_line = base.mark_line(point=True, opacity=0.7, color='lightblue').encode(
                y=alt.Y('key_symptom_sum:Q', title='Key Symptom Score'), # Title defined here
                tooltip=[
                    alt.Tooltip('date:T', title='Date'),
                    alt.Tooltip('key_symptom_sum:Q', title='Score', format='.1f')
                ]
            )

            trend_line = base.transform_loess(
                'date',
                'key_symptom_sum',
                bandwidth=0.3
            ).mark_line(color='red', strokeDash=[5,5], size=2).encode(
                # *** ADD THIS Y-ENCODING FOR THE TRANSFORMED DATA ***
                # Use the same column name as input; LOESS output retains it.
                # No need to repeat the title as it's shared with raw_line.
                y=alt.Y('key_symptom_sum:Q')
            )

            combined_chart = alt.layer(raw_line, trend_line).properties(
                title='Daily Key Symptom Score and LOESS Trend'
            ).interactive()

            st.altair_chart(combined_chart, use_container_width=True)
        else:
            st.info("Not enough valid data points to plot Key Symptom score trend.")
    else:
        st.info("Key Symptom score data not available for plotting.")

# ───────────────────────────────────────────────
# 7. Timeline
# ───────────────────────────────────────────────
st.subheader("Flag Timeline")

flag_colors = {
    "flag_low_energy": "#f94144",     # red
    "flag_sleep_predict": "#f9c74f",  # yellow
    "flag_sleep_var": "#277da1",      # blue
    "flag_key_symptoms": "#9c89b8",   # purple
    "flag_low_cal": "#90be6d",        # green
    "flag_few_items": "#43aa8b",      # teal
    "flag_no_pm_snack": "#577590"     # slate
}

flags_to_plot = [flag for flag in flag_colors if flag in df_feat.columns]

if flags_to_plot:
    flags_long = (
        df_feat[flags_to_plot]
        .reset_index()
        .melt(id_vars="date", var_name="flag", value_name="value")
    )
    flags_long = flags_long[flags_long["value"] > 0]

    if not flags_long.empty:
        chart = (
            alt.Chart(flags_long)
            .mark_circle(size=80, opacity=0.7)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("flag:N", title="Triggered Flag", axis=None),
                color=alt.Color(
                    "flag:N",
                    scale=alt.Scale(domain=list(flag_colors.keys()), range=list(flag_colors.values())),
                    legend=alt.Legend(title="Flag Type")
                ),
                 tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("flag:N", title="Flag")]
            )
            .properties(title='Daily Triggered Alerts')
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else: st.info("No flags were triggered in the selected data range for the timeline.")
else: st.info("No flags available to display on the timeline.")

# ───────────────────────────────────────────────
# 8. Explainers
# ───────────────────────────────────────────────
st.markdown("---") # Separator before explainers

# --- Alert Meanings Expander ---
with st.expander("❓ Flag Meanings & Caveats", expanded=True):
    num_alerts_available = sum(1 for _, flag, _ in alerts if flag in df_feat.columns) # Count only available flags
    if num_alerts_available > 0:
        cols = st.columns(min(num_alerts_available, 3)) # Max 3 columns per row
        col_idx = 0
        for i, (label, flag, desc) in enumerate(alerts):
            if flag in df_feat.columns: # Check flag exists before showing explanation
                 with cols[col_idx]:
                    st.markdown(f"**{label}**")
                    st.markdown(desc) # Display the enhanced description
                    # Add a small space below each item for better readability
                    st.markdown("<br>", unsafe_allow_html=True)
                 col_idx = (col_idx + 1) % len(cols)
    else:
        st.write("No alert types could be calculated based on the available data columns.")


# --- Key Symptom Group Explainer ---
with st.expander("🧠 Understanding the Key Symptom Group", expanded=False):
    st.markdown("""
This group contains symptoms selected from an earlier exploratory analysis of roughly 60 days of historical data. In that dataset, they showed relatively strong same-day association with lower mood and related self-reported difficulties.

The combined score is useful as a compact pattern marker for further inspection. It does **not** establish causality, diagnose a condition, or predict what will happen on a given day.
""")
    if FINAL_KEY_SYMPTOM_COLS:
        st.markdown("**Symptoms currently in this group:**")
        col1, col2 = st.columns(2)
        half = len(FINAL_KEY_SYMPTOM_COLS) // 2 + len(FINAL_KEY_SYMPTOM_COLS) % 2
        with col1:
            for symptom in FINAL_KEY_SYMPTOM_COLS[:half]: st.markdown(f"- {symptom}")
        with col2:
            if len(FINAL_KEY_SYMPTOM_COLS) > half:
                 for symptom in FINAL_KEY_SYMPTOM_COLS[half:]: st.markdown(f"- {symptom}")
    else: st.info("The Key Symptom list is not defined.")

# ───────────────────────────────────────────────
# 9. Download processed data
# ───────────────────────────────────────────────
st.markdown("---")
st.markdown("### Download Processed Data")
csv_bytes = df_feat.to_csv(index=True).encode('utf-8')
st.download_button(
    label="📥 Download Processed CSV",
    data=csv_bytes,
    file_name=f"bearable_processed_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
    mime='text/csv',
    key='download_button'
)

with st.expander("Show Processed Dataframe"):
    st.dataframe(df_feat)

# --- End of Script ---