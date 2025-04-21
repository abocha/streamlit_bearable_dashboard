# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import timedelta, date
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import re

# ── Nutrition parsing helpers ─────────────────────────────────────
def parse_amount(detail):
    m = re.search(r'Amount eaten\s*-\s*(\w+)', str(detail))
    return {'Little':1, 'Moderate':2, 'A lot':3}.get(m.group(1), np.nan) if m else np.nan

def count_meals(detail):
    s = str(detail)
    return len([i for i in s.replace('Meals:','').split('|') if i.strip()]) if s.startswith('Meals:') else 0

# Calorie estimates (extend as needed)
CAL_MAP = {
    'Coffee':5, 'Instant noodles':400, 'Shawarma':600, 'Banana':100, 'Yogurt':150,
    'Banana and yogurt':250, 'Chicken':200, 'Chicken and rice':500, 'Chicken Salad':300,
    'Bánh Mì':550, 'Burrito':500, 'Chips':150, 'Chocolate':200, 'Chocopie':100,
    'Coke':140, 'Egg and sausage':250, 'KFC':800, 'Nuts':200, 'Oreo':50,
    'Pizza':300, 'Smoothie':200, 'Snickers':250, 'Waffle':250
}

def estimate_cal(detail):
    if str(detail).startswith('Meals:'):
        return sum(CAL_MAP.get(f.strip(), 0) for f in detail.replace('Meals:','').split('|'))
    return 0

# ───────────────────────────────────────────────
# 0. Page config
# ───────────────────────────────────────────────
st.set_page_config(page_title="Bearable Dashboard", layout="wide")

# ───────────────────────────────────────────────
# 1. Utility: scrub & feature engineer
# ───────────────────────────────────────────────
def load_and_clean(csv_file: Path | str) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df.columns = [c.strip().replace('"', "") for c in df.columns]
    df["date"] = pd.to_datetime(df["date formatted"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating/amount"], errors="coerce")
    df = df.dropna(subset=["date"])

    cal = pd.DataFrame(index=pd.to_datetime(df["date"].dt.date.unique()))
    cal.index.name = "date"

    # helpers
    energy_map = {"v. low": 1, "low": 2, "ok": 3, "high": 4, "v. high": 5}
    qual_map = {"poor": 1, "ok": 2, "good": 3, "great": 4}
    sleep_pat = re.compile(
        r"Asleep\s+(\d{1,2}):(\d{2})\s*[–-]\s*(\d{1,2}):(\d{2})", re.I
    )

    def hhmm_to_hours(s):
        if pd.isna(s):
            return np.nan
        m = re.match(r"^(\d{1,2}):(\d{2})$", str(s).strip())
        return np.nan if not m else int(m[1]) + int(m[2]) / 60

    def asleep_hours(detail):
        if pd.isna(detail):
            return (np.nan, np.nan)
        m = sleep_pat.search(detail)
        if not m:
            return (np.nan, np.nan)
        h1, m1, h2, m2 = map(int, m.groups())
        start = timedelta(hours=h1, minutes=m1)
        end = timedelta(hours=h2, minutes=m2)
        if end < start:
            end += timedelta(days=1)
        return ((end - start).total_seconds() / 3600, h1)

    # mood
    mood = (
        df[df.category == "Mood"]
        .dropna(subset=["rating"])
        .groupby(df["date"].dt.date)["rating"]
        .mean()
        .rename("average_mood")
    )
    cal = cal.join(mood)

    # energy
    en = df[df.category == "Energy"].copy()
    if not en.empty:
        en["val"] = en.detail.str.lower().map(energy_map).fillna(en["rating"])
        cal = cal.join(
            en.groupby(en["date"].dt.date)["val"].mean().rename("average_energy")
        )

    # ── SLEEP ─────────────────────────────────────────────────────────
    sl = df[df.category == 'Sleep'].copy()
    if not sl.empty:
        sl['hours_num'] = sl['rating'].apply(hhmm_to_hours)
        sl[['hours_det', 'bed_hour']] = (                    # <- keep bed_hour
            sl['detail'].apply(asleep_hours).apply(pd.Series)
        )
        sl['sleep_duration_hours'] = sl['hours_num'].fillna(sl['hours_det'])
        sl['late_bedtime_flag']    = (sl['bed_hour'] > 1).astype(int)

        cal = (
            cal
            .join(                                             # sleep duration
                sl.groupby(sl['date'].dt.date)['sleep_duration_hours']
                  .first().rename('sleep_duration_hours')
            )
            .join(                                             # existing flag
                sl.groupby(sl['date'].dt.date)['late_bedtime_flag']
                  .max().rename('late_bedtime_flag')
            )
            .join(                                             # **NEW → keeps raw bedtime**
                sl.groupby(sl['date'].dt.date)['bed_hour']
                  .first().rename('bed_hour')
            )
        )


    # sleep quality
    sq = df[df.category == "Sleep quality"].copy()
    if not sq.empty:
        sq["qnum"] = sq["rating"].fillna(sq.detail.str.lower().map(qual_map))
        cal = cal.join(
            sq.groupby(sq["date"].dt.date)["qnum"].mean().rename("sleep_quality_score")
        )

    # symptoms
    sym = df[df.category == "Symptom"].copy()
    if not sym.empty:
        sym["name"] = sym.detail.str.replace(
            r"\s*\((?:Mild|Moderate|Severe|Unbearable|Extreme)\)", "", regex=True
        ).str.strip()
        sym["sev"] = sym["rating"].astype("Int64").clip(1, 4)
        piv = sym.groupby([sym["date"].dt.date, "name"])["sev"].max().unstack()
        cal = cal.join(piv)
        cal["TotalSymptomScore"] = piv.fillna(0).sum(axis=1)

    cal = cal.astype(float)
    cal = cal.loc[:, cal.notna().sum() >= 3]
    return cal.sort_index()


def add_features(df: pd.DataFrame, sleep_thr: float) -> pd.DataFrame:
    df2 = df.copy()
    df2["mood_7d_ma"] = df2["average_mood"].rolling(7, min_periods=1).mean()
    df2["mood_delta"] = df2["average_mood"] - df2["mood_7d_ma"]
    df2["is_weekend"] = (
        df2.index.to_series().dt.weekday.isin([5, 0]).astype(int)
    )  # Sat & Mon

    # rule: short sleep flag shifted +2 days
    df2["flag_sleep_predict"] = (
        (df2["sleep_duration_hours"] < sleep_thr).shift(2).fillna(False).astype(int)
    )
    return df2


# ───────────────────────────────────────────────
# 2. Sidebar controls
# ───────────────────────────────────────────────
st.sidebar.header("⚙️ Alert Thresholds")
energy_thr = st.sidebar.slider("Energy < threshold triggers flag", 1.0, 5.0, 3.0, 0.5)
sleep_thr = st.sidebar.slider("Sleep hours threshold (<) for 2‑day mood dip", 4.0, 8.0, 6.0, 0.5)
cluster3_std = st.sidebar.checkbox("Cluster 3 flag uses median + 1 SD", value=False)

# ───────────────────────────────────────────────
# 3. Auto‑load latest CSV from Google Drive
# ───────────────────────────────────────────────
st.title("🐰 Bearable Dashboard & Early‑Warning Flags")

EXPORT_DIR = Path("G:/Мой диск/Bearable_export")
pattern = re.compile(
    r"Bearable App - Data Export\. Generated (\d{2}-\d{2}-\d{4})\.csv"
)
latest_file = None
if EXPORT_DIR.exists():
    matches = []
    for file in EXPORT_DIR.glob("*.csv"):
        m = pattern.match(file.name)
        if m:
            dt = pd.to_datetime(m.group(1), format="%d-%m-%Y", errors="coerce")
            if pd.notna(dt):
                matches.append((dt, file))
    if matches:
        latest_file = sorted(matches, reverse=True)[0][1]

if latest_file:
    st.success(f"Auto‑loaded: {latest_file.name}")
    df_raw = load_and_clean(latest_file)
else:
    st.warning("No auto‑load file found – upload manually ⬇")
    uploaded = st.file_uploader("Upload Bearable CSV", type="csv")
    if not uploaded:
        st.stop()
    df_raw = load_and_clean(uploaded)

# ───────────────────────────────────────────────
# 4. Feature engineering & flags
# ───────────────────────────────────────────────
df_feat = add_features(df_raw, sleep_thr)
# 7‑day rolling standard‑deviation of sleep hours
df_feat['sleep_std_7d'] = (
    df_feat['sleep_duration_hours'].rolling(7, min_periods=1).std()
)

# ── Nutrition features & flags ────────────────────────────────────
# Use the same `latest_file` you loaded above for raw data
raw_nut = pd.read_csv(latest_file)
raw_nut['date'] = pd.to_datetime(raw_nut['date formatted'], errors='coerce').dt.date

nut_df = raw_nut[raw_nut['category']=='Nutrition'].copy()
# parse
nut_df['nutrition_amount']    = nut_df['detail'].map(parse_amount)
nut_df['nutrition_num_items'] = nut_df['detail'].map(count_meals)
# aggregate
agg = nut_df.groupby('date').agg({
    'nutrition_amount':'max',
    'nutrition_num_items':'sum'
})
df_feat = df_feat.join(agg, how='left').fillna({
    'nutrition_amount':0,
    'nutrition_num_items':0
})

# calories
nut_df['calories'] = nut_df['detail'].map(estimate_cal)
cal_agg = nut_df.groupby('date')['calories'].sum()
df_feat['daily_calories'] = cal_agg.reindex(df_feat.index, fill_value=0)

# evening items
pm_df = nut_df[nut_df['time of day']=='pm'].copy()
pm_df['pm_items'] = pm_df['detail'].map(count_meals)
evening_agg = pm_df.groupby('date')['pm_items'].sum()
df_feat['evening_num_items'] = evening_agg.reindex(df_feat.index, fill_value=0)

# now flags
median_cal = df_feat['daily_calories'].median()
df_feat['flag_low_cal']     = (df_feat['daily_calories'] < median_cal).astype(int)
df_feat['flag_few_items']   = (df_feat['nutrition_num_items'] < 3).astype(int)
df_feat['flag_no_pm_snack'] = (df_feat['evening_num_items'] == 0).astype(int)


# Sidebar slider for variability threshold
var_thr = st.sidebar.slider(
    "Sleep‑variability flag (7‑day SD > … hours)",
    0.0, 3.0, 1.0, 0.1
)

df_feat['flag_sleep_var'] = (df_feat['sleep_std_7d'] > var_thr).astype(int)

# cluster 3 creation
exclude_cols = {
    "average_mood", "average_energy", "sleep_duration_hours", "late_bedtime_flag",
    "sleep_quality_score", "mood_7d_ma", "mood_delta", "is_weekend",
    "flag_sleep_predict", "flag_low_energy", "flag_cluster3", "flag_sleep_var",
    "sleep_std_7d", "cluster3_sum", "bed_hour"  # ← make sure this is in the list
}
symptom_cols = [
    col for col in df_feat.columns
    if col not in exclude_cols and df_feat[col].dtype == float
]
Z = linkage(df_feat[symptom_cols].T.fillna(0), method="ward", metric="euclidean")
cluster_ids = fcluster(Z, 4, criterion="maxclust")
cluster3_cols = [symptom_cols[i] for i, cid in enumerate(cluster_ids) if cid == 3]

df_feat["cluster3_sum"] = df_feat[cluster3_cols].sum(axis=1)
thr3 = df_feat["cluster3_sum"].median() + (df_feat["cluster3_sum"].std() if cluster3_std else 0)
df_feat["flag_cluster3"] = (df_feat["cluster3_sum"] > thr3).astype(int)

# energy flag
df_feat["flag_low_energy"] = (df_feat["average_energy"] < energy_thr).astype(int)

# ───────────────────────────────────────────────
# 4b. Nutrition‐based flags
# ───────────────────────────────────────────────

# Compute your personal median calories
median_cal = df_feat["daily_calories"].median()

# Flag “low fuel” days
df_feat["flag_low_cal"] = (df_feat["daily_calories"] < median_cal).astype(int)

# Flag “too few items” days
df_feat["flag_few_items"] = (df_feat["nutrition_num_items"] < 3).astype(int)

# Flag “no evening snack” days
df_feat["flag_no_pm_snack"] = (df_feat["evening_num_items"] == 0).astype(int)


# ───────────────────────────────────────────────
# 5. Today’s alerts
# ───────────────────────────────────────────────
today = pd.to_datetime(date.today())
st.subheader(f"🚨 Alerts for {today.date()}")

labels_and_flags = [
    ("🔋 Low Energy",       "flag_low_energy"),
    ("💤 Short Sleep (t−2)", "flag_sleep_predict"),
    ("📈 Sleep Variability","flag_sleep_var"),
    ("⚠️ Cluster 3 spike",  "flag_cluster3"),
    ("🍽️ Low Calories",     "flag_low_cal"),
    ("🥄 Few Items",        "flag_few_items"),
    ("🌙 No PM Snack",      "flag_no_pm_snack"),
]

if today in df_feat.index:
    cols = st.columns(len(labels_and_flags))
    for col, (label, flag) in zip(cols, labels_and_flags):
        with col:
            st.metric(label, bool(df_feat.loc[today, flag]))
else:
    st.info("No data for today in this export.")



# ───────────────────────────────────────────────
# 6. Charts
# ───────────────────────────────────────────────
st.subheader("Key Relationships & Trends")
c1, c2, c3 = st.columns(3)

with c1:
    st.caption("Energy ➜ Mood (same day)")
    fig1, ax1 = plt.subplots()
    sns.regplot(
        x="average_energy",
        y="average_mood",
        data=df_feat,
        ax=ax1,
        scatter_kws={"s": 30},
        ci=None,
    )
    st.pyplot(fig1)

with c2:
    st.caption("Sleep < thr ➜ Mood (t+2)")
    df_p = df_feat.dropna(subset=["sleep_duration_hours", "average_mood"])
    df_p["sleep_flag"] = np.where(df_p["sleep_duration_hours"] < sleep_thr, "<thr", "≥thr")
    fig2, ax2 = plt.subplots()
    sns.boxplot(
        x="sleep_flag", y=df_p["average_mood"].shift(-2), data=df_p, ax=ax2
    )
    ax2.set_ylabel("Mood two days later")
    st.pyplot(fig2)

with c3:
    st.caption("Cluster 3 sum over time")
    st.line_chart(df_feat["cluster3_sum"])


# ───────────────────────────────────────────────
# 7. Timeline
# ───────────────────────────────────────────────
# 1. Define your flag colors
flag_colors = {
    "flag_low_energy": "#f94144",     # red
    "flag_sleep_predict": "#f9c74f",  # yellow
    "flag_sleep_var": "#277da1",      # blue
    "flag_cluster3": "#9c89b8",       # purple
}
# Add your nutrition flags to the color map
flag_colors.update({
    "flag_low_cal":    "#90be6d",  # green
    "flag_few_items":  "#43aa8b",  # teal
    "flag_no_pm_snack":"#577590",  # slate
})

# 2. Melt your flags into “long” form
# When you do the melt, include the new flags:
flags_long = (
    df_feat[
      ["flag_low_energy","flag_sleep_predict","flag_sleep_var",
       "flag_cluster3","flag_low_cal","flag_few_items","flag_no_pm_snack"]
    ]
    .reset_index()
    .melt(id_vars="date", var_name="flag", value_name="value")
)

# 3. Build an interactive Altair bar chart
chart = (
    alt.Chart(flags_long)
    .mark_bar()
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Flag On?"),
        color=alt.Color(
            "flag:N",
            scale=alt.Scale(
                domain=list(flag_colors.keys()),
                range=list(flag_colors.values())
            ),
            legend=alt.Legend(title="Flag Type")
        ),
        tooltip=["date:T", "flag:N", "value:Q"]
    )
    .interactive()  # pan & zoom
)

# 4. Display in Streamlit
st.subheader("Flag Timeline")
st.altair_chart(chart, use_container_width=True)

# ───────────────────────────────────────────────
# 8. Explainers
# ───────────────────────────────────────────────

with st.expander("❓ What do the alerts mean?", expanded=False):
    cols = st.columns(4)
    expls = [
        ("🔴 🔋 Low Energy", "You're running low today.\nEnergy below your threshold → fatigue, distraction, low mood."),
        ("🟡 💤 Short Sleep (t−2)", "Slept <6 h two days ago → watch for irritability or low motivation."),
        ("🔵 📈 Sleep Variability", "Big swings in sleep over the past week → focus crashes, mood instability."),
        ("🟣 ⚠️ Cluster 3", "ADHD/depression symptoms spiking → expect overwhelm and dysregulation.")
    ]
    for col, (title, text) in zip(cols, expls):
        with col:
            st.markdown(f"**{title}**")
            st.write(text)

with st.expander("🍽 What do the nutrition alerts mean?", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**⛽ Low Fuel**  \nCalories below your personal median → expect flatter mood/energy.")
    with c2:
        st.markdown("**🍽 Few Items**  \nFewer than 3 log‑entries → consider adding a snack or small meal.")
    with c3:
        st.markdown("**🌙 No PM Snack**  \nNo evening intake → a small bedtime snack can help round out your day’s energy.")


with st.expander("🧠 What is Cluster 3? (ADHD / Depression Symptom Group)", expanded=False):
    st.markdown("""
Cluster 3 includes symptoms that most strongly correlate with **same-day mood drops**.
They reflect patterns often tied to **executive dysfunction**, **mental restlessness**, and **emotional reactivity**.
""")

    # Dynamically split the list into two columns
    col1, col2 = st.columns(2)
    half = len(cluster3_cols) // 2 + len(cluster3_cols) % 2
    col1_list = cluster3_cols[:half]
    col2_list = cluster3_cols[half:]

    with col1:
        for symptom in col1_list:
            st.markdown(f"- {symptom}")

    with col2:
        for symptom in col2_list:
            st.markdown(f"- {symptom}")


# ───────────────────────────────────────────────
# 9. Download processed data
# ───────────────────────────────────────────────
st.markdown("### Download processed CSV")
csv_bytes = df_feat.to_csv(index=True).encode()
st.download_button(
    "📥 Download",
    data=csv_bytes,
    file_name=f"bearable_processed_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv",
)
