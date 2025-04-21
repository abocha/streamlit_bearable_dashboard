# Full rewritten streamlit_app.py

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

# ───────────────────────────────────────────────
# 0. Page config
# ───────────────────────────────────────────────
st.set_page_config(page_title="Bearable Dashboard", layout="wide")

# ───────────────────────────────────────────────
# 1. Helpers: parsing and ingestion
# ───────────────────────────────────────────────
def parse_amount(detail):
    """Map 'Amount eaten – X' to 1/2/3."""
    m = re.search(r'Amount eaten\s*[-–]\s*(\w+)', str(detail))
    return {'Little':1, 'Moderate':2, 'A lot':3}.get(m.group(1), np.nan) if m else np.nan

def count_meals(detail):
    """Count items in 'Meals: item1 | item2 | ...'."""
    s = str(detail)
    if s.startswith('Meals:'):
        return sum(1 for part in s.replace('Meals:','').split('|') if part.strip())
    return 0

# Calorie estimates (customize per user)
CAL_MAP = {
    'Coffee':5, 'Instant noodles':400, 'Shawarma':600, 'Banana':100, 'Yogurt':150,
    'Banana and yogurt':250, 'Chicken':200, 'Chicken and rice':500,
    'Chicken Salad':300, 'Bánh Mì':550, 'Burrito':500, 'Chips':150,
    'Chocolate':200, 'Chocopie':100, 'Coke':140, 'Egg and sausage':250,
    'KFC':800, 'Nuts':200, 'Oreo':50, 'Pizza':300, 'Smoothie':200,
    'Snickers':250, 'Waffle':250
}

def estimate_cal(detail):
    """Sum calories for each logged meal."""
    s = str(detail)
    if s.startswith('Meals:'):
        return sum(CAL_MAP.get(item.strip(), 0) 
                   for item in s.replace('Meals:','').split('|'))
    return 0

def load_and_clean(csv_file) -> pd.DataFrame:
    """Clean raw Bearable CSV into daily DataFrame of mood, energy, sleep, symptoms."""
    df = pd.read_csv(csv_file)
    df.columns = [c.strip().replace('"','') for c in df.columns]
    df['date'] = pd.to_datetime(df['date formatted'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating/amount'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Calendar index
    cal = pd.DataFrame(index=pd.to_datetime(df['date'].dt.date.unique()))
    cal.index.name = 'date'

    # Maps
    energy_map = {'v. low':1, 'low':2, 'ok':3, 'high':4, 'v. high':5}
    qual_map   = {'poor':1, 'ok':2, 'good':3, 'great':4}
    sleep_pat  = re.compile(r'Asleep\s+(\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})')

    def hhmm_to_hours(s):
        if pd.isna(s): return np.nan
        m = re.match(r'(\d{1,2}):(\d{2})', str(s).strip())
        return (int(m[1]) + int(m[2])/60) if m else np.nan

    def asleep_hours(detail):
        if pd.isna(detail): return (np.nan, np.nan)
        m = sleep_pat.search(detail)
        if not m: return (np.nan, np.nan)
        h1,m1,h2,m2 = map(int, m.groups())
        start=pd.Timedelta(hours=h1, minutes=m1)
        end  =pd.Timedelta(hours=h2, minutes=m2)
        if end<start: end+=pd.Timedelta(days=1)
        return ((end-start).total_seconds()/3600, h1)

    # Mood
    mood = (df[df.category=='Mood']
            .dropna(subset=['rating'])
            .groupby(df['date'].dt.date)['rating']
            .mean().rename('average_mood'))
    cal = cal.join(mood)

    # Energy
    en = df[df.category=='Energy'].copy()
    if not en.empty:
        en['val']=en.detail.str.lower().map(energy_map).fillna(en['rating'])
        cal = cal.join(en.groupby(en['date'].dt.date)['val']
                       .mean().rename('average_energy'))

    # Sleep
    sl = df[df.category=='Sleep'].copy()
    if not sl.empty:
        sl['hours_num']=sl['rating'].apply(hhmm_to_hours)
        sl[['hours_det','bed_hour']] = (
            sl['detail'].apply(asleep_hours).apply(pd.Series)
        )
        sl['sleep_duration_hours']=sl['hours_num'].fillna(sl['hours_det'])
        sl['late_bedtime_flag']=(sl['bed_hour']>1).astype(int)
        cal=(cal
             .join(sl.groupby(sl['date'].dt.date)['sleep_duration_hours']
                   .first().rename('sleep_duration_hours'))
             .join(sl.groupby(sl['date'].dt.date)['late_bedtime_flag']
                   .max().rename('late_bedtime_flag')))
    
    # Sleep quality
    sq = df[df.category=='Sleep quality'].copy()
    if not sq.empty:
        sq['qnum']=sq['rating'].fillna(sq.detail.str.lower().map(qual_map))
        cal = cal.join(sq.groupby(sq['date'].dt.date)['qnum']
                       .mean().rename('sleep_quality_score'))

    # Symptoms
    sym = df[df.category=='Symptom'].copy()
    if not sym.empty:
        sym['name']=sym.detail.str.replace(
            r'\s*\((Mild|Moderate|Severe|Unbearable|Extreme)\)','',regex=True
        ).str.strip()
        sym['sev']=sym['rating'].astype('Int64').clip(1,4)
        piv = sym.groupby([sym['date'].dt.date,'name'])['sev'].max().unstack()
        cal=cal.join(piv)
        cal['TotalSymptomScore']=piv.fillna(0).sum(axis=1)

    return cal.astype(float).loc[:, cal.notna().sum()>=3].sort_index()

def add_features(df: pd.DataFrame,
                 energy_thr: float,
                 sleep_thr: float,
                 var_thr: float) -> pd.DataFrame:
    """Add rolling averages, deltas, and early-warning flags."""
    df2 = df.copy()
    df2['mood_7d_ma'] = df2['average_mood'].rolling(7,min_periods=1).mean()
    df2['mood_delta']=df2['average_mood']-df2['mood_7d_ma']
    df2['is_weekend']=df2.index.to_series().dt.weekday.isin([5,0]).astype(int)
    # Energy flag
    df2['flag_low_energy']=(df2['average_energy']<energy_thr).astype(int)
    # Sleep predict (t−2)
    df2['flag_sleep_predict']=(df2['sleep_duration_hours']<sleep_thr).shift(2).fillna(0).astype(int)
    # Sleep variability
    df2['sleep_std_7d']=df2['sleep_duration_hours'].rolling(7,min_periods=1).std()
    df2['flag_sleep_var']=(df2['sleep_std_7d']>var_thr).astype(int)
    # Cluster 3
    exclude={'average_mood','average_energy','sleep_duration_hours',
             'late_bedtime_flag','sleep_quality_score','mood_7d_ma',
             'mood_delta','is_weekend','flag_low_energy','flag_sleep_predict',
             'flag_sleep_var','sleep_std_7d'}
    symptom_cols=[c for c in df2.columns if c not in exclude and df2[c].dtype==float]
    Z=linkage(df2[symptom_cols].T.fillna(0),method='ward',metric='euclidean')
    ids=fcluster(Z,4,criterion='maxclust')
    cols3=[symptom_cols[i] for i,cid in enumerate(ids) if cid==3]
    df2['cluster3_sum']=df2[cols3].sum(axis=1)
    thr3=df2['cluster3_sum'].median()
    df2['flag_cluster3']=(df2['cluster3_sum']>thr3).astype(int)
    return df2

# ───────────────────────────────────────────────
# 2. Sidebar controls
# ───────────────────────────────────────────────
st.sidebar.header("⚙️ Alert Thresholds")
energy_thr = st.sidebar.slider("Energy < threshold",1.0,5.0,3.0,0.5)
sleep_thr  = st.sidebar.slider("Sleep < threshold (t−2)",4.0,8.0,6.0,0.5)
var_thr    = st.sidebar.slider("Sleep variability (7d SD)",0.0,3.0,1.0,0.1)

# ───────────────────────────────────────────────
# 3. Load CSV (auto or manual)
# ───────────────────────────────────────────────
EXPORT_DIR=Path("G:/Мой диск/Bearable_export")
pattern=re.compile(r"Bearable App - Data Export\. Generated (\d{2}-\d{2}-\d{4})\.csv")
latest_file=None
if EXPORT_DIR.exists():
    files=[(pd.to_datetime(m.group(1),format="%d-%m-%Y"),f)
           for f in EXPORT_DIR.glob("*.csv") if (m:=pattern.match(f.name))]
    if files: latest_file=sorted(files,reverse=True)[0][1]
if latest_file:
    st.sidebar.success(f"Auto‑loaded: {latest_file.name}")
    source=latest_file
else:
    source=st.sidebar.file_uploader("Upload Bearable CSV",type="csv")
    if not source: st.stop()

# ───────────────────────────────────────────────
# 4. Clean & feature engineering
# ───────────────────────────────────────────────
df_raw = load_and_clean(source)
raw    = pd.read_csv(source)
df_feat=add_features(df_raw,energy_thr,sleep_thr,var_thr)

# ── Nutrition merge & flags ─────────────────────────────────────────
if 'Nutrition' in raw['category'].unique():
    nut=raw[raw['category']=='Nutrition'].copy()
    nut['date']=pd.to_datetime(nut['date formatted'],errors='coerce').dt.date
    nut['nutrition_amount']   =nut['detail'].map(parse_amount)
    nut['nutrition_num_items']=nut['detail'].map(count_meals)
    agg=nut.groupby('date').agg({'nutrition_amount':'max','nutrition_num_items':'sum'})
    df_feat=df_feat.join(agg,how='left').fillna({'nutrition_amount':0,'nutrition_num_items':0})
    nut['calories']=nut['detail'].map(estimate_cal)
    df_feat['daily_calories']=nut.groupby(nut['date'])['calories'].sum().reindex(df_feat.index,fill_value=0)
    pm=nut[nut['time of day']=='pm'].copy()
    pm['pm_items']=pm['detail'].map(count_meals)
    df_feat['evening_num_items']=pm.groupby(pm['date'])['pm_items'].sum().reindex(df_feat.index,0)
else:
    df_feat[['nutrition_amount','nutrition_num_items','daily_calories','evening_num_items']]=0

# Flags for nutrition
median_cal=df_feat['daily_calories'].median()
df_feat['flag_low_cal']     =(df_feat['daily_calories']<median_cal).astype(int)
df_feat['flag_few_items']   =(df_feat['nutrition_num_items']<3).astype(int)
df_feat['flag_no_pm_snack'] =(df_feat['evening_num_items']==0).astype(int)

# ───────────────────────────────────────────────
# 5. Today's alerts
# ───────────────────────────────────────────────
today=pd.to_datetime(date.today())
st.subheader(f"🚨 Alerts for {today.date()}")

alerts=[
    ("🔋 Low Energy",      'flag_low_energy'),
    ("💤 Short Sleep",     'flag_sleep_predict'),
    ("📈 Sleep Var",       'flag_sleep_var'),
    ("⚠️ Cluster 3",        'flag_cluster3'),
    ("⛽ Low Fuel",         'flag_low_cal'),
    ("🥄 Few Items",       'flag_few_items'),
    ("🌙 No PM Snack",     'flag_no_pm_snack'),
]
if today in df_feat.index:
    cols=st.columns(len(alerts))
    for col,(label,flag) in zip(cols,alerts):
        with col:
            st.metric(label, bool(df_feat.loc[today,flag]))
else:
    st.info("No data for today.")

# ───────────────────────────────────────────────
# 6. Charts
# ───────────────────────────────────────────────
st.subheader("Key Trends")
c1,c2,c3=st.columns(3)

with c1:
    st.caption("Energy ➔ Mood")
    fig,ax=plt.subplots()
    sns.regplot(x='average_energy',y='average_mood',data=df_feat,ax=ax,ci=None)
    st.pyplot(fig)

with c2:
    st.caption("Sleep < thr ➔ Mood (t+2)")
    dfp=df_feat.dropna(subset=['sleep_duration_hours','average_mood'])
    dfp['slp_flag']=np.where(dfp['sleep_duration_hours']<sleep_thr,'<thr','>=thr')
    fig,ax=plt.subplots()
    sns.boxplot(x='slp_flag',y=dfp['average_mood'].shift(-2),data=dfp,ax=ax)
    ax.set_ylabel("Mood t+2")
    st.pyplot(fig)

with c3:
    st.caption("Cluster 3 over time")
    st.line_chart(df_feat['cluster3_sum'])

# ───────────────────────────────────────────────
# 7. Timeline
# ───────────────────────────────────────────────
flag_colors={
    'flag_low_energy':'#f94144','flag_sleep_predict':'#f9c74f',
    'flag_sleep_var':'#277da1','flag_cluster3':'#9c89b8',
    'flag_low_cal':'#90be6d','flag_few_items':'#43aa8b',
    'flag_no_pm_snack':'#577590'
}
flags_long=(df_feat[list(flag_colors)]
            .reset_index()
            .melt('date',var_name='flag',value_name='value'))
chart=(
    alt.Chart(flags_long)
    .mark_bar()
    .encode(x='date:T',y='value:Q',color=alt.Color('flag:N',
        scale=alt.Scale(domain=list(flag_colors),range=list(flag_colors.values()))),
        tooltip=['date:T','flag:N','value:Q'])
    .interactive()
)
st.subheader("Flag Timeline")
st.altair_chart(chart,use_container_width=True)

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
