# 📊 Bearable Mood Tracker – Streamlit Dashboard

This Streamlit app visualizes mood, energy, sleep, and symptom data exported from the **Bearable app**, with a focus on detecting early warning signs of low mood and executive dysfunction.

---

## ✅ Features

- 📅 **Daily Alerts** for:
  - Low energy
  - Sleep < 6h (lagged 2 days)
  - High sleep variability (7-day std)
  - Cluster 3 symptom spikes

- 🧠 **Flag Timeline** chart with color-coded, interactive indicators
- 🔎 **Cluster 3 Explainer**: Executive function, restlessness, mood lability
- ⚙️ **Sidebar Controls** to adjust thresholds (energy, sleep, variability)
- 📦 **Automatic loading** of latest Bearable export from a synced Google Drive folder

---

## 🚀 How to Use

1. **Set up Google Drive sync**:
   - Your Bearable app exports should save to:
     ```
     G:/Мой диск/Bearable_export/
     ```
     (or whatever your synced Google Drive path is)

2. **Run the app** from your terminal:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **The app will automatically**:
   - Load the most recent CSV export
   - Clean and parse it using your `load_and_clean()` function
   - Generate daily insights and display timelines

---

## 🔧 Configurable Flags

You can customize detection thresholds via the **sidebar sliders**:

- **Energy threshold** (default = 3.0)
- **Sleep duration threshold** (default = 6.0h, t−2)
- **Sleep variability threshold** (default = 1.0h std over 7d)

---

## 🔁 How to Adapt the App Later

### 1. Add New Flags
- Define a new flag in the `add_features()` step
- Add it to:
  - The **alerts block** (`st.metric(...)`)
  - The **timeline chart**
  - The **explainers section**

### 2. Update Symptom Clustering
- Re-run clustering in Jupyter with more data
- Update `cluster3_cols` in the app with the new symptom list

### 3. Use External Uploads Instead of Drive (optional)
- Replace the Google Drive path with:
  ```python
  uploaded_file = st.file_uploader("Upload your Bearable export")
  ```
  for manual CSV uploads

---

## 💡 Tips

- Keep your `scrub.py` / `load_and_clean()` up to date with any changes in Bearable's export format
- Revisit your EDA every ~30 days to refine thresholds, validate hypotheses, or discover new patterns

---

## 🧠 Background

This app is part of a self-tracking system for a user with **ADHD**, designed to surface meaningful patterns and give early warnings for:
- Mood dips
- Executive dysfunction episodes
- Sleep-related instability

All flags are derived from exploratory data analysis and refined over time based on empirical signal and lived experience.