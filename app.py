import joblib
import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_V2     = 'pollution_model_v2.pkl'
COLS_V2      = 'model_columns_v2.pkl'
NO2_MODEL    = 'pollution_model_NO2_log.pkl'
NON_NO2_FILE = 'model_non_NO2_targets.pkl'
LAG_LOOKUP   = 'station_lag_lookup.pkl'
MODEL_V1     = 'pollution_model.pkl'

# ── Google Drive file ID — only the large model (95 MB) is hosted on Drive ────
# The other 4 small pkl files are committed directly to the GitHub repo.
# After uploading pollution_model_v2.pkl to Google Drive (Share → Anyone with link),
# paste the ID from the share URL: drive.google.com/file/d/<ID>/view
GDRIVE_MODEL_V2_ID = '1rdpbpCZlRaju2Id3xLwl3gknvjc5fwBs'

# Download the large model from Drive if not already present
if not os.path.exists(MODEL_V2) and not GDRIVE_MODEL_V2_ID.startswith('PASTE_'):
    gdown.download(f'https://drive.google.com/uc?id={GDRIVE_MODEL_V2_ID}',
                   MODEL_V2, quiet=False)

if os.path.exists(MODEL_V2) and os.path.exists(COLS_V2) and os.path.exists(LAG_LOOKUP):
    model           = joblib.load(MODEL_V2)
    model_cols      = joblib.load(COLS_V2)
    no2_model       = joblib.load(NO2_MODEL)      if os.path.exists(NO2_MODEL)    else None
    non_NO2_targets = joblib.load(NON_NO2_FILE)   if os.path.exists(NON_NO2_FILE) else None
    lag_lookup      = joblib.load(LAG_LOOKUP)
    uses_lag        = True
    uses_season     = 'season' in model_cols
else:
    # Fall back to original v1 model
    if not os.path.exists(MODEL_V1):
        gdown.download('https://drive.google.com/uc?id=12GH455E2HXYiFVcph6l_upto35eMa1yv',
                       MODEL_V1, quiet=False)
    model           = joblib.load(MODEL_V1)
    model_cols      = joblib.load('model_columns.pkl')
    no2_model       = None
    non_NO2_targets = None
    lag_lookup      = {}
    uses_lag        = False
    uses_season     = False

# ── WHO/EU Drinking Water Safety Thresholds ───────────────────────────────────
WHO_THRESHOLDS = {
    'O2':  {'safe_min': 6.0,  'caution_min': 4.0,  'unit': 'mg/L', 'direction': 'above'},
    'NO3': {'safe_max': 10.0, 'caution_max': 50.0,  'unit': 'mg/L', 'direction': 'below'},
    'NO2': {'safe_max': 0.1,  'caution_max': 1.0,   'unit': 'mg/L', 'direction': 'below'},
    'SO4': {'safe_max': 250,  'caution_max': 500,   'unit': 'mg/L', 'direction': 'below'},
    'PO4': {'safe_max': 0.1,  'caution_max': 0.5,   'unit': 'mg/L', 'direction': 'below'},
    'CL':  {'safe_max': 250,  'caution_max': 500,   'unit': 'mg/L', 'direction': 'below'},
}

def classify_safety(pollutant, value):
    t = WHO_THRESHOLDS.get(pollutant)
    if t is None:
        return '', '#888888'
    if t['direction'] == 'above':
        if value >= t['safe_min']:    return '✅ Safe',    '#2e7d32'
        if value >= t['caution_min']: return '⚠️ Caution', '#e65100'
        return '❌ Unsafe', '#b71c1c'
    else:
        if value <= t['safe_max']:    return '✅ Safe',    '#2e7d32'
        if value <= t['caution_max']: return '⚠️ Caution', '#e65100'
        return '❌ Unsafe', '#b71c1c'

# ── Pollutant metadata ─────────────────────────────────────────────────────────
POLLUTANTS = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']
POLLUTANT_NAMES = {
    'O2': 'Dissolved Oxygen', 'NO3': 'Nitrate', 'NO2': 'Nitrite',
    'SO4': 'Sulfate', 'PO4': 'Phosphate', 'CL': 'Chloride'
}
SEASON_NAMES = {1: '❄️ Winter (Dec–Feb)', 2: '🌱 Spring (Mar–May)',
                3: '☀️ Summer (Jun–Aug)', 4: '🍂 Fall (Sep–Nov)'}

# ── Page styling ───────────────────────────────────────────────────────────────
st.set_page_config(page_title='Water Quality Predictor', page_icon='💧', layout='centered')
st.markdown("""
<style>
  .title    { text-align:center; font-size:34px; color:#0066cc; font-weight:700; }
  .subtitle { text-align:center; font-size:16px; color:#555; margin-bottom:4px; }
  .badge    { text-align:center; font-size:13px; color:#888; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>💧 Water Quality Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered multi-pollutant forecasting with WHO/EU safety classification</div>",
            unsafe_allow_html=True)
model_tag = 'Lag-feature RF · v2 (avg CV R²=0.48)' if uses_lag else 'Random Forest · v1'
st.markdown(f"<div class='badge'>Model: {model_tag}</div>", unsafe_allow_html=True)
st.markdown("---")

# ── Input UI ───────────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns(3)
with col_a:
    year_input = st.number_input("📅 Year", min_value=2000, max_value=2100, value=2025)
with col_b:
    station_id = st.selectbox("🏭 Station ID", options=[str(i) for i in range(1, 23)])
with col_c:
    season_input = st.selectbox(
        "🌤️ Season", options=list(SEASON_NAMES.keys()),
        format_func=lambda s: SEASON_NAMES[s],
        disabled=not uses_season,
        help="Season (Winter/Spring/Summer/Fall) captures temperature-driven water quality cycles."
    )

# Show which lag values will be used (transparency for recruiter demo)
if uses_lag and station_id in lag_lookup:
    with st.expander("📋 Last known readings used as lag input (auto-populated from training data)"):
        lag_df = pd.DataFrame([lag_lookup[station_id]], index=['Last reading'])
        st.dataframe(lag_df.style.format("{:.3f}"))
elif uses_lag:
    st.caption(f"⚠️ No historical data for Station {station_id} — lag features will default to 0.")

# ── Prediction ─────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Water Quality", use_container_width=True):
    # Build input row
    row = {'year': [year_input], 'id': [station_id]}
    if uses_season:
        row['season'] = [season_input]

    # Inject lag features from last known station readings
    if uses_lag and station_id in lag_lookup:
        for p in POLLUTANTS:
            row[f'{p}_lag1'] = [lag_lookup[station_id].get(p, 0.0)]
    elif uses_lag:
        for p in POLLUTANTS:
            row[f'{p}_lag1'] = [0.0]

    input_df  = pd.DataFrame(row)
    input_enc = pd.get_dummies(input_df, columns=['id'])
    for col in model_cols:
        if col not in input_enc.columns:
            input_enc[col] = 0
    input_enc = input_enc[model_cols]

    # Run model(s)
    if non_NO2_targets and no2_model:
        main_preds = model.predict(input_enc)[0]
        no2_pred   = float(np.expm1(no2_model.predict(input_enc)[0]))
        pred_dict  = dict(zip(non_NO2_targets, main_preds))
        pred_dict['NO2'] = no2_pred
        predicted = [pred_dict[p] for p in POLLUTANTS]
    else:
        predicted = list(model.predict(input_enc)[0])

    season_label = f" · {SEASON_NAMES[season_input]}" if uses_season else ""
    st.markdown(f"### 📊 Station **{station_id}**{season_label} · **{year_input}**")
    st.markdown("**WHO/EU Key:** ✅ Safe &nbsp; ⚠️ Caution &nbsp; ❌ Unsafe")
    st.markdown("")

    col1, col2 = st.columns(2)
    safe_count = 0
    for i, (p, val) in enumerate(zip(POLLUTANTS, predicted)):
        status, _ = classify_safety(p, val)
        if status.startswith('✅'):
            safe_count += 1
        unit      = WHO_THRESHOLDS[p]['unit']
        full_name = POLLUTANT_NAMES[p]
        (col1 if i % 2 == 0 else col2).metric(
            label=f"{p} — {full_name}",
            value=f"{val:.3f} {unit}",
            delta=status
        )

    st.markdown("---")
    colour = '🟢' if safe_count == len(POLLUTANTS) else ('🟡' if safe_count >= 4 else '🔴')
    st.info(f"{colour} **{safe_count}/{len(POLLUTANTS)} parameters** within WHO safe limits.")
    if uses_lag:
        st.caption("Predictions are conditioned on last known sensor readings (lag features). "
                   "Refresh `station_lag_lookup.pkl` with live sensor data for real-time deployment.")
