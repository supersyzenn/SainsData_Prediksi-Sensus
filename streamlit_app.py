# streamlit_app_xgb_simple_merge.py
"""
Streamlit app: Dashboard Sensus + XGBoost (Umur, Jenis Kelamin, Pendidikan)
Perubahan utama: Kelas target yang frekuensinya < min_count akan DIGABUNG menjadi 'Other'
(sehingga tidak lagi menyebabkan error stratify saat train_test_split).
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import zipfile
import xml.etree.ElementTree as ET
import re
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
from collections import Counter

# try import xgboost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception:
    XGBClassifier = None
    xgb_available = False

st.set_page_config(page_title="Sensus — XGBoost (merge rare classes)", layout="wide")
CURRENT_YEAR = datetime.now().year

# ---------- Simple XLSX parser fallback (same as before) ----------
_COL_LETTER_RE = re.compile(r"([A-Z]+)(\d+)")
def _col_letter_to_idx(col):
    idx = 0
    for c in col:
        idx = idx * 26 + (ord(c) - ord('A') + 1)
    return idx - 1

def _parse_shared_strings(zf):
    try:
        with zf.open("xl/sharedStrings.xml") as f:
            tree = ET.parse(f); root = tree.getroot()
            ss = []
            for si in root.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                texts = [t.text or "" for t in si.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
                ss.append("".join(texts))
            return ss
    except KeyError:
        return []

def _get_first_sheet_path(zf):
    with zf.open("xl/workbook.xml") as f:
        tree = ET.parse(f); root = tree.getroot()
        ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sheet = root.find(".//ns:sheets/ns:sheet", ns)
        if sheet is None:
            raise ValueError("No sheets")
        rid = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    with zf.open("xl/_rels/workbook.xml.rels") as f:
        tree = ET.parse(f); root = tree.getroot()
        for rel in root.findall(".//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"):
            if rel.attrib.get("Id") == rid:
                tgt = rel.attrib.get("Target"); return "xl/" + tgt.replace("\\","/")
    raise ValueError("sheet path not found")

def _parse_sheet(zf, path, shared_strings):
    with zf.open(path) as f:
        tree = ET.parse(f); root = tree.getroot()
        ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows = {}; max_col = -1
        for row in root.findall(".//ns:row", ns):
            r_idx = int(row.attrib.get("r", "0")) - 1
            rowd = {}
            for c in row.findall("ns:c", ns):
                ref = c.attrib.get("r","")
                m = _COL_LETTER_RE.match(ref)
                if not m: continue
                col_letters = m.group(1); col_idx = _col_letter_to_idx(col_letters)
                if col_idx > max_col: max_col = col_idx
                ctype = c.attrib.get("t")
                inline = c.find("ns:is", ns)
                val = ""
                if inline is not None:
                    val = "".join([t.text or "" for t in inline.findall(".//ns:t", ns)])
                else:
                    v = c.find("ns:v", ns)
                    if v is not None and v.text is not None:
                        if ctype == "s":
                            try: val = shared_strings[int(v.text)]
                            except: val = v.text
                        else:
                            val = v.text
                rowd[col_idx] = val
            rows[r_idx] = rowd
        max_row = max(rows.keys()) if rows else -1
        data = []
        for r in range(max_row + 1):
            rd = rows.get(r, {})
            data.append([rd.get(c,"") for c in range(max_col+1)])
        df = pd.DataFrame(data)
        if not df.empty:
            header = df.iloc[0].astype(str).str.strip().tolist()
            non_empty = sum(1 for x in header if x != "" and not x.isdigit())
            if non_empty >= max(1, len(header)//2):
                df.columns = header; df = df.drop(index=0).reset_index(drop=True)
        df.columns = [str(c).strip() if (pd.notna(c) and str(c).strip()!='') else f"col_{i}" for i,c in enumerate(df.columns)]
        return df

def read_xlsx_fallback(bytes_io):
    zf = zipfile.ZipFile(bytes_io)
    ss = _parse_shared_strings(zf)
    sp = _get_first_sheet_path(zf)
    return _parse_sheet(zf, sp, ss)

# ---------- Helpers ----------
def normalize_text(x):
    if pd.isna(x): return x
    s = str(x).strip()
    return " ".join(s.split()) if s!="" else np.nan

def to_int_safe(x):
    try:
        if pd.isna(x): return np.nan
        return int(float(x))
    except:
        return np.nan

def dedupe_columns(df):
    cols = list(df.columns); seen = {}; new = []
    for c in cols:
        if c not in seen:
            seen[c] = 1; new.append(c)
        else:
            seen[c] += 1; new.append(f"{c}_{seen[c]-1}")
    df = df.copy(); df.columns = new; return df

# ---------- Safe merge function for rare classes ----------
def merge_rare_classes(series_labels, min_count=3, other_label='Other'):
    """
    series_labels: pd.Series of original labels (strings)
    min_count: classes with count < min_count will be merged into other_label
    returns new_series, counts_before, counts_after
    """
    counts = series_labels.value_counts()
    rare = counts[counts < min_count].index.tolist()
    if len(rare) == 0:
        return series_labels.copy(), counts, counts
    new = series_labels.copy().apply(lambda v: other_label if v in rare else v)
    return new, counts, new.value_counts()

# ---------- App UI & Load ----------
st.title("Dashboard Sensus — Prediksi Jenis Pekerjaan (merge rare classes)")
st.write("Upload CSV atau XLSX. Model pakai fitur: Umur, Jenis Kelamin, Pendidikan. Kelas jarang digabung jadi 'Other' sebelum split.")

uploaded = st.file_uploader("Upload CSV / Excel", type=['csv','xlsx','xls'])

@st.cache_data
def load_data(file) -> pd.DataFrame:
    if file is None: return pd.DataFrame()
    name = getattr(file, "name", "")
    b = file.read()
    if name.lower().endswith('.csv'):
        try: return pd.read_csv(BytesIO(b))
        except: return pd.read_csv(BytesIO(b), sep=';')
    if name.lower().endswith(('.xlsx','.xls')):
        try:
            return pd.read_excel(BytesIO(b))
        except Exception:
            if name.lower().endswith('.xlsx'):
                try: return read_xlsx_fallback(BytesIO(b))
                except Exception:
                    return pd.DataFrame()
            else:
                return pd.DataFrame()
    return pd.DataFrame()

df_raw = load_data(uploaded)
if df_raw is None or df_raw.empty:
    st.info("Belum ada data valid. Upload file yang berisi kolom yang diperlukan.")
    st.stop()

# Preprocess & rename
df = df_raw.copy()
df.columns = [str(c).strip() for c in df.columns]

# rename requested labels
if "Pendidikan Terakhir" in df.columns and "Pendidikan" not in df.columns:
    df = df.rename(columns={"Pendidikan Terakhir": "Pendidikan"})
if "Pekerjaan" in df.columns and "Jenis Pekerjaan" not in df.columns:
    df = df.rename(columns={"Pekerjaan": "Jenis Pekerjaan"})

# ensure required columns exist
required = ["Nama Lengkap","Nama Kepala Keluarga","Tahun Lahir","Jenis Kelamin","Agama","Pendidikan","Jenis Pekerjaan","RT","RW","Longitude","Latitude"]
for c in required:
    if c not in df.columns:
        df[c] = np.nan

# clean text cols
for c in ["Nama Lengkap","Nama Kepala Keluarga","Agama","Pendidikan","Jenis Pekerjaan","Jenis Kelamin"]:
    if c in df.columns:
        df[c] = df[c].apply(normalize_text).apply(lambda x: str(x).title() if pd.notna(x) else x)

# Tahun Lahir -> Umur
df['Tahun Lahir Raw'] = df['Tahun Lahir']
df['Tahun Lahir'] = df['Tahun Lahir'].apply(to_int_safe)
df['Umur'] = df['Tahun Lahir'].apply(lambda y: (CURRENT_YEAR - int(y)) if (not pd.isna(y)) else np.nan)

# coordinate numeric
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['_has_coords'] = df[['Longitude','Latitude']].notna().all(axis=1)

# ---------- Filter & KPIs ----------
st.sidebar.header("Filter")
rt_choice = st.sidebar.multiselect("RT", options=sorted(df['RT'].dropna().unique()), default=[])
rw_choice = st.sidebar.multiselect("RW", options=sorted(df['RW'].dropna().unique()), default=[])
agama_choice = st.sidebar.multiselect("Agama", options=sorted(df['Agama'].dropna().unique()), default=[])
edu_choice = st.sidebar.multiselect("Pendidikan", options=sorted(df['Pendidikan'].dropna().unique()), default=[])
jk_choice = st.sidebar.multiselect("Jenis Kelamin", options=sorted(df['Jenis Kelamin'].dropna().unique()), default=[])
jp_choice = st.sidebar.multiselect("Jenis Pekerjaan", options=sorted(df['Jenis Pekerjaan'].dropna().unique()), default=[])
umur_min, umur_max = st.sidebar.slider("Rentang Umur", 0, 120, (0, 120))

mask = pd.Series(True, index=df.index)
if rt_choice:
    mask &= df['RT'].isin(rt_choice)
if rw_choice:
    mask &= df['RW'].isin(rw_choice)
if agama_choice:
    mask &= df['Agama'].isin(agama_choice)
if edu_choice:
    mask &= df['Pendidikan Terakhir'].isin(edu_choice)
if jk_choice:
    mask &= df['Jenis Kelamin'].isin(jk_choice)
if jp_choice:
    mask &= df['Jenis Pekerjaan'].isin(jp_choice)
mask &= df['Umur'].between(umur_min, umur_max, inclusive='both')

df_filtered = df[mask].copy()

# quick KPIs
st.markdown("### Ringkasan Cepat")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Responden (baris)", f"{len(df_filtered):,}")
male = (df_filtered['Jenis Kelamin']=='Laki-Laki').sum()
female = (df_filtered['Jenis Kelamin']=='Perempuan').sum()
k3.metric("L / P", f"{male} / {female}")
k4.metric("Baris Koordinat", f"{df_filtered['_has_coords'].sum():,}")
k5.metric("Rata-rata Umur", f"{int(df['Umur'].dropna().mean()) if df['Umur'].dropna().any() else 'N/A'}")

# Distribution Pendidikan
st.markdown("### Distribusi Pendidikan")
edu_counts = df['Pendidikan'].fillna('missing').value_counts().rename_axis('Pendidikan').reset_index(name='count')
edu_counts = dedupe_columns(edu_counts)
if not edu_counts.empty:
    st.altair_chart(alt.Chart(edu_counts).mark_bar().encode(x=alt.X('Pendidikan:N', sort='-y', title='Pendidikan'), y=alt.Y('count:Q', title='Jumlah'), tooltip=['Pendidikan','count']).properties(height=260), use_container_width=True)

# Agama
st.subheader("Agama")
agama_counts = df_filtered['Agama'].value_counts().rename_axis('Agama').reset_index(name='count')
if not agama_counts.empty:
    agama_counts = dedupe_columns(agama_counts)
    chart_agama = alt.Chart(agama_counts).mark_bar().encode(
        x=alt.X('Agama:N', sort='-y', title='Agama'),
        y=alt.Y('count:Q', title='Jumlah'),
        tooltip=['Agama','count']
    ).properties(height=220)
    st.altair_chart(chart_agama, use_container_width=True)
else:
    st.write("Tidak ada data agama setelah filter.")

# Jenis Kelamin pie-like bar
st.subheader("Jenis Kelamin")
jk_counts = df_filtered['Jenis Kelamin'].fillna('Unknown').value_counts().rename_axis('JenisKelamin').reset_index(name='count')
if not jk_counts.empty:
    jk_counts = dedupe_columns(jk_counts)
    jk_chart = alt.Chart(jk_counts).mark_bar().encode(
        x=alt.X('JenisKelamin:N', title='Jenis Kelamin'),
        y=alt.Y('count:Q', title='Jumlah'),
        tooltip=['JenisKelamin','count']
    ).properties(height=200)
    st.altair_chart(jk_chart, use_container_width=True)

# Umur (binned)
st.subheader("Distribusi Umur (kelompok 10 tahun)")
# buat bins 0-9,10-19,...,90+
bins = list(range(0, 100, 10)) + [200]  # last bin 100+
labels = [f"{i}-{i+9}" for i in range(0,90,10)] + ["90+"]
# handle missing umur
umur_series = df_filtered['Umur'].copy()
# NaN -> mark as 'Unknown'
umur_valid = umur_series.dropna()
if umur_valid.empty:
    st.info("Tidak ada data umur setelah filter.")
else:
    umur_binned = pd.cut(umur_series.fillna(-1), bins=[-1]+bins, labels=["Unknown"]+labels, right=False)
    age_counts = umur_binned.value_counts().reindex(["Unknown"]+labels, fill_value=0).reset_index()
    age_counts.columns = ['UmurGroup','count']
    age_counts = dedupe_columns(age_counts)
    chart_age = alt.Chart(age_counts).mark_bar().encode(
        x=alt.X('UmurGroup:N', title='Kelompok Umur', sort=None),
        y=alt.Y('count:Q', title='Jumlah'),
        tooltip=['UmurGroup','count']
    ).properties(height=260)
    st.altair_chart(chart_age, use_container_width=True)

# Jenis Pekerjaan (bar chart)
st.subheader("Jenis Pekerjaan")
job_counts = df_filtered['Jenis Pekerjaan'].fillna('missing').value_counts().rename_axis('JenisPekerjaan').reset_index(name='count')
if job_counts.empty:
    st.info("Tidak ada data Jenis Pekerjaan setelah filter.")
else:
    job_counts = dedupe_columns(job_counts)
    job_chart = alt.Chart(job_counts).mark_bar().encode(
        x=alt.X('JenisPekerjaan:N', sort='-y', title='Jenis Pekerjaan'),
        y=alt.Y('count:Q', title='Jumlah'),
        tooltip=['JenisPekerjaan','count']
    ).properties(height=320)
    st.altair_chart(job_chart, use_container_width=True)


# ---------- Modeling with merge rare classes ----------
st.markdown("---")
st.header("Prediksi Jenis Pekerjaan (XGBoost) — Merge Rare Classes")

if not xgb_available:
    st.error("xgboost belum terinstall. Jalankan: pip install xgboost lalu restart app.")
else:
    modeling_df = df[df['Jenis Pekerjaan'].notna()].copy().reset_index(drop=True)
    st.write(f"Baris berlabel tersedia: {len(modeling_df):,}")

    # UI: set min_count threshold
    min_count = st.slider("Ambang minimal frekuensi kelas (kelas dengan count < min_count akan digabung menjadi 'Other')", min_value=2, max_value=50, value=3, step=1)
    st.write("Threshold saat ini:", min_count)

    # show counts before merge
    counts_before = modeling_df['Jenis Pekerjaan'].value_counts()
    st.write("Top 10 kelas (sebelum merge):")
    st.dataframe(counts_before.head(10))

    # merge rare classes
    y_series = modeling_df['Jenis Pekerjaan'].astype(str)
    y_merged, counts_before_full, counts_after = merge_rare_classes(y_series, min_count=min_count, other_label='Other')
    st.write("Top 10 kelas (setelah merge):")
    st.dataframe(counts_after.head(10))

    # Prepare features
    feature_cols = ['Umur','Jenis Kelamin','Pendidikan']
    X = modeling_df[feature_cols].copy()
    # impute numeric
    num_imputer = SimpleImputer(strategy='median')
    X[['Umur']] = num_imputer.fit_transform(X[['Umur']])

    # label encode categorical features
    cat_cols = ['Jenis Kelamin','Pendidikan']
    encoders = {}
    for c in cat_cols:
        X[c] = X[c].fillna('missing').astype(str)
        le = LabelEncoder(); X[c] = le.fit_transform(X[c]); encoders[c] = le

    # label encode target AFTER merging
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y_merged)

    # check enough data
    if len(y_encoded) < 30 or len(np.unique(y_encoded)) < 2:
        st.warning("Data kurang besar/variasi label untuk training yang reliable (butuh >=30 baris berlabel dan >=2 kelas setelah merge).")
        can_train = False
    else:
        can_train = True

    # training controls
    col_a, col_b = st.columns([1,3])
    with col_a:
        test_size = st.slider("Proporsi test set", 0.1, 0.5, 0.25, step=0.05, key='testsize_merge')
        train_btn = st.button("Latih Model XGBoost (merge rare)")
    with col_b:
        st.write("Model prototype. 'Other' mewakili gabungan kelas langka.")

    model = None
    if can_train and train_btn:
        # stratify using y_merged (string series)
        X_train, X_test, y_train, y_test = train_test_split(X, y_merged, test_size=test_size, stratify=y_merged, random_state=42)
        # need numeric y for xgboost -> encode y_train/y_test with le_target
        y_train_enc = le_target.transform(y_train)
        y_test_enc = le_target.transform(y_test)

        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
        with st.spinner("Training XGBoost..."):
            model.fit(X_train, y_train_enc)

        y_pred_enc = model.predict(X_test)
        # decode preds
        y_pred = le_target.inverse_transform(y_pred_enc)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Training selesai. Accuracy (test): {acc:.4f}")
        st.text("Classification report:")
        st.text(classification_report(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=le_target.classes_)
        cm_df = pd.DataFrame(cm, index=le_target.classes_, columns=le_target.classes_)
        st.write("Confusion matrix (counts):")
        st.dataframe(cm_df)

        # feature importance
        try:
            fi = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
            st.subheader("Feature importance")
            st.altair_chart(alt.Chart(fi).mark_bar().encode(x='importance:Q', y=alt.Y('feature:N', sort='-x'), tooltip=['feature','importance']).properties(height=240), use_container_width=True)
        except Exception:
            st.warning("Gagal tampilkan feature importance.")

        # persist to session
        st.session_state['xgb_model'] = model
        st.session_state['xgb_le_target'] = le_target
        st.session_state['xgb_encoders'] = encoders
        st.session_state['xgb_num_imputer'] = num_imputer
        st.session_state['xgb_feature_cols'] = feature_cols
        st.session_state['xgb_y_merged_values'] = y_merged

    # load model if exists
    if 'xgb_model' in st.session_state:
        model = st.session_state['xgb_model']
        le_target = st.session_state['xgb_le_target']
        encoders = st.session_state['xgb_encoders']
        num_imputer = st.session_state['xgb_num_imputer']
        feature_cols = st.session_state['xgb_feature_cols']
        st.info("Model XGBoost aktif (dari session).")

    # download model artifact
    if model is not None:
        if st.button("Download model (.joblib)"):
            artifact = {
                'model': model,
                'le_target': le_target,
                'encoders': encoders,
                'num_imputer': num_imputer,
                'feature_cols': feature_cols
            }
            joblib.dump(artifact, "xgb_jenis_pekerjaan_merge.joblib")
            with open("xgb_jenis_pekerjaan_merge.joblib","rb") as f:
                st.download_button("Klik untuk unduh model", data=f, file_name="xgb_jenis_pekerjaan_merge.joblib", mime="application/octet-stream")

    # single-row prediction UI
    st.markdown("#### Uji Prediksi Satu Baris (manual)")
    with st.form("single_pred"):
        c1,c2,c3 = st.columns(3)
        with c1:
            inp_umur = st.number_input("Umur", min_value=0, max_value=120, value=30)
            inp_jk = st.selectbox("Jenis Kelamin", options=sorted(df['Jenis Kelamin'].dropna().unique().tolist()))
        with c2:
            inp_pendidikan = st.selectbox("Pendidikan", options=sorted(df['Pendidikan'].dropna().unique().tolist()))
            submit_pred = st.form_submit_button("Predict")
        with c3:
            st.write("")

    if 'submit_pred' in locals() and submit_pred:
        if model is None:
            st.warning("Model belum tersedia. Latih model dulu atau load model dari file.")
        else:
            x_single = pd.DataFrame([{
                'Umur': inp_umur,
                'Jenis Kelamin': inp_jk if inp_jk is not None else 'missing',
                'Pendidikan': inp_pendidikan if inp_pendidikan is not None else 'missing'
            }])
            # impute num
            x_single[['Umur']] = num_imputer.transform(x_single[['Umur']])
            # encode cats
            for c, le in encoders.items():
                val = str(x_single.iloc[0][c])
                if val not in le.classes_:
                    if 'missing' in le.classes_:
                        mapped = le.transform(['missing'])[0]
                    else:
                        classes = list(le.classes_) + [val]
                        le.classes_ = np.array(classes)
                        mapped = le.transform([val])[0]
                else:
                    mapped = le.transform([val])[0]
                x_single[c] = mapped
            X_ready = x_single[feature_cols]
            pred_enc = model.predict(X_ready)[0]
            pred_label = le_target.inverse_transform([pred_enc])[0]
            st.success(f"Prediksi Jenis Pekerjaan: **{pred_label}**")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_ready)[0]
                prob_df = pd.DataFrame({'label': le_target.inverse_transform(range(len(probs))), 'prob': probs})
                st.table(prob_df.sort_values('prob', ascending=False).head(8))

# ---------- Table & Export ----------
st.markdown("---")
st.header("Tabel & Export")
display_cols = ["Nama Lengkap","Nama Kepala Keluarga","Tahun Lahir","Umur","Jenis Kelamin","Agama","Pendidikan","Jenis Pekerjaan","RT","RW","Longitude","Latitude"]
st.dataframe(dedupe_columns(df[display_cols].head(500)), use_container_width=True, height=300)
csv_bytes = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV Bersih", data=csv_bytes, file_name="sensus_bersih.csv", mime="text/csv")

st.markdown("---")
st.write("Catatan: Strategi merge membuat kelas minor menjadi 'Other' sehingga stabil untuk splitting dan training. Namun ini mengurangi granularitas prediksi—pertimbangkan trade-off ini sebelum pakai model untuk keputusan sensitif.")
