import streamlit as st
import pandas as pd
import numpy as np
import calendar
import re
from collections import defaultdict
from scipy.stats import linregress
from rapidfuzz import fuzz
import plotly.express as px
from plotly import graph_objects as go
import base64
import traceback

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cleaning Audit Dashboard", layout="wide")

# --- TABS ---
tabs = st.tabs(["📘 Instructions & Upload", "📊 Scores & Heatmap", "🧑‍💼 Auditor Overview"])

# --- TAB 0: INSTRUCTIONS & UPLOAD ---
with tabs[0]:
    st.title("🧼 Cleaning Audit Dashboard")
    st.markdown("### Upload the Excel file with multiple monthly sheets (.xlsx)")
    # Authentication
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if not (username == "andron" and password == "andron25"):
        st.error("Invalid username or password")
        st.stop()
    uploaded_file = st.file_uploader("Upload your file here", type=["xlsx"]);

    with st.expander("📌 How to use", expanded=True):
        st.markdown(
            """
            1. Authenticate with your username and password.  
            2. Each sheet must include columns:  
               `Date Completed, Site, Answered by, Percentage Received, Score, Questionnaire Result, Yes, No, N/A`.  
            3. Avoid extra summary sheets.  
            4. Download the template if needed:  
            """
        )
        try:
            with open("audit_template.xlsx", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<a href="data:application/octet-stream;base64,{b64}" download="audit_template.xlsx">📥 Download Template</a>',
                    unsafe_allow_html=True
                )
        except FileNotFoundError:
            st.info("Template not found on server.")

# --- PROCESSING AFTER UPLOAD ---
if uploaded_file:
    try:
        # 1. Load & validate sheets
        @st.cache_data
        def load_clean_sheet(file, sheet_name: str) -> pd.DataFrame:
            df = pd.read_excel(file, sheet_name=sheet_name)
            df.columns = (
                df.columns
                  .str.strip()
                  .str.replace(r"\s+", " ", regex=True)
                  .str.replace("Questionarie Result", "Questionnaire Result", regex=False)
            )
            expected = {
                "Date Completed","Site","Answered by",
                "Percentage Received","Score","Questionnaire Result",
                "Yes","No","N/A"
            }
            missing = expected - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns in sheet '{sheet_name}': {missing}")
            return df

        xls = pd.ExcelFile(uploaded_file)
        df_all = pd.concat(
            [load_clean_sheet(uploaded_file, sh) for sh in xls.sheet_names],
            ignore_index=True
        )

        # 2. Cleaning & Clustering (pipeline from notebook)
        def normalize_name(s: str) -> str:
            s = str(s).lower().strip()
            s = re.sub(r"(\d+)\s*-\s*(\d+)", lambda m: str(max(int(m.group(1)),int(m.group(2)))), s)
            s = re.sub(r"\bstreet\b", "st", s)
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            return re.sub(r"\s+", " ", s).strip()

        df_all["site_norm"] = df_all["Site"].apply(normalize_name)
        mask_num = df_all['site_norm'].str.match(r'^\d+')
        nums = df_all.loc[mask_num, 'site_norm']
        collapsed = nums.str.replace(r'^(\d+)[\s-]+(\d+)',
                                     lambda m: str(max(int(m.group(1)), int(m.group(2)))),
                                     regex=True).str.strip()
        df_all.loc[mask_num, 'prefix'] = collapsed.str.extract(r'^(\d+)', expand=False)
        df_all.loc[mask_num, 'suffix_primary'] = (
            collapsed.str.replace(r'^\d+\s*', '', regex=True)
                     .str.replace(r"'?s\b", '', regex=True)
                     .str.replace(r'\bst\b', '', regex=True)
                     .str.strip().str.split().str[0]
        )
        df_all.loc[
          mask_num & (df_all['prefix']=='72') & df_all['suffix_primary'].fillna('').eq(''),
          'suffix_primary'] = 'jermyn'
        modes = (df_all[mask_num & df_all['suffix_primary'].ne('')]
                 .groupby('prefix')['suffix_primary']
                 .agg(lambda s: s.mode()[0]).to_dict())
        df_all.loc[mask_num, 'suffix_primary'] = (
            df_all.loc[mask_num].apply(lambda r: r['suffix_primary'] or modes.get(r['prefix'], ''), axis=1)
        )
        df_all['cluster1'] = pd.NA
        valid = mask_num & df_all['suffix_primary'].ne('')
        df_all.loc[valid, 'cluster1'] = (
            df_all.loc[valid, 'prefix'] + ' ' + df_all.loc[valid, 'suffix_primary']
        )
        df_all.drop(columns=['prefix','suffix_primary'], inplace=True)
        others = df_all.loc[df_all['cluster1'].isna(), 'site_norm'].unique().tolist()
        idx2 = {n:i for i,n in enumerate(others)}; parent = list(range(len(others)))
        def find2(i):
            while parent[i]!=i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        def union2(a,b):
            ra, rb = find2(a), find2(b)
            if ra!=rb: parent[rb] = ra
        TH = 90
        for i, ni in enumerate(others):
            for j in range(i+1, len(others)):
                if fuzz.token_sort_ratio(ni, others[j]) >= TH:
                    union2(i,j)
        clusters2 = defaultdict(list)
        for name,i in idx2.items(): clusters2[find2(i)].append(name)
        freq2 = df_all['site_norm'].value_counts().to_dict(); canon2 = {}
        for cl in clusters2.values():
            rep = max(cl, key=lambda x: freq2.get(x,0))
            for v in cl: canon2[v] = rep
        def ensure_st(name: str) -> str:
            parts = name.split()
            if parts and parts[0].isdigit(): return f"{parts[0]} st {' '.join(parts[1:])}".strip()
            return name
        df_all['Site_clean'] = pd.NA
        mask_num_clean = df_all['cluster1'].notna()
        df_all.loc[mask_num_clean, 'Site_clean'] = df_all.loc[mask_num_clean, 'cluster1'].apply(ensure_st)
        mask_fuzzy = df_all['Site_clean'].isna()
        df_all.loc[mask_fuzzy, 'Site_clean'] = df_all.loc[mask_fuzzy, 'site_norm'].map(canon2)
        df_all.loc[df_all['Site_clean']=='72 st','Site_clean'] = '72 st jermyn'
        df_all.drop(columns=['site_norm','cluster1'], inplace=True)

        df_all['Answered by'] = df_all['Answered by'].str.strip().str.title()
        df_all['Date Completed'] = pd.to_datetime(df_all['Date Completed'], errors='coerce')
        df_all['Month'] = df_all['Date Completed'].dt.month.map(
            lambda m: calendar.month_abbr[int(m)] if pd.notnull(m) else None
        )
        df_all['Valid Questions'] = df_all['Yes'] + df_all['No']
        df_all['Calculated Score'] = np.where(
            df_all['Valid Questions']>0,
            df_all['Yes']/df_all['Valid Questions']*100,
            np.nan
        )
        score_nums = df_all['Score'].str.split('/', expand=True).astype(float)
        df_all['Score_num'], df_all['Score_den'] = score_nums[0], score_nums[1]
        mask_high = df_all['Percentage Received'] > 100
        df_all.loc[mask_high, 'Percentage Received'] = (
            df_all.loc[mask_high, 'Yes']/df_all['Valid Questions']*100
        )
        q1, q3 = df_all['Percentage Received'].quantile([0.25,0.75])
        lb = q1 - 1.5*(q3-q1)
        df_all['is_outlier_low'] = df_all['Percentage Received'] < lb
        def classify_score(s):
            if pd.isna(s): return 'Not Applicable'
            if s>=80: return 'Approved'
            if s>=60: return 'Acceptable'
            return 'Critical'
        df_all['Evaluation'] = df_all.apply(
            lambda r: 'Not Enough Data' if r['Valid Questions']<=5 else classify_score(r['Calculated Score']),
            axis=1
        )

        # --- TAB 1: Scores & Heatmap with Plotly ---
        with tabs[1]:
            st.subheader("📊 Average Score by Site")
            # Controls above chart
            col1, col2, col3 = st.columns(3)
            if 'view' not in st.session_state:
                st.session_state.view = 'all'
            if col1.button("Top 10"):
                st.session_state.view = 'top'
            if col2.button("Bottom 10"):
                st.session_state.view = 'bottom'
            if col3.button("All Sites"):
                st.session_state.view = 'all'
            view = st.session_state.view

            # Slider for N sites (only applies to top/bottom)
            unique_sites = sorted(df_all['Site_clean'].unique())
            max_n = len(unique_sites)
            n = st.slider("Number of Sites", min_value=5, max_value=max_n, value=10, step=1)

                        # Sidebar filters
            st.sidebar.header("Filters")
            sel_month = st.sidebar.selectbox("Month", ['All'] + sorted(df_all['Month'].dropna().unique().tolist()))
            date_min = df_all['Date Completed'].min()
            date_max = df_all['Date Completed'].max()
            sel_date = st.sidebar.date_input("Date Range", [date_min, date_max], min_value=date_min, max_value=date_max)
            with st.sidebar.expander("Sites", expanded=False):
                sel_sites = st.multiselect("Select Sites", sorted(df_all['Site_clean'].unique()), default=sorted(df_all['Site_clean'].unique()))
            with st.sidebar.expander("Evaluation", expanded=False):
                sel_evals = st.multiselect("Select Evaluation", sorted(df_all['Evaluation'].unique()), default=sorted(df_all['Evaluation'].unique()))
            with st.sidebar.expander("Auditors", expanded=False):
                sel_users = st.multiselect("Answered by", sorted(df_all['Answered by'].unique()), default=sorted(df_all['Answered by'].unique()))

            # apply filters
            filt = df_all.copy()
            if sel_month != 'All':
                filt = filt[filt['Month'] == sel_month]
            filt = filt[(filt['Date Completed'] >= pd.to_datetime(sel_date[0])) & (filt['Date Completed'] <= pd.to_datetime(sel_date[1]))]
            filt = filt[filt['Site_clean'].isin(sel_sites)]
            filt = filt[filt['Evaluation'].isin(sel_evals)]
            filt = filt[filt['Answered by'].isin(sel_users)]

            avg = filt.groupby('Site_clean')['Calculated Score'].mean()
            if view == 'top':
                avg = avg.sort_values(ascending=False).head(n)
            elif view == 'bottom':
                avg = avg.sort_values().head(n)
            else:
                avg = avg.sort_values()
            df_bar = avg.reset_index().rename(columns={'Calculated Score':'Score'})
            cmap = {'Approved':'#2ecc71','Acceptable':'#f1c40f','Critical':'#e74c3c','Not Enough Data':'#95a5a6'}
            df_bar['Evaluation'] = df_bar['Score'].apply(lambda s: 'Approved' if s>=80 else ('Acceptable' if s>=60 else 'Critical'))
            fig1 = px.bar(
                df_bar, x='Score', y='Site_clean', orientation='h',
                color='Evaluation', color_discrete_map=cmap,
                hover_data={'Score':':.1f', 'Evaluation':True}
            )
            fig1.update_layout(
                xaxis_title='Average Score (%)',
                yaxis_title='Site',
                height=max(400, n*30),
                dragmode='zoom'
            )
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("🔥 Heatmap of Monthly Scores")
            overall = filt.groupby('Site_clean')['Calculated Score'].mean().sort_values()
            top10 = overall.tail(10).index.tolist(); bot10 = overall.head(10).index.tolist()
            order = top10 + bot10
            pivot = (
                filt[filt['Site_clean'].isin(order)]
                    .pivot_table('Calculated Score','Site_clean','Month',aggfunc='mean')
                    .reindex(columns=sorted(filt['Month'].dropna().unique()), index=order)
            )
            fig2 = px.imshow(
                pivot, text_auto='.1f', aspect='auto',
                color_continuous_scale='RdYlGn', range_color=[0,100],
                labels={'x':'Month','y':'Site','color':'Score (%)'}
            )
            fig2.update_layout(height=max(400, len(order)*25), dragmode='zoom')
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("🔍 % Received vs Calculated Score")
            sc = filt.dropna(subset=['Percentage Received','Calculated Score'])
            if len(sc)>1:
                fig3 = px.scatter(
                    sc, x='Percentage Received', y='Calculated Score',
                    hover_data=['Answered by'], labels={'x':'% Received','y':'Score (%)'}
                )
                # add regression line
                slope, intercept, r_val, _, _ = linregress(sc['Percentage Received'], sc['Calculated Score'])
                x_line = np.linspace(0,100,100)
                y_line = slope*x_line + intercept
                fig3.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', line=dict(color='red', dash='dash'),
                                          name=f'Fit (R²={r_val**2:.2f})'))
                fig3.update_layout(xaxis=dict(rangeslider=dict(visible=True)), dragmode='zoom')
                fig3.add_hline(y=80, line_dash='dash', line_color='green', annotation_text='Approved (80%)')
                fig3.add_hline(y=60, line_dash='dash', line_color='orange', annotation_text='Acceptable (60%)')
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("Insufficient data for regression.")

        # --- TAB 2: Auditor Overview ---
        with tabs[2]:
            auditor_list = sorted(filt['Answered by'].dropna().unique())
            aud = st.selectbox("Answered by", auditor_list)
            adf = filt[filt['Answered by'] == aud]
            st.markdown(f"### ✅ Total Audits: {len(adf)}")
            freq = (
                adf.groupby(['Site_clean','Month']).size()
                   .reset_index(name='Count')
                   .sort_values(['Site_clean','Month'])
            )
            st.dataframe(freq)
            cnts = adf['Site_clean'].value_counts().sort_index()
            fig4 = px.bar(x=cnts.values, y=cnts.index, orientation='h', labels={'x':'Audit Count','y':'Site'})
            fig4.update_layout(height=max(400, len(cnts)*25))
            st.plotly_chart(fig4, use_container_width=True)

        # --- DOWNLOAD CSV ---
        st.download_button(
            "📥 Download Filtered Data as CSV",
            data=filt.to_csv(index=False),
            file_name="filtered_audits.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.text(traceback.format_exc())

else:
    with tabs[1]:
        st.info("👆 Upload your Excel file to begin.")
