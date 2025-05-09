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

# --- CREATE TABS ---
tabs = st.tabs([
    "📘 Instructions & Upload",
    "📊 Scores & Heatmap",
    "🧑‍💼 Auditor Overview"
])

# --- TAB 0: INSTRUCTIONS & UPLOAD ---
with tabs[0]:
    st.title("🧼 Cleaning Audit Dashboard")
    st.markdown("### Upload the Excel file with multiple monthly sheets (.xlsx)")

    # Simple username/password gate
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if not (username == "andron" and password == "andron25"):
        st.error("Invalid username or password")
        st.stop()

    uploaded_file = st.file_uploader("Upload your file here", type=["xlsx"])
    with st.expander("📌 How to use", expanded=True):
        st.markdown("""
        1. Authenticate with your username and password.  
        2. Each sheet must include columns:  
           `Date Completed, Site, Answered by, Percentage Received, Score, Questionnaire Result, Yes, No, N/A`.  
        3. Avoid extra summary sheets.  
        4. Download the template if needed:
        """)
        try:
            with open("audit_template.xlsx", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<a href="data:application/octet-stream;base64,{b64}" download="audit_template.xlsx">📥 Download Template</a>',
                    unsafe_allow_html=True
                )
        except FileNotFoundError:
            st.info("Template not found on server.")

# --- MAIN WORKFLOW ---
if uploaded_file:
    try:
        # 1. Load & validate all sheets
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

        # 2. Cleaning & Clustering pipeline (as per notebook)
        def normalize_name(s: str) -> str:
            s = str(s).lower().strip()
            # collapse numeric ranges
            s = re.sub(r"(\d+)\s*-\s*(\d+)",
                       lambda m: str(max(int(m.group(1)), int(m.group(2)))), s)
            # street → st
            s = re.sub(r"\bstreet\b", "st", s)
            # remove punctuation
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            # ensure space between digit↔letter and letter↔digit
            s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
            s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
            return re.sub(r"\s+", " ", s).strip()

        df_all["site_norm"] = df_all["Site"].apply(normalize_name)

        # Numeric‐Prefix Clustering
        mask_num = df_all["site_norm"].str.match(r"^\d+")
        nums = df_all.loc[mask_num, "site_norm"]
        collapsed = (
            nums
            .str.replace(r"^(\d+)[\s-]+(\d+)",
                         lambda m: str(max(int(m.group(1)), int(m.group(2)))),
                         regex=True)
            .str.strip()
        )
        df_all.loc[mask_num, "prefix"] = collapsed.str.extract(r"^(\d+)", expand=False)
        rest = collapsed.str.replace(r"^\d+\s+", "", regex=True).str.strip()
        rest = rest.str.replace(r"\s+st$", "", regex=True)
        rest = rest.str.replace(r"['’]s\b", "", regex=True)

        # avoid rest == prefix
        df_all.loc[mask_num & (rest == df_all.loc[mask_num, "prefix"]), "suffix_primary"] = ""
        def first_non_st(tokens):
            for t in tokens:
                if t != "st":
                    return t
            return ""
        primary = rest.str.split().apply(first_non_st)
        df_all.loc[mask_num, "suffix_primary"] = primary

        # special fix for "72"
        df_all.loc[
            mask_num & (df_all["prefix"]=="72") & df_all["suffix_primary"].eq(""),
            "suffix_primary"
        ] = "jermyn"

        modes = (
            df_all[mask_num & df_all["suffix_primary"].ne("")]
                .groupby("prefix")["suffix_primary"]
                .agg(lambda s: s.mode()[0])
                .to_dict()
        )
        df_all.loc[mask_num, "suffix_primary"] = (
            df_all.loc[mask_num]
                .apply(lambda r: r["suffix_primary"] or modes.get(r["prefix"], ""), axis=1)
        )

        df_all["cluster1"] = pd.NA
        valid = mask_num & df_all["suffix_primary"].ne("")
        df_all.loc[valid, "cluster1"] = (
            df_all.loc[valid, "prefix"] + " " + df_all.loc[valid, "suffix_primary"]
        )
        df_all.drop(columns=["prefix","suffix_primary"], inplace=True)

        # Fuzzy‐Clustering for the rest
        raw = df_all.loc[df_all["cluster1"].isna(), "site_norm"].unique().tolist()
        keys = [re.sub(r"^\d+\s*", "", n) for n in raw]
        parent = list(range(len(raw)))
        def find2(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union2(a, b):
            ra, rb = find2(a), find2(b)
            if ra != rb:
                parent[rb] = ra

        TH = 90
        for i, k in enumerate(keys):
            for j in range(i+1, len(keys)):
                if fuzz.token_sort_ratio(k, keys[j]) >= TH:
                    union2(i, j)

        clusters2 = defaultdict(list)
        for idx, name in enumerate(raw):
            clusters2[find2(idx)].append(name)

        freq2 = df_all["site_norm"].value_counts().to_dict()
        def pick_representative(cl):
            with_num = [n for n in cl if re.match(r"^\d+\s", n)]
            return with_num[0] if with_num else max(cl, key=lambda x: freq2.get(x, 0))

        canon2 = {}
        for cl in clusters2.values():
            rep = pick_representative(cl)
            for v in cl:
                canon2[v] = rep

        # Final Site_clean assembly
        def ensure_st(name: str) -> str:
            parts = name.split()
            return f"{parts[0]} st {' '.join(parts[1:])}".strip() if parts and parts[0].isdigit() else name

        df_all["Site_clean"] = pd.NA
        mask1 = df_all["cluster1"].notna()
        df_all.loc[mask1, "Site_clean"] = df_all.loc[mask1, "cluster1"].apply(ensure_st)
        mask2 = df_all["Site_clean"].isna()
        df_all.loc[mask2, "Site_clean"] = df_all.loc[mask2, "site_norm"].map(canon2)

        df_all["Site_clean"] = (
            df_all["Site_clean"]
              .str.replace(r"\bst\b", "", regex=True)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip()
        )
        df_all.drop(columns=["site_norm","cluster1"], inplace=True)

        # 3. Feature Engineering & Outliers
        df_all['Answered by'] = df_all['Answered by'].str.strip().str.title()
        df_all['Date Completed'] = pd.to_datetime(df_all['Date Completed'], errors="coerce")
        df_all['Month'] = df_all['Date Completed'].dt.month.map(
            lambda m: calendar.month_abbr[int(m)] if pd.notnull(m) else None
        )
        df_all['Valid Questions'] = df_all['Yes'] + df_all['No']
        df_all['Calculated Score'] = np.where(
            df_all['Valid Questions'] > 0,
            df_all['Yes'] / df_all['Valid Questions'] * 100,
            np.nan
        )
        score_nums = df_all['Score'].str.split('/', expand=True).astype(float)
        df_all['Score_num'], df_all['Score_den'] = score_nums[0], score_nums[1]
        mask_high = df_all['Percentage Received'] > 100
        df_all.loc[mask_high, 'Percentage Received'] = (
            df_all.loc[mask_high,'Yes'] / df_all.loc[mask_high,'Valid Questions'] * 100
        )
        q1, q3 = df_all['Percentage Received'].quantile([0.25,0.75])
        lb = q1 - 1.5*(q3-q1)
        df_all['is_outlier_low'] = df_all['Percentage Received'] < lb

        # 4. Audit‐level Classification
        def classify_score(s):
            if pd.isna(s): return 'Not Enough Data'
            if s >= 80:    return 'Approved'
            if s >= 70:    return 'Acceptable'
            return 'Critical'

        df_all['Evaluation'] = df_all.apply(
            lambda r: 'Not Enough Data' if r['Valid Questions'] <= 5 else classify_score(r['Calculated Score']),
            axis=1
        )

        # --- TAB 1: Scores & Heatmap ---
        with tabs[1]:
            st.subheader("📊 Average Score by Site")

            # --- Mode toggle & month select ---
            mode = st.radio("View mode", ["Cumulative", "Monthly"], horizontal=True)
            if mode == "Monthly":
                month_list = sorted(
                    df_all['Month'].dropna().unique(),
                    key=lambda m: list(calendar.month_abbr).index(m)
                )
                sel_plot_month = st.selectbox("Select Month", month_list)

            # --- Sidebar filters ---
            st.sidebar.header("Filters")
            date_min, date_max = df_all['Date Completed'].min(), df_all['Date Completed'].max()
            sel_date = st.sidebar.date_input("Date Range", [date_min, date_max],
                                             min_value=date_min, max_value=date_max)

            with st.sidebar.expander("Sites", expanded=False):
                sel_sites = st.multiselect("Select Sites",
                                           sorted(df_all['Site_clean'].unique()),
                                           default=sorted(df_all['Site_clean'].unique()))
            with st.sidebar.expander("Evaluation", expanded=False):
                sel_evals = st.multiselect("Select Evaluation",
                                           sorted(df_all['Evaluation'].unique()),
                                           default=sorted(df_all['Evaluation'].unique()))
            with st.sidebar.expander("Auditors", expanded=False):
                sel_users = st.multiselect("Answered by",
                                           sorted(df_all['Answered by'].unique()),
                                           default=sorted(df_all['Answered by'].unique()))

            # --- Build filtered df_plot ---
            if mode == "Monthly":
                df_plot = df_all[df_all['Month'] == sel_plot_month]
            else:
                df_plot = df_all.copy()

            df_plot = df_plot[
                (df_plot['Date Completed'] >= pd.to_datetime(sel_date[0])) &
                (df_plot['Date Completed'] <= pd.to_datetime(sel_date[1])) &
                df_plot['Site_clean'].isin(sel_sites) &
                df_plot['Evaluation'].isin(sel_evals) &
                df_plot['Answered by'].isin(sel_users)
            ]

            # --- Top/Bottom/All + slider ---
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

            max_n = df_plot['Site_clean'].nunique()
            n = st.slider("Number of Sites", min_value=5, max_value=max_n, value=10, step=1)

            # --- Compute avg & classify per site ---
            avg = df_plot.groupby('Site_clean')['Calculated Score'].mean()
            if view == 'top':
                avg = avg.sort_values(ascending=False).head(n)
            elif view == 'bottom':
                avg = avg.sort_values().head(n)
            else:
                avg = avg.sort_values()

            df_bar = avg.reset_index().rename(columns={'Calculated Score':'Score'})
            total_valids = df_plot.groupby('Site_clean')['Valid Questions'].sum()
            df_bar['Total_Valid_Questions'] = df_bar['Site_clean'].map(total_valids)

            def classify_site(r):
                if r['Total_Valid_Questions'] <= 5: return 'Not Enough Data'
                if r['Score'] >= 80: return 'Approved'
                if r['Score'] >= 70: return 'Acceptable'
                return 'Critical'

            df_bar['Evaluation'] = df_bar.apply(classify_site, axis=1)
            cmap = {
                'Approved':'#2ecc71',
                'Acceptable':'#f1c40f',
                'Critical':'#e74c3c',
                'Not Enough Data':'#95a5a6'
            }

            fig1 = px.bar(
                df_bar, x='Score', y='Site_clean', orientation='h',
                color='Evaluation', color_discrete_map=cmap,
                hover_data={'Score':':.1f','Total_Valid_Questions':True}
            )
            fig1.add_vline(x=80, line_dash='dash', line_color='green',
                           annotation_text='Approved (80%)', annotation_position='top left')
            fig1.add_vline(x=70, line_dash='dash', line_color='orange',
                           annotation_text='Acceptable (70%)', annotation_position='bottom left')
            fig1.add_vline(x=0,  line_dash='dash', line_color='red',
                           annotation_text='Critical (<70%)', annotation_position='bottom left')
            fig1.update_layout(
                title=f'Top & Bottom {n} Sites – {mode}',
                xaxis_title='Score (%)',
                xaxis=dict(range=[0,100]),
                height=400 + n*25,
                margin=dict(l=200, r=40, t=80, b=40)
            )
            st.plotly_chart(fig1, use_container_width=True)

            # --- 7) Site Evolution ---
            st.subheader("📈 Site Evolution")
            site_list = sorted(df_all['Site_clean'].unique())
            sel_site = st.selectbox("Select Site", site_list)
            evo_mode = st.radio("Evolution Type", ["Monthly","Cumulative"], horizontal=True)
            df_evo = df_plot[df_plot['Site_clean'] == sel_site]
            months = sorted(df_all['Month'].dropna().unique(),
                            key=lambda m: list(calendar.month_abbr).index(m))
            evo_df = df_evo.groupby('Month')['Calculated Score'].mean().reindex(months).reset_index()
            if evo_mode == "Cumulative":
                evo_df['Cumulative'] = evo_df['Calculated Score'].expanding().mean()
                y_col = 'Cumulative'
            else:
                y_col = 'Calculated Score'
            fig_evo = px.line(evo_df, x='Month', y=y_col, markers=True, labels={y_col:'Score (%)'})
            fig_evo.update_layout(title=f'{sel_site} Score Evolution ({evo_mode})', yaxis=dict(range=[0,100]))
            st.plotly_chart(fig_evo, use_container_width=True)

            # --- 8) Heatmap of Monthly Scores ---
            st.subheader("🔥 Heatmap of Monthly Scores")
            overall = df_plot.groupby('Site_clean')['Calculated Score'].mean().sort_values()
            top10 = overall.tail(10).index.tolist()
            bot10 = overall.head(10).index.tolist()
            order = top10 + bot10
            pivot = (
                df_plot[df_plot['Site_clean'].isin(order)]
                       .pivot_table('Calculated Score','Site_clean','Month',aggfunc='mean')
                       .reindex(columns=months, index=order)
            )
            fig2 = px.imshow(
                pivot, text_auto='.1f', aspect='auto',
                color_continuous_scale='RdYlGn', range_color=[0,100],
                labels={'x':'Month','y':'Site','color':'Score (%)'}
            )
            fig2.update_layout(height=400 + len(order)*25, margin=dict(l=200,r=40,t=40,b=40))
            st.plotly_chart(fig2, use_container_width=True)

            # --- 9) % Received vs Calculated Score ---
            st.subheader("% Received vs Calculated Score")
            sc = df_plot.dropna(subset=['Percentage Received','Calculated Score'])
            if len(sc) > 1:
                fig3 = px.scatter(
                    sc, x='Percentage Received', y='Calculated Score',
                    hover_data=['Answered by'], labels={'x':'% Received','y':'Score (%)'}
                )
                slope, intercept, r_val, *_ = linregress(sc['Percentage Received'], sc['Calculated Score'])
                x_line = np.linspace(0,100,100)
                fig3.add_trace(
                    go.Scatter(x=x_line, y=slope*x_line+intercept, mode='lines',
                               line=dict(color='red', dash='dash'),
                               name=f'Fit (R²={r_val**2:.2f})')
                )
                fig3.add_hline(y=80, line_dash='dash', line_color='green', annotation_text='Approved (80%)')
                fig3.add_hline(y=70, line_dash='dash', line_color='orange', annotation_text='Acceptable (70%)')
                fig3.add_hline(y=0,  line_dash='dash', line_color='red', annotation_text='Critical (<70%)')
                fig3.update_layout(dragmode='zoom', height=500, margin=dict(l=40,r=40,t=40,b=40))
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("Insufficient data for regression.")

            # --- 10) Executive Summary ---
            st.subheader("📝 Executive Summary")
            st.markdown(f"- **Total Audits Displayed:** {len(df_plot)}")
            st.markdown(f"- **Unique Sites Displayed:** {df_plot['Site_clean'].nunique()}")
            eval_counts = df_plot['Evaluation'].value_counts().to_dict()
            for ev, cnt in eval_counts.items():
                st.markdown(f"  - **{ev}:** {cnt}")

        # --- TAB 2: Auditor Overview ---
        with tabs[2]:
            st.subheader("🧑‍💼 Auditor Overview")
            month_options = ["All"] + sorted(
                df_all['Month'].dropna().unique(),
                key=lambda m: list(calendar.month_abbr).index(m)
            )
            sel_aud_month = st.selectbox("Select Month", month_options)
            auditor_list = sorted(df_all['Answered by'].dropna().unique())
            sel_auditor = st.selectbox("Answered by", auditor_list)

            df_aud = df_all.copy()
            if sel_aud_month != "All":
                df_aud = df_aud[df_aud['Month'] == sel_aud_month]
            df_aud = df_aud[df_aud['Answered by'] == sel_auditor]

            total_audits = len(df_aud)
            st.markdown(f"### ✅ Total Audits: {total_audits}")

            if sel_aud_month == "All":
                freq = (
                    df_aud.groupby(['Site_clean','Month'])
                          .size()
                          .reset_index(name='Count')
                          .sort_values(['Site_clean','Month'])
                )
                st.markdown("#### Audits by Site & Month")
                st.dataframe(freq)
            else:
                freq = (
                    df_aud['Site_clean']
                         .value_counts()
                         .rename_axis('Site_clean')
                         .reset_index(name='Count')
                )
                st.markdown(f"#### Audits by Site in {sel_aud_month}")
                st.dataframe(freq)

            st.markdown("#### 📊 Audit Count by Site")
            cnts = df_aud['Site_clean'].value_counts().sort_index()
            fig4 = px.bar(
                x=cnts.values, y=cnts.index,
                orientation='h', labels={'x':'Audit Count','y':'Site'},
                text=cnts.values
            )
            fig4.update_traces(textposition='outside')
            fig4.update_layout(height=max(300,len(cnts)*25), margin=dict(l=200,r=40,t=40,b=40))
            st.plotly_chart(fig4, use_container_width=True)

        # --- DOWNLOAD FILTERED DATA AS CSV ---
        st.download_button(
            "📥 Download Filtered Data as CSV",
            data=df_plot.to_csv(index=False),
            file_name="filtered_audits.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.text(traceback.format_exc())

else:
    # If no file yet, show a prompt on Tab 1
    with tabs[1]:
        st.info("👆 Upload your Excel file to begin.")
