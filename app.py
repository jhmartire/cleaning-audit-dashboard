# Bibliotecas padrão
import io
import re
import calendar
import base64
from collections import defaultdict

# Bibliotecas de terceiros
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly import graph_objects as go
from fpdf import FPDF
import plotly.io as pio
from scipy.stats import linregress
from rapidfuzz import fuzz


# Tratamento de erros
import traceback


# --- Funções auxiliares ---
def normalize_name(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"(\d+)\s*-\s*(\d+)",
               lambda m: str(max(int(m.group(1)),int(m.group(2)))),
               s)
    s = re.sub(r"\bstreet\b", "st", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def find2(i, parent):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def union2(a, b, parent):
    ra, rb = find2(a, parent), find2(b, parent)
    if ra != rb:
        parent[rb] = ra

def classify_score(s):
    if pd.isna(s):
        return 'Not Enough Data'
    if s >= 80:
        return 'Approved'
    if s >= 70:
        return 'Acceptable'
    return 'Critical'

# --- Page Configuration ---
st.set_page_config(page_title="Cleaning Audit Dashboard", layout="wide")

tabs = st.tabs([
    "📘 Instructions & Upload",
    "📊 Scores & Heatmap",
    "🧑‍💼 Auditor Overview",
    "📈 Executive Dashboard",
    "📌 Monthly Highlights"
])

# --- TAB 0: Upload ---
with tabs[0]:
    st.title("🧼 Cleaning Audit Dashboard")
    st.markdown("### Upload the Excel file (.xlsx) containing monthly audit sheets")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if not (username == "andron" and password == "andron25"):
        st.error("Invalid username or password")
        st.stop()

    uploaded_file = st.file_uploader("Upload your file here", type=["xlsx"])

    with st.expander("📌 How to use", expanded=True):
        st.markdown("""
        1. Authenticate using the provided username and password.  
        2. Each sheet must include the following columns:  
           **Date Completed, Site, Answered by, Percentage Received, Score, Questionnaire Result, Yes, No, N/A**.  
        3. Avoid extra/empty sheets in the file.  
        4. Download the template if needed.
        """)

        try:
            with open("audit_template.xlsx", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<a href="data:application/octet-stream;base64,{b64}" '
                    'download="audit_template.xlsx">📥 Download Template</a>',
                    unsafe_allow_html=True
                )
        except FileNotFoundError:
            st.info("Template not found on server.")

if uploaded_file:
    try:
        @st.cache_data
        def load_clean_sheet(file, sheet_name):
            df = pd.read_excel(file, sheet_name=sheet_name)
            df.columns = (df.columns
                            .str.strip()
                            .str.replace(r"\s+"," ",regex=True)
                            .str.replace("Questionarie Result",
                                         "Questionnaire Result",
                                         regex=False))
            esperado = {"Date Completed","Site","Answered by",
                        "Percentage Received","Score",
                        "Questionnaire Result","Yes","No","N/A"}
            faltam = esperado - set(df.columns)
            if faltam:
                raise ValueError(f"Missing columns em '{sheet_name}': {faltam}")
            return df

        # 1) carrega tudo
        xls = pd.ExcelFile(uploaded_file)
        df_all = pd.concat([load_clean_sheet(uploaded_file, sh)
                            for sh in xls.sheet_names],
                           ignore_index=True)

        # Processa coluna de datas e mês
        df_all['Date Completed'] = pd.to_datetime(df_all['Date Completed'], errors='coerce')
        df_all['Month'] = df_all['Date Completed'].dt.strftime('%b')  # útil para os filtros mensais
        

        # 🔄 Botão para resetar filtros 
        if st.sidebar.button("🔄 Reset Filters"):
            st.rerun()
        
        # 2) normaliza + clusteriza
        df_all["site_norm"] = df_all["Site"].apply(normalize_name)
        mask_num = df_all['site_norm'].str.match(r'^\d+')
        nums = df_all.loc[mask_num,'site_norm']
        collapsed = nums.str.replace(
            r'^(\d+)[\s-]+(\d+)',
            lambda m: str(max(int(m.group(1)),int(m.group(2)))),
            regex=True
        ).str.strip()

        df_all.loc[mask_num,'prefix'] = collapsed.str.extract(r'^(\d+)',expand=False)
        df_all.loc[mask_num,'suffix_primary'] = (
            collapsed
              .str.replace(r'^\d+\s*','',regex=True)
              .str.replace(r"'s\b","",regex=True)
              .str.replace(r'\bst\b','',regex=True)
              .str.strip().str.split().str[0]
        )
        # fix 72
        df_all.loc[
          mask_num & (df_all['prefix']=='72') &
          df_all['suffix_primary'].fillna('').eq(''),
          'suffix_primary'
        ] = 'jermyn'

        modos = (df_all[mask_num & df_all['suffix_primary'].ne('')]
                  .groupby('prefix')['suffix_primary']
                  .agg(lambda s: s.mode()[0])
                  .to_dict())
        df_all.loc[mask_num,'suffix_primary'] = (
            df_all.loc[mask_num]
              .apply(lambda r: r['suffix_primary'] or modos.get(r['prefix'],''),axis=1)
        )
        df_all['cluster1'] = pd.NA
        valid = mask_num & df_all['suffix_primary'].ne('')
        df_all.loc[valid,'cluster1'] = (
            df_all.loc[valid,'prefix'] + ' ' +
            df_all.loc[valid,'suffix_primary']
        )
        df_all.drop(columns=['prefix','suffix_primary'],inplace=True)

        others = df_all.loc[df_all['cluster1'].isna(),'site_norm'].unique().tolist()
        idx2,parent = {n:i for i,n in enumerate(others)}, list(range(len(others)))
        for i,ni in enumerate(others):
            for j in range(i+1,len(others)):
                if fuzz.token_sort_ratio(ni,others[j])>=90:
                    union2(i,j,parent)

        clusters2=defaultdict(list)
        for name,i in idx2.items():
            clusters2[find2(i,parent)].append(name)
        freq2=df_all['site_norm'].value_counts().to_dict()
        canon2={}
        for cl in clusters2.values():
            rep=max(cl,key=lambda x:freq2.get(x,0))
            for v in cl: canon2[v]=rep

        df_all['Site_clean']=pd.NA
        df_all.loc[df_all['cluster1'].notna(),'Site_clean']=df_all.loc[
          df_all['cluster1'].notna(),'cluster1']
        df_all.loc[df_all['Site_clean'].isna(),'Site_clean']=df_all.loc[
          df_all['Site_clean'].isna(),'site_norm'].map(canon2)
        df_all['Site_clean']=(
            df_all['Site_clean']
              .str.replace(r"\bst\b","",regex=True)
              .str.replace(r"\s+"," ",regex=True)
              .str.strip()
        )
        df_all.drop(columns=['site_norm','cluster1'],inplace=True)
        df_all['Site_clean']=df_all['Site_clean'].replace({"72 72":"72 jermyn"})
        df_all.loc[df_all['Site'].str.contains(r"(?i)\b68\b.*jermyn"),'Site_clean']="68 jermyn"
        df_all.loc[df_all['Site_clean']=="72 st",'Site_clean']="72 st jermyn"

        # 3) feature eng.
        df_all['Answered by']=df_all['Answered by'].str.strip().str.title()
        df_all['Date Completed']=pd.to_datetime(df_all['Date Completed'],errors='coerce')
        df_all['Month']=df_all['Date Completed'].dt.month.map(
            lambda m: calendar.month_abbr[int(m)] if pd.notnull(m) else None)
        df_all['Valid Questions']=df_all['Yes']+df_all['No']
        df_all['Calculated Score']=np.where(
            df_all['Valid Questions']>0,
            df_all['Yes']/df_all['Valid Questions']*100,
            np.nan)
        score_nums=df_all['Score'].str.split('/',expand=True).astype(float)
        df_all['Score_num'],df_all['Score_den']=score_nums[0],score_nums[1]
        q1,q3=df_all['Percentage Received'].quantile([0.25,0.75])
        lb=q1-1.5*(q3-q1)
        df_all['is_outlier_low']=df_all['Percentage Received']<lb
        df_all['Evaluation']=df_all.apply(
            lambda r:'Not Enough Data' if r['Valid Questions']<=5
                      else classify_score(r['Calculated Score']),
            axis=1)

        # --- TAB 1: Scores & Heatmap ---
        with tabs[1]:
            st.subheader("📊 Average Score by Site")
            st.markdown("This visual shows the average audit score by site, allowing comparison and performance tracking.")
            st.toast("Filters applied successfully.", icon="✅")
            
            # Seletor de modo de visualização
            mode = st.radio("View mode", ["Cumulative", "Monthly"], horizontal=True)
        
            if mode == "Monthly":
                meses = sorted(
                    df_filtered['Month'].dropna().unique(),
                    key=lambda m: list(calendar.month_abbr).index(m)
                )
                sel_mes = st.selectbox("Select Month", meses)
        
            st.sidebar.header("Filters")
            with st.sidebar.expander("📅 Date Range"):
                dmin, dmax = df_all['Date Completed'].min(), df_all['Date Completed'].max()
                sel_date = st.date_input("Select Date Range", [dmin, dmax], min_value=dmin, max_value=dmax)
            
            # --- Filtro de Sites ---
            with st.sidebar.expander("Sites"):
                site_options = sorted(df_all['Site_clean'].dropna().unique())
            
                if "selected_sites" not in st.session_state:
                    st.session_state.selected_sites = site_options
            
                # Botão de reset deve vir antes do multiselect para que o valor atualizado apareça imediatamente
                if st.button("🔁 Reset Site Filter"):
                    st.session_state.selected_sites = site_options
                    st.rerun()
            
                sel_sites = st.multiselect(
                    "Select Sites",
                    options=site_options,
                    default=st.session_state.selected_sites,
                    key="site_filter"
                )
            
            # --- Filtro de Evaluation ---
            with st.sidebar.expander("Evaluation"):
                sel_evals = st.multiselect("Select Evaluation", sorted(df_all['Evaluation'].unique()),
                                           default=sorted(df_all['Evaluation'].unique()))
            
            # --- Filtro de Auditors ---
            with st.sidebar.expander("Auditors"):
                sel_users = st.multiselect("Answered by", sorted(df_all['Answered by'].unique()),
                                           default=sorted(df_all['Answered by'].unique()))
        
            df_plot = df_all.copy()
            if mode == "Monthly":
                df_plot = df_plot[df_plot['Month'] == sel_mes]
            df_plot = df_plot[
                (df_plot['Date Completed'] >= pd.to_datetime(sel_date[0])) &
                (df_plot['Date Completed'] <= pd.to_datetime(sel_date[1])) &
                df_plot['Site_clean'].isin(sel_sites) &
                df_plot['Evaluation'].isin(sel_evals) &
                df_plot['Answered by'].isin(sel_users)
            ]
        
            c1, c2, c3 = st.columns(3)
            if 'view' not in st.session_state:
                st.session_state.view = 'all'
            if c1.button("Top 10"):
                st.session_state.view = 'top'
            if c2.button("Bottom 10"):
                st.session_state.view = 'bottom'
            if c3.button("All Sites"):
                st.session_state.view = 'all'
            view = st.session_state.view
        
            max_n = df_plot['Site_clean'].nunique()
            n = st.slider("Number of Sites", min_value=5, max_value=max_n, value=10)
        
            avg = df_plot.groupby('Site_clean')['Calculated Score'].mean()
            if view == 'top':
                sel = avg.sort_values(ascending=False).head(n)
            elif view == 'bottom':
                sel = avg.sort_values(ascending=True).head(n)
            else:
                sel = avg.sort_values(ascending=True)
        
            df_bar = sel.reset_index().rename(columns={'Calculated Score': 'Score'})
            totv = df_plot.groupby('Site_clean')['Valid Questions'].sum()
            df_bar['Total_Valid_Questions'] = df_bar['Site_clean'].map(totv)
            df_bar['Evaluation'] = df_bar.apply(
                lambda r: 'Not Enough Data' if r['Total_Valid_Questions'] <= 5 else classify_score(r['Score']), axis=1)
        
            g_app = df_bar[df_bar['Evaluation'] == 'Approved'].sort_values('Score', ascending=False)
            g_acc = df_bar[df_bar['Evaluation'] == 'Acceptable'].sort_values('Score', ascending=False)
            g_cri = df_bar[df_bar['Evaluation'] == 'Critical'].sort_values('Score', ascending=False)
            g_ne = df_bar[df_bar['Evaluation'] == 'Not Enough Data']
            df_bar = pd.concat([g_app, g_acc, g_cri, g_ne], ignore_index=True)
        
            cmap = {'Approved': '#2ecc71', 'Acceptable': '#f1c40f',
                    'Critical': '#e74c3c', 'Not Enough Data': '#95a5a6'}
            site_order = df_bar['Site_clean'].tolist()
            eval_order = ['Approved', 'Acceptable', 'Critical', 'Not Enough Data']
        
            fig1 = px.bar(df_bar, x='Score', y='Site_clean', orientation='h',
                          color='Evaluation', color_discrete_map=cmap,
                          category_orders={'Site_clean': site_order, 'Evaluation': eval_order},
                          hover_data={'Score': ':.1f', 'Total_Valid_Questions': True})
            fig1.add_vline(x=80, line_dash='dash', line_color='green', annotation_text='Approved (80%)')
            fig1.add_vline(x=70, line_dash='dash', line_color='orange', annotation_text='Acceptable (70%)')
            fig1.add_vline(x=0, line_dash='dash', line_color='red', annotation_text='Critical (<70%)')
            fig1.update_layout(height=400 + n * 25, margin=dict(l=200, r=40, t=80, b=40))
            st.plotly_chart(fig1, use_container_width=True)
        
            st.subheader("📈 Site Evolution")
            st.markdown("Track how the average score for each site evolves over time.")
        
            sel_site = st.selectbox("Select Site", sorted(df_all['Site_clean'].unique()))
            evo_mode = st.radio("Evolution Type", ["Monthly", "Cumulative"], horizontal=True)
            df_evo = df_all[df_all['Site_clean'] == sel_site]
            months = sorted(df_all['Month'].dropna().unique(),
                            key=lambda m: list(calendar.month_abbr).index(m))
            evo_df = df_evo.groupby('Month')['Calculated Score'].mean().reindex(months).reset_index()
            ycol = 'Cumulative' if evo_mode == "Cumulative" else 'Calculated Score'
            if evo_mode == "Cumulative":
                evo_df['Cumulative'] = evo_df['Calculated Score'].expanding().mean()
            fig_evo = px.line(evo_df, x='Month', y=ycol, markers=True, labels={ycol: 'Score (%)'})
            fig_evo.update_layout(yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_evo, use_container_width=True)
        
            st.subheader("🔥 Heatmap")
            st.markdown("This heatmap highlights the performance of the top and bottom 10 sites across months.")
        
            overall = df_plot.groupby('Site_clean')['Calculated Score'].mean().sort_values()
            t10, b10 = overall.tail(10).index.tolist(), overall.head(10).index.tolist()
            order = t10 + b10
            pivot = (df_plot[df_plot['Site_clean'].isin(order)]
                     .pivot_table('Calculated Score', 'Site_clean', 'Month', aggfunc='mean')
                     .reindex(columns=sorted(df_plot['Month'].dropna().unique()), index=order))
            fig2 = px.imshow(pivot, text_auto='.1f', aspect='auto',
                            color_continuous_scale='RdYlGn', range_color=[0, 100],
                            labels={'x': 'Month', 'y': 'Site', 'color': 'Score (%)'})
            fig2.update_layout(title="Heatmap", height=max(400, len(order) * 25))
            st.plotly_chart(fig2, use_container_width=True)
        
            st.subheader("🚨 Sites Not Audited in Selected Period")
            st.markdown("List of buildings that were not audited within the selected date range.")
        
            sites_total = set(df_all['Site_clean'].unique())
            sites_auditados = set(df_plot['Site_clean'].unique())
            sites_nao_auditados = sorted(sites_total - sites_auditados)
            st.write(f"🔍 {len(sites_nao_auditados)} site(s) not audited in the selected period.")
        
            if sites_nao_auditados:
                df_missing = pd.DataFrame({'Site': sites_nao_auditados})
                df_missing['Audit Status'] = "Not Audited"
                fig_missing = px.bar(
                    df_missing,
                    y='Site',
                    x=[1] * len(df_missing),
                    orientation='h',
                    labels={'x': '', 'Site': 'Site'},
                    title='Sites Not Audited',
                    text='Audit Status')
                fig_missing.update_traces(marker_color='red', textposition='outside')
                fig_missing.update_layout(
                    height=300 + 20 * len(sites_nao_auditados),
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    margin=dict(l=150, r=40, t=60, b=40))
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.info("✅ All selected sites have been audited in the current period.")


        # --- TAB 2: Auditor Overview ---
        with tabs[2]:
            st.subheader("🧑‍💼 Auditor Overview")
            st.markdown("Explore the activity and performance of each auditor based on completed audits, scores, and site coverage.")
        
            st.toast("Filters applied successfully.", icon="✅")
        
            # Filtros laterais
            with st.sidebar.expander("🧑 Select Auditor"):
                selected_auditor = st.selectbox("Answered by", sorted(df_all['Answered by'].dropna().unique()), key="auditor_user")
        
            with st.sidebar.expander("📅 Month Filter"):
                months = ["All"] + sorted(
                    df_all['Month'].dropna().unique(),
                    key=lambda m: list(calendar.month_abbr).index(m)
                )
                selected_month = st.selectbox("Select Month", months, key="auditor_month")
        
            # Aplica filtros
            df_aud = df_all.copy()
            if selected_month != "All":
                df_aud = df_aud[df_aud['Month'] == selected_month]
            df_aud = df_aud[df_aud['Answered by'] == selected_auditor]
        
            # KPI
            total_audits = len(df_aud)
            st.markdown(f"### ✅ Total Audits Performed: {total_audits}")
        
            # Tabela de frequência
            st.markdown("#### 📋 Audit Count per Site")
            if selected_month == "All":
                freq = df_aud.groupby(['Site_clean', 'Month']).size().reset_index(name='Count')
            else:
                freq = df_aud['Site_clean'].value_counts().rename_axis('Site_clean').reset_index(name='Count')
            st.dataframe(freq, use_container_width=True)
        
            # Gráfico de barras horizontais
            st.markdown("#### 📊 Audit Frequency by Site")
            df_counts = df_aud['Site_clean'].value_counts().reset_index()
            df_counts.columns = ['Site', 'Audit Count']
            df_counts = df_counts.sort_values('Audit Count', ascending=True)
        
            fig = px.bar(
                df_counts,
                x='Audit Count',
                y='Site',
                orientation='h',
                text='Audit Count',
                labels={'Audit Count': 'Audit Count', 'Site': 'Site'},
                title=f"Number of Audits by Site – Auditor: {selected_auditor}"
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(
                height=max(300, len(df_counts) * 25),
                margin=dict(l=100, r=40, t=60, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
        

        # --- TAB 3: Executive Dashboard ---
        with tabs[3]:
            st.subheader("📈 Executive Dashboard")
            st.markdown("This section provides a high-level summary of audit activity and performance indicators across all sites.")
        
            st.toast("Executive metrics loaded successfully.", icon="✅")
        
            # Filter dataset
            df_filtered = df_all[
                (df_all['Date Completed'] >= pd.to_datetime(sel_date[0])) &
                (df_all['Date Completed'] <= pd.to_datetime(sel_date[1])) &
                df_all['Site_clean'].isin(sel_sites) &
                df_all['Evaluation'].isin(sel_evals) &
                df_all['Answered by'].isin(sel_users)
            ]
        
            # KPIs
            k1, k2, k3 = st.columns(3)
            total_audits = len(df_filtered)
            total_sites = df_filtered['Site_clean'].nunique()
            eval_dist = df_filtered['Evaluation'].value_counts().to_dict()
            approved = eval_dist.get("Approved", 0)
            percentage_approved = (approved / total_audits) * 100 if total_audits > 0 else 0
        
            with k1:
                st.metric("Total Audits", f"{total_audits}")
            with k2:
                st.metric("Unique Sites Audited", f"{total_sites}")
            with k3:
                st.metric("Approved Audits (%)", f"{percentage_approved:.1f}%")
        
            st.markdown(f"**Selected Period:** {sel_date[0].strftime('%d %b %Y')} to {sel_date[1].strftime('%d %b %Y')}")
        
            # --- Summary ---
            st.markdown("### 🗒️ Audit Summary Overview")
            try:
                audited_sites = df_filtered['Site_clean'].nunique()
                all_sites = df_all['Site_clean'].nunique()
                not_audited = all_sites - audited_sites
                start_str = sel_date[0].strftime("%A, %d %B %Y")
                end_str = sel_date[1].strftime("%A, %d %B %Y")
        
                st.markdown(f"""
                - **Selected Period:** {start_str} to {end_str}  
                - **Total audits in selected period:** `{total_audits}`  
                - **Total distinct buildings audited:** `{audited_sites}`  
                - **Buildings NOT audited in this period:** `{not_audited}`  
                """)
            except:
                st.warning("⚠️ Could not generate audit summary.")
        
            # --- Evaluation Distribution ---
            st.markdown("### 📊 Evaluation Distribution")
            st.markdown("Count of audits per evaluation category (Approved, Acceptable, Critical).")
        
            eval_df = (
                df_filtered['Evaluation']
                .value_counts()
                .rename_axis('Evaluation')
                .reset_index(name='Count')
            )
        
            fig_eval = px.bar(
                eval_df,
                x='Count',
                y='Evaluation',
                orientation='h',
                color='Evaluation',
                text='Count',
                color_discrete_map={
                    'Approved': '#2ecc71',
                    'Acceptable': '#f1c40f',
                    'Critical': '#e74c3c',
                    'Not Enough Data': '#95a5a6'
                },
                title="Evaluation Summary"
            )
            fig_eval.update_layout(height=300)
            fig_eval.update_traces(textposition='outside')
            st.plotly_chart(fig_eval, use_container_width=True)
        
            # --- Top 5 Most Audited Sites ---
            st.markdown("### 🏢 Top 5 Most Audited Sites")
            st.markdown("Bar chart of the five buildings with the most audits during the selected period.")
        
            top5_sites = (
                df_filtered['Site_clean']
                .value_counts()
                .head(5)
                .reset_index(name='Audit Count')
                .rename(columns={'index': 'Site_clean'})
            )
        
            fig_top5 = px.bar(
                top5_sites,
                x='Audit Count',
                y='Site_clean',
                orientation='h',
                text='Audit Count',
                title='Top 5 Most Audited Sites',
                labels={'Audit Count': 'Number of Audits', 'Site_clean': 'Site'}
            )
            fig_top5.update_layout(
                yaxis=dict(categoryorder='total ascending'),
                height=400,
                margin=dict(l=80, r=40, t=60, b=40)
            )
            fig_top5.update_traces(marker_color='#3498db', textposition='outside')
            st.plotly_chart(fig_top5, use_container_width=True)
        
            # --- Monthly Evaluation Distribution ---
            st.markdown("### 📅 Monthly Evaluation Distribution")
            st.markdown("Stacked bar chart showing percentage of each evaluation by month.")
        
            valid_eval = ['Approved', 'Acceptable', 'Critical']
            df_all['Month'] = pd.to_datetime(df_all['Date Completed']).dt.strftime('%b')
            df_eval = df_all[df_all['Evaluation'].isin(valid_eval)].copy()
        
            monthly_eval = (
                df_eval.groupby(['Month', 'Evaluation'])
                .size()
                .reset_index(name='Count')
            )
        
            total_by_month = monthly_eval.groupby('Month')['Count'].sum().reset_index(name='Total')
            monthly_eval = monthly_eval.merge(total_by_month, on='Month')
            monthly_eval['Percentage'] = 100 * monthly_eval['Count'] / monthly_eval['Total']
        
            month_order = [m for m in calendar.month_abbr[1:] if m in monthly_eval['Month'].unique()]
            monthly_eval['Month'] = pd.Categorical(monthly_eval['Month'], categories=month_order, ordered=True)
            monthly_eval = monthly_eval.sort_values('Month')
        
            fig_monthly = px.bar(
                monthly_eval,
                x='Percentage',
                y='Month',
                color='Evaluation',
                orientation='h',
                text=monthly_eval['Percentage'].map(lambda x: f'{x:.1f}%'),
                color_discrete_map={
                    'Approved': '#2ecc71',
                    'Acceptable': '#f1c40f',
                    'Critical': '#e74c3c'
                },
                category_orders={'Month': month_order, 'Evaluation': valid_eval[::-1]},
                title='Monthly Distribution of Building Evaluations'
            )
            fig_monthly.update_layout(
                xaxis_title='% of Buildings',
                yaxis_title='Month',
                barmode='stack',
                legend_title='Evaluation',
                height=400,
                margin=dict(l=80, r=40, t=60, b=40)
            )
            fig_monthly.update_traces(
                textposition='inside',
                hovertemplate='<b>%{y}</b><br>%{color}: %{x:.1f}%',
                marker_line_width=0
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
        
            # --- Building Performance Over Time ---
            st.markdown("### 🏢 Building Performance Over Time")
            st.markdown("Scatter plot of average audit scores per site over the months.")
        
            try:
                df_valid = df_filtered[df_filtered['Valid Questions'] > 5].copy()
                df_valid['Month'] = pd.to_datetime(df_valid['Date Completed'], errors='coerce').dt.strftime('%b')
                df_valid = df_valid[df_valid['Month'].notna()]
        
                df_scatter = (
                    df_valid.groupby(['Site_clean', 'Month'])['Calculated Score']
                    .mean()
                    .reset_index()
                )
        
                month_order = list(calendar.month_abbr[1:])
                df_scatter['Month'] = pd.Categorical(df_scatter['Month'], categories=month_order, ordered=True)
                df_scatter = df_scatter.sort_values(['Month', 'Site_clean'])
        
                fig_scatter = px.scatter(
                    df_scatter,
                    x='Month',
                    y='Calculated Score',
                    color='Site_clean',
                    hover_data={
                        'Site_clean': True,
                        'Month': True,
                        'Calculated Score': ':.1f'
                    },
                    title='Building Performance Over Time',
                    labels={'Calculated Score': 'Average Score'},
                    height=500,
                    category_orders={'Month': month_order}
                )
        
                fig_scatter.add_hline(
                    y=80, line_dash='dash', line_color='green',
                    annotation_text='Approved (80%)', annotation_position='top left'
                )
                fig_scatter.add_hline(
                    y=70, line_dash='dash', line_color='orange',
                    annotation_text='Acceptable (70%)', annotation_position='bottom left'
                )
        
                fig_scatter.update_layout(
                    xaxis_title='Month',
                    yaxis_title='Calculated Score',
                    yaxis_range=[0, 100],
                    legend_title='Site',
                    margin=dict(l=60, r=40, t=60, b=40)
                )
        
                st.plotly_chart(fig_scatter, use_container_width=True)
        
            except Exception as e:
                st.error("⚠️ Failed to generate the scatter plot.")
                st.text(traceback.format_exc())

            

        # --- TAB 4: Monthly Highlights ---
        with tabs[4]:
            st.subheader("📌 Monthly Highlights")
            st.markdown("This section showcases the top and bottom performing sites for each month based on audit scores.")
            st.toast("Monthly highlights loaded successfully.", icon="✅")
        
            # Calculate monthly scores
            df_scores = df_all.groupby(['Month', 'Site_clean'])['Calculated Score'].mean().reset_index()
            month_order = sorted(df_scores['Month'].dropna().unique(), key=lambda m: list(calendar.month_abbr).index(m))
        
            # Display monthly highlights
            for month in month_order:
                st.markdown(f"### 📅 {month}")
                st.markdown(f"Top and bottom 10 sites based on average scores for {month}.")
        
                month_df = df_scores[df_scores['Month'] == month]
        
                try:
                    top10 = month_df.sort_values('Calculated Score', ascending=False).head(10)
                    bottom10 = month_df.sort_values('Calculated Score', ascending=True).head(10)
        
                    col1, col2 = st.columns(2)
        
                    with col1:
                        fig_top = px.scatter(
                            top10,
                            x='Calculated Score',
                            y='Site_clean',
                            text=top10['Calculated Score'].map(lambda x: f'{x:.1f}%'),
                            color_discrete_sequence=['#2ecc71'],
                            title=f"Top 10 Sites – {month}",
                            labels={'Site_clean': 'Site', 'Calculated Score': 'Score (%)'}
                        )
                        fig_top.update_traces(marker=dict(size=12), textposition='middle right')
                        fig_top.update_layout(
                            xaxis_range=[60, 104],
                            height=400,
                            margin=dict(l=80, r=40, t=50, b=40)
                        )
                        st.plotly_chart(fig_top, use_container_width=True)
        
                    with col2:
                        fig_bot = px.scatter(
                            bottom10,
                            x='Calculated Score',
                            y='Site_clean',
                            text=bottom10['Calculated Score'].map(lambda x: f'{x:.1f}%'),
                            color_discrete_sequence=['#e74c3c'],
                            title=f"Bottom 10 Sites – {month}",
                            labels={'Site_clean': 'Site', 'Calculated Score': 'Score (%)'}
                        )
                        fig_bot.update_traces(marker=dict(size=12), textposition='middle right')
                        fig_bot.update_layout(
                            xaxis_range=[60, 102],
                            height=400,
                            margin=dict(l=80, r=40, t=50, b=40)
                        )
                        st.plotly_chart(fig_bot, use_container_width=True)
        
                except Exception as e:
                    st.error(f"❌ Error generating highlights for {month}.")
                    st.text(traceback.format_exc())
        
            # --- Export Filtered Data ---
            st.markdown("### 📥 Export Filtered Dataset")
            st.markdown("Download the filtered audit data based on selected filters.")
            st.download_button(
                label="⬇️ Download Filtered Data as CSV",
                data=df_plot.to_csv(index=False),
                file_name="filtered_audits.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error("❌ Error processing the uploaded file.")
        st.text(traceback.format_exc())
