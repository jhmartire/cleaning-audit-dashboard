import streamlit as st
import pandas as pd
import numpy as np
import calendar
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import base64
import re
import traceback

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cleaning Audit Dashboard", layout="wide")

# --- TABS ---
tabs = st.tabs(["📘 Instructions & Upload", "📊 Scores & Heatmap", "🧑‍💼 Auditor Overview"])

# --- TAB 0: INSTRUÇÕES & UPLOAD ---
with tabs[0]:
    st.title("🧼 Cleaning Audit Dashboard")
    st.markdown("### Upload the Excel file with multiple monthly sheets")
    uploaded_file = st.file_uploader("Upload your file here", type=["xlsx"])
    with st.expander("📌 How to use", expanded=False):
        st.markdown("""
        1. Cada aba deve ter colunas:  
           `Date Completed, Site, Answered by, Percentage Received, Score, Questionnaire Result, Yes, No, N/A`  
        2. Evite abas de resumo/extras.  
        3. Baixe o template, se quiser:
        """)
        try:
            st.image("img/example_sheet.png", use_container_width=True)
        except:
            pass
        try:
            with open("audit_template.xlsx", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<a href="data:application/octet-stream;base64,{b64}" '
                    'download="audit_template.xlsx">📥 Download Template</a>',
                    unsafe_allow_html=True
                )
        except:
            pass

if uploaded_file:
    sheets = pd.ExcelFile(uploaded_file).sheet_names

    @st.cache_data
    def load_clean_sheet(file, sh):
        df = pd.read_excel(file, sheet_name=sh)
        df.columns = (df.columns
                        .str.strip()
                        .str.replace(r'\s+', ' ', regex=True)
                        .str.replace('Questionarie Result','Questionnaire Result'))
        need = ['Date Completed','Site','Answered by','Percentage Received',
                'Score','Questionnaire Result','Yes','No','N/A']
        if not set(need).issubset(df.columns):
            raise ValueError(f"Faltando colunas em '{sh}'")
        return df

    try:
        df = pd.concat([load_clean_sheet(uploaded_file,sh) for sh in sheets], ignore_index=True)

        # --- 1) NORMALIZAÇÃO DE SITE + FUZZY GROUPING ---
        def normalize_name(name: str) -> str:
            s = name.lower()
            # ranges "xx-yy" -> maior dos dois
            s = re.sub(r'(\d+)\s*-\s*(\d+)',
                       lambda m: str(max(int(m.group(1)),int(m.group(2)))), s)
            # street -> st
            s = re.sub(r'\bstreet\b','st', s)
            # “32st” ou “32 st” -> “32”
            s = re.sub(r'(\d+)\s*st\b', r'\1', s)
            # remove tokens de pouco valor
            s = re.sub(r'\b(s|house|place|floor)\b','', s)
            # deixa só alfanum e espaço
            s = re.sub(r'[^a-z0-9\s]',' ', s)
            return re.sub(r'\s+',' ', s).strip()

        df['Site_norm'] = df['Site'].astype(str).apply(normalize_name)
        df['token_key'] = df['Site_norm'].apply(lambda s: ' '.join(sorted(s.split())))

        canon_map = {}
        for key, grp in df.groupby('token_key'):
            canon = grp['Site_norm'].mode()[0]
            for v in grp['Site_norm'].unique():
                canon_map[v] = canon

        df['Site'] = df['Site_norm'].map(canon_map)
        df.drop(columns=['Site_norm','token_key'], inplace=True)

        # 1) garantir “st” após o número
        def ensure_st_suffix(canonical: str) -> str:
            parts = canonical.split()
            if parts and parts[0].isdigit() and (len(parts)>1 and parts[1] != 'st'):
                return f"{parts[0]} st {' '.join(parts[1:])}"
            return canonical

        df['Site'] = df['Site'].apply(ensure_st_suffix)

        # 2) remover um “st” sobrando no final
        df['Site'] = df['Site'] \
            .str.replace(r'\bst$', '', regex=True) \
            .str.strip()

        # --- 3) CÁLCULOS PRINCIPAIS ---
        df['Answered by'] = df['Answered by'].str.strip().str.title()
        df['Date Completed'] = pd.to_datetime(df['Date Completed'], errors='coerce')
        df['Month'] = df['Date Completed'].dt.month.apply(
            lambda x: calendar.month_abbr[int(x)] if pd.notnull(x) else None
        )
        df['Valid Questions'] = df['Yes'] + df['No']
        df['Calculated Score'] = np.where(
            df['Valid Questions']>0,
            (df['Yes']/df['Valid Questions'])*100,
            np.nan
        )

        # --- 4) CLASSIFICAÇÃO “Not Enough Data” ---
        def classify_score(s):
            if pd.isnull(s):   return 'Not Applicable'
            if s>=80:          return 'Approved'
            if s>=60:          return 'Acceptable'
            return 'Critical'

        df['Evaluation'] = df.apply(
            lambda r: 'Not Enough Data'
                      if r['Valid Questions']<=5
                      else classify_score(r['Calculated Score']),
            axis=1
        )

        # --- 5) SIDEBAR FILTROS ---
        st.sidebar.header(":bar_chart: Analysis View")
        view = st.sidebar.radio("Select View",
               ['Top 10 Sites','Bottom 10 Sites','All Sites'])
        st.sidebar.markdown("---")
        st.sidebar.header("🔍 Filters")

        months = ['All'] + sorted(df['Month'].dropna().unique().tolist())
        sites  = sorted(df['Site'].dropna().unique())
        evals  = sorted(df['Evaluation'].dropna().unique())
        users  = sorted(df['Answered by'].dropna().unique())
        dmin, dmax = df['Date Completed'].min(), df['Date Completed'].max()

        sel_month = st.sidebar.selectbox("Month", months)
        sel_range = st.sidebar.date_input("Date Range",
                       [dmin,dmax], min_value=dmin, max_value=dmax)
        sel_sites = st.sidebar.multiselect("Sites", sites, default=sites)
        sel_evals = st.sidebar.multiselect("Evaluation", evals, default=evals)
        sel_users = st.sidebar.multiselect("Answered by", users, default=users)

        filt = df.copy()
        if sel_month!='All': 
            filt = filt[filt['Month']==sel_month]
        filt = filt[filt['Site'].isin(sel_sites)]
        filt = filt[filt['Evaluation'].isin(sel_evals)]
        filt = filt[filt['Answered by'].isin(sel_users)]
        filt = filt[
            (filt['Date Completed']>=pd.to_datetime(sel_range[0])) &
            (filt['Date Completed']<=pd.to_datetime(sel_range[1]))
        ]

        # --- 6) TAB Scores & Heatmap ---
        with tabs[1]:
            st.subheader("📊 Average Score by Site")
            avg = filt.groupby('Site')['Calculated Score'].mean().dropna()
            if view=='Top 10 Sites':
                avg = avg.sort_values(ascending=False).head(10)
            elif view=='Bottom 10 Sites':
                avg = avg.sort_values().head(10)
            else:
                avg = avg.sort_values()

            df_bar = avg.reset_index().rename(columns={'Calculated Score':'Score'})
            emap   = filt.set_index('Site')['Evaluation'].to_dict()
            df_bar['Evaluation'] = df_bar['Site'].map(emap)
            cmap   = {
                'Approved':'#2ecc71','Acceptable':'#f1c40f',
                'Critical':'#e74c3c','Not Enough Data':'#95a5a6'
            }
            df_bar['Color'] = df_bar['Evaluation'].map(cmap)

            fig1,ax1 = plt.subplots()
            sns.barplot(
                x=df_bar['Score'], y=df_bar['Site'],
                palette=df_bar['Color'].tolist(), ax=ax1
            )
            ax1.axvline(80,color='green',linestyle='--',label='Approved (>=80%)')
            ax1.axvline(60,color='red',linestyle='--',  label='Critical (<60%)')
            ax1.set_xlabel("Average Score (%)")
            ax1.legend()
            st.pyplot(fig1)

            st.subheader("🔥 Heatmap of Monthly Scores")
            overall = filt.groupby('Site')['Calculated Score'].mean().sort_values()
            top10 = list(overall.tail(10).index)
            bot10 = list(overall.head(10).index)
            order = top10 + bot10

            hm = filt[filt['Site'].isin(order)]
            pivot = hm.pivot_table(
                'Calculated Score','Site','Month',aggfunc='mean'
            )
            months_ord = ['Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec']
            pivot = pivot.reindex(columns=[m for m in months_ord if m in pivot.columns],
                                  index=order)

            fig2,ax2 = plt.subplots(figsize=(10,6))
            sns.heatmap(
                pivot,annot=True,fmt='.1f',cmap='RdYlGn',
                vmin=0,vmax=100,linewidths=0.5,
                cbar_kws={'label':'Score (%)'}, ax=ax2
            )
            ax2.set_title("Heatmap: Monthly Score per Site")
            st.pyplot(fig2)

            st.subheader("🔍 % Received vs Calculated Score")
            sc = filt[filt['Calculated Score']<=100]
            x,y = sc['Percentage Received'], sc['Calculated Score']
            if x.notna().sum()>1 and y.notna().sum()>1:
                s,i,r,_,_ = linregress(x,y)
                rl = s*x + i
                fig3,ax3 = plt.subplots()
                sns.scatterplot(x=x,y=y,alpha=.6,s=60,ax=ax3)
                ax3.plot(x,rl,color='red',label=f'R²={r**2:.2f}')
                ax3.axhline(80,color='green',linestyle='--')
                ax3.axhline(60,color='orange',linestyle='--')
                ax3.legend()
                st.pyplot(fig3)
            else:
                st.warning("Dados insuficientes para regressão")

        # --- 7) TAB Auditor Overview ---
        with tabs[2]:
            aud = st.selectbox("Answered by", users)
            adf = filt[filt['Answered by']==aud]
            st.markdown(f"### ✅ Total Audits: {len(adf)}")
            freq = (adf.groupby(['Site','Month'])
                      .size()
                      .reset_index(name='Count')
                      .sort_values('Count',ascending=False))
            st.dataframe(freq)
            cnts = adf['Site'].value_counts()
            fig4,ax4 = plt.subplots(figsize=(10,max(5,len(cnts)*.3)))
            sns.barplot(x=cnts.values,y=cnts.index,palette="Blues_d",ax=ax4)
            ax4.set_xlabel("Audit Count"); ax4.set_ylabel("Site")
            st.pyplot(fig4)

        # --- 8) DOWNLOAD CSV ---
        st.download_button(
            "📥 Download CSV",
            data=filt.to_csv(index=False),
            file_name="filtered.csv", mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.text(traceback.format_exc())

else:
    with tabs[1]:
        st.info("👆 Upload your Excel to begin.")