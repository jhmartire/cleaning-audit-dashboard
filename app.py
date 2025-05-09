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

# --- Configuração da página ---
st.set_page_config(page_title="Cleaning Audit Dashboard", layout="wide")

tabs = st.tabs([
    "📘 Instructions & Upload",
    "📊 Scores & Heatmap",
    "🧑‍💼 Auditor Overview"
])

# --- TAB 0: Upload ---
with tabs[0]:
    st.title("🧼 Cleaning Audit Dashboard")
    st.markdown("### Upload the Excel file (.xlsx) com múltiplas sheets mensais")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if not (username=="andron" and password=="andron25"):
        st.error("Invalid username or password")
        st.stop()
    uploaded_file = st.file_uploader("Upload seu arquivo aqui", type=["xlsx"])
    with st.expander("📌 How to use", expanded=True):
        st.markdown("""
        1. Authenticate com username/senha.  
        2. Cada sheet deve ter colunas:  
           Date Completed, Site, Answered by, Percentage Received, Score, Questionnaire Result, Yes, No, N/A.  
        3. Evite sheets extras.  
        4. Baixe template se precisar.
        """)
        try:
            with open("audit_template.xlsx","rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                st.markdown(
                  f'<a href="data:application/octet-stream;base64,{b64}" '
                  'download="audit_template.xlsx">📥 Download Template</a>',
                  unsafe_allow_html=True
                )
        except FileNotFoundError:
            st.info("Template não está no servidor.")

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
            mode=st.radio("View mode",["Cumulative","Monthly"],horizontal=True)
            if mode=="Monthly":
                meses=sorted(df_all['Month'].dropna().unique(),
                             key=lambda m:list(calendar.month_abbr).index(m))
                sel_mes=st.selectbox("Select Month",meses)

            st.sidebar.header("Filters")
            dmin,dmax=df_all['Date Completed'].min(),df_all['Date Completed'].max()
            sel_date=st.sidebar.date_input("Date Range",[dmin,dmax],min_value=dmin,max_value=dmax)
            with st.sidebar.expander("Sites"):
                sel_sites=st.multiselect("Select Sites",sorted(df_all['Site_clean'].unique()),
                                         default=sorted(df_all['Site_clean'].unique()))
            with st.sidebar.expander("Evaluation"):
                sel_evals=st.multiselect("Select Evaluation",sorted(df_all['Evaluation'].unique()),
                                         default=sorted(df_all['Evaluation'].unique()))
            with st.sidebar.expander("Auditors"):
                sel_users=st.multiselect("Answered by",sorted(df_all['Answered by'].unique()),
                                         default=sorted(df_all['Answered by'].unique()))

            df_plot=df_all.copy()
            if mode=="Monthly": df_plot=df_plot[df_plot['Month']==sel_mes]
            df_plot=df_plot[
                (df_plot['Date Completed']>=pd.to_datetime(sel_date[0])) &
                (df_plot['Date Completed']<=pd.to_datetime(sel_date[1])) &
                df_plot['Site_clean'].isin(sel_sites) &
                df_plot['Evaluation'].isin(sel_evals) &
                df_plot['Answered by'].isin(sel_users)
            ]

            c1,c2,c3=st.columns(3)
            if 'view' not in st.session_state: st.session_state.view='all'
            if c1.button("Top 10"):    st.session_state.view='top'
            if c2.button("Bottom 10"): st.session_state.view='bottom'
            if c3.button("All Sites"): st.session_state.view='all'
            view=st.session_state.view
            max_n=df_plot['Site_clean'].nunique()
            n=st.slider("Number of Sites",min_value=5,max_value=max_n,value=10)

            avg=df_plot.groupby('Site_clean')['Calculated Score'].mean()
            if view=='top':   sel=avg.sort_values(ascending=False).head(n)
            elif view=='bottom': sel=avg.sort_values(ascending=True).head(n)
            else:             sel=avg.sort_values(ascending=True)

            df_bar=sel.reset_index().rename(columns={'Calculated Score':'Score'})
            totv=df_plot.groupby('Site_clean')['Valid Questions'].sum()
            df_bar['Total_Valid_Questions']=df_bar['Site_clean'].map(totv)
            df_bar['Evaluation']=df_bar.apply(
                lambda r:'Not Enough Data' if r['Total_Valid_Questions']<=5
                          else classify_score(r['Score']),
                axis=1)

            # — separa por categoria e concatena na ordem desejada —
            g_app = df_bar[df_bar['Evaluation']=='Approved']    .sort_values('Score',ascending=False)
            g_acc = df_bar[df_bar['Evaluation']=='Acceptable']  .sort_values('Score',ascending=False)
            g_cri = df_bar[df_bar['Evaluation']=='Critical']    .sort_values('Score',ascending=False)
            g_ne  = df_bar[df_bar['Evaluation']=='Not Enough Data']
            df_bar = pd.concat([g_app,g_acc,g_cri,g_ne],ignore_index=True)

            # — fixa ordem de y=eixo e legenda —
            cmap={'Approved':'#2ecc71','Acceptable':'#f1c40f',
                  'Critical':'#e74c3c','Not Enough Data':'#95a5a6'}
            site_order=df_bar['Site_clean'].tolist()
            eval_order=['Approved','Acceptable','Critical','Not Enough Data']

            fig1=px.bar(df_bar,x='Score',y='Site_clean',orientation='h',
                        color='Evaluation',color_discrete_map=cmap,
                        category_orders={'Site_clean':site_order,
                                         'Evaluation':eval_order},
                        hover_data={'Score':':.1f','Total_Valid_Questions':True})
            fig1.add_vline(x=80,line_dash='dash',line_color='green',
                           annotation_text='Approved (80%)')
            fig1.add_vline(x=70,line_dash='dash',line_color='orange',
                           annotation_text='Acceptable (70%)')
            fig1.add_vline(x=0, line_dash='dash',line_color='red',
                           annotation_text='Critical (<70%)')
            fig1.update_layout(height=400+n*25,
                               margin=dict(l=200,r=40,t=80,b=40))
            st.plotly_chart(fig1,use_container_width=True)

            # — Site Evolution —
            st.subheader("📈 Site Evolution")
            sel_site=st.selectbox("Select Site",sorted(df_all['Site_clean'].unique()))
            evo_mode=st.radio("Evolution Type",["Monthly","Cumulative"],horizontal=True)
            df_evo=df_all[df_all['Site_clean']==sel_site]
            months=sorted(df_all['Month'].dropna().unique(),
                          key=lambda m:list(calendar.month_abbr).index(m))
            evo_df=df_evo.groupby('Month')['Calculated Score'].mean()\
                         .reindex(months).reset_index()
            if evo_mode=="Cumulative":
                evo_df['Cumulative']=evo_df['Calculated Score'].expanding().mean()
                ycol='Cumulative'
            else:
                ycol='Calculated Score'
            fig_evo=px.line(evo_df,x='Month',y=ycol,markers=True,
                            labels={ycol:'Score (%)'})
            fig_evo.update_layout(yaxis=dict(range=[0,100]))
            st.plotly_chart(fig_evo,use_container_width=True)

            # — Heatmap —
            overall=df_plot.groupby('Site_clean')['Calculated Score'].mean().sort_values()
            t10, b10 = overall.tail(10).index.tolist(), overall.head(10).index.tolist()
            order = t10 + b10
            pivot=(df_plot[df_plot['Site_clean'].isin(order)]
                   .pivot_table('Calculated Score','Site_clean','Month',aggfunc='mean')
                   .reindex(columns=sorted(df_plot['Month'].dropna().unique()), index=order))
            fig2=px.imshow(pivot,text_auto='.1f',aspect='auto',
                          color_continuous_scale='RdYlGn',range_color=[0,100],
                          labels={'x':'Month','y':'Site','color':'Score (%)'})
            fig2.update_layout(title="🔥 Heatmap",height=max(400,len(order)*25))
            st.plotly_chart(fig2,use_container_width=True)

            # — Scatter —
            sc=df_plot.dropna(subset=['Percentage Received','Calculated Score'])
            if len(sc)>1:
                fig3=px.scatter(sc,x='Percentage Received',y='Calculated Score',
                                hover_data=['Answered by'],labels={'x':'% Received','y':'Score'})
                slope,inter,r,*_=linregress(sc['Percentage Received'],sc['Calculated Score'])
                xln=np.linspace(0,100,100)
                fig3.add_trace(go.Scatter(x=xln,y=slope*xln+inter,mode='lines',
                                          line=dict(color='red',dash='dash'),
                                          name=f'Fit (R²={r**2:.2f})'))
                fig3.update_layout(title='% Received vs Score',
                                   yaxis=dict(range=[0,100]))
                st.plotly_chart(fig3,use_container_width=True)

        # --- TAB 2: Auditor Overview ---
        with tabs[2]:
            st.subheader("🧑‍💼 Auditor Overview")
            meses = ["All"] + sorted(df_all['Month'].dropna().unique(),
                                     key=lambda m:list(calendar.month_abbr).index(m))
            sel_m = st.selectbox("Select Month", meses)
            sel_u = st.selectbox("Answered by", sorted(df_all['Answered by'].unique()))
            df_aud = df_all.copy()
            if sel_m!="All": df_aud=df_aud[df_aud['Month']==sel_m]
            df_aud=df_aud[df_aud['Answered by']==sel_u]
            st.markdown(f"### ✅ Total Audits: {len(df_aud)}")
            if sel_m=="All":
                freq=df_aud.groupby(['Site_clean','Month']).size().reset_index(name='Count')
            else:
                freq=df_aud['Site_clean'].value_counts()\
                      .rename_axis('Site_clean').reset_index(name='Count')
            st.dataframe(freq)
            cnts=df_aud['Site_clean'].value_counts().sort_index()
            fig4=px.bar(x=cnts.values,y=cnts.index,orientation='h',
                        text=cnts.values,labels={'x':'Count','y':'Site'})
            fig4.update_traces(textposition='outside')
            fig4.update_layout(height=max(300,len(cnts)*25))
            st.plotly_chart(fig4,use_container_width=True)

        # --- Download CSV ---
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
    with tabs[1]:
        st.info("👆 Upload seu arquivo para começar.")
