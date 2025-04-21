import streamlit as st
import pandas as pd
import numpy as np
import calendar
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cleaning Audit Dashboard", layout="wide")
st.title("🧼 Cleaning Audit Dashboard")

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload the Excel file with multiple monthly sheets", type=["xlsx"])

if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    @st.cache_data
    def load_clean_sheet(file, sheet_name):
        df = pd.read_excel(file, sheet_name=sheet_name)
        df.columns = df.columns.str.strip().str.replace(' +', ' ', regex=True)
        df.columns = df.columns.str.replace('Questionarie Result', 'Questionnaire Result', regex=False)

        expected_cols = [
            'Date Completed', 'Site', 'Answered by',
            'Percentage Received', 'Score', 'Questionnaire Result',
            'Yes', 'No', 'N/A'
        ]
        if not set(expected_cols).issubset(df.columns):
            raise ValueError(f"Missing columns in sheet '{sheet_name}'")
        return df

    all_data = [load_clean_sheet(uploaded_file, sheet) for sheet in sheet_names]
    df = pd.concat(all_data, ignore_index=True)

    # --- DATA CLEANING ---
    df['Answered by'] = df['Answered by'].str.strip().str.title()

    # --- DATA ENRICHMENT ---
    df['Date Completed'] = pd.to_datetime(df['Date Completed'], errors='coerce')
    df['Month'] = df['Date Completed'].dt.month.apply(lambda x: calendar.month_abbr[x] if pd.notnull(x) else None)
    df['Valid Questions'] = df['Yes'] + df['No']
    df['Calculated Score'] = np.where(df['Valid Questions'] > 0, (df['Yes'] / df['Valid Questions']) * 100, np.nan)

    def classify_score(score):
        if pd.isnull(score): return 'Not Applicable'
        elif score >= 80: return 'Approved'
        elif score >= 60: return 'Acceptable'
        else: return 'Critical'

    df['Evaluation'] = df['Calculated Score'].apply(classify_score)
    df['Evaluation Adjusted'] = df.apply(
        lambda row: 'Not Enough Data' if row['Valid Questions'] <= 5 else classify_score(row['Calculated Score']),
        axis=1
    )

    # --- FILTERS ---
    st.sidebar.header("🔍 Filters")
    months = sorted(df['Month'].dropna().unique())
    sites = sorted(df['Site'].dropna().unique())
    evaluations = df['Evaluation Adjusted'].dropna().unique().tolist()
    answered_by = sorted(df['Answered by'].dropna().unique())

    selected_month = st.sidebar.selectbox("Select Month", options=['All'] + months)
    selected_sites = st.sidebar.multiselect("Select Sites", options=sites, default=sites)
    selected_eval = st.sidebar.multiselect("Select Evaluation", options=evaluations, default=evaluations)
    selected_users = st.sidebar.multiselect("Select Answered By", options=answered_by, default=answered_by)

    date_min = df['Date Completed'].min()
    date_max = df['Date Completed'].max()
    selected_date_range = st.sidebar.date_input("Select Date Range", [date_min, date_max], min_value=date_min, max_value=date_max)

    filtered_df = df.copy()
    if selected_month != 'All':
        filtered_df = filtered_df[filtered_df['Month'] == selected_month]
    filtered_df = filtered_df[filtered_df['Site'].isin(selected_sites)]
    filtered_df = filtered_df[filtered_df['Evaluation Adjusted'].isin(selected_eval)]
    filtered_df = filtered_df[filtered_df['Answered by'].isin(selected_users)]
    filtered_df = filtered_df[
        (filtered_df['Date Completed'] >= pd.to_datetime(selected_date_range[0])) &
        (filtered_df['Date Completed'] <= pd.to_datetime(selected_date_range[1]))
    ]

    # --- ANALYSIS TYPE ---
    st.sidebar.header(":bar_chart: Analysis View")
    view_option = st.sidebar.radio("Select View", options=['Top 10 Sites', 'Bottom 10 Sites', 'All Sites'])

    # --- BAR PLOT WITH DYNAMIC COLOURS ---
    st.subheader("📊 Average Score by Site")
    avg_scores = filtered_df.groupby('Site')['Calculated Score'].mean().dropna()

    if view_option == 'Top 10 Sites':
        avg_scores = avg_scores.sort_values(ascending=False).head(10)
    elif view_option == 'Bottom 10 Sites':
        avg_scores = avg_scores.sort_values().head(10)
    else:
        avg_scores = avg_scores.sort_values()

    def score_to_color(score):
        if score >= 80:
            return '#2ca02c'  # green
        elif score >= 60:
            return '#ff7f0e'  # orange
        else:
            return '#d62728'  # red

    colors = avg_scores.apply(score_to_color).values

    fig1, ax1 = plt.subplots()
    sns.barplot(x=avg_scores.values, y=avg_scores.index, palette=colors, ax=ax1)
    ax1.axvline(80, color='green', linestyle='--', label='Approved (80%)')
    ax1.axvline(60, color='red', linestyle='--', label='Critical (60%)')
    ax1.set_xlabel("Average Score")
    ax1.set_title("Site Score Distribution")
    ax1.legend()
    st.pyplot(fig1)

    # --- HEATMAP ---
    st.subheader(":fire: Heatmap of Monthly Scores")
    top_bottom_sites = avg_scores.index.tolist()
    df_heatmap = filtered_df[filtered_df['Site'].isin(top_bottom_sites)]
    pivot_table = df_heatmap.pivot_table(values='Calculated Score', index='Site', columns='Month', aggfunc='mean')

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=.5, ax=ax2, vmin=0, vmax=100)
    ax2.set_title("Heatmap: Monthly Score per Site")
    st.pyplot(fig2)

    # --- SCATTER WITH REGRESSION ---
    st.subheader(":mag_right: % Received vs Calculated Score")
    x = filtered_df['Percentage Received']
    y = filtered_df['Calculated Score']

    if len(x.dropna()) > 1 and len(y.dropna()) > 1:
        slope, intercept, r_value, _, _ = linregress(x, y)
        reg_line = slope * x + intercept

        fig3, ax3 = plt.subplots()
        sns.scatterplot(x=x, y=y, ax=ax3, s=60, alpha=0.6)
        ax3.plot(x, reg_line, color='red', label=f'Trend (R²={r_value**2:.2f})')
        ax3.axhline(80, color='green', linestyle='--')
        ax3.axhline(60, color='orange', linestyle='--')
        ax3.set_xlabel('Percentage Received')
        ax3.set_ylabel('Calculated Score')
        ax3.set_title('Relationship Between % Received and Score')
        ax3.legend()
        st.pyplot(fig3)
    else:
        st.warning("Not enough data to calculate linear regression.")

# --- AUDITOR SITE COVERAGE WITH MONTH AND SITE FILTER + KPI ---
st.subheader(":busts_in_silhouette: Auditor Site Coverage")

# Select an auditor
selected_auditor = st.selectbox("Select an Auditor", options=answered_by)

# Filtered data for this auditor
auditor_df = filtered_df[filtered_df['Answered by'] == selected_auditor]

# Extra filters
months_available = sorted(auditor_df['Month'].dropna().unique().tolist())
sites_available = sorted(auditor_df['Site'].dropna().unique())

selected_month_filter = st.selectbox("Optional: Filter by Month", options=['All'] + months_available)
selected_site_filter = st.selectbox("Optional: Filter by Site", options=['All'] + sites_available)

# Apply extra filters
filtered_audit_df = auditor_df.copy()
if selected_month_filter != 'All':
    filtered_audit_df = filtered_audit_df[filtered_audit_df['Month'] == selected_month_filter]
if selected_site_filter != 'All':
    filtered_audit_df = filtered_audit_df[filtered_audit_df['Site'] == selected_site_filter]

# KPI - total audits after filters
total_audits_filtered = len(filtered_audit_df)
st.markdown(f"### ✅ Total Audits: `{total_audits_filtered}`")

# Audit frequency table
audit_period = (
    filtered_audit_df.groupby(['Site', 'Month'])
    .size()
    .reset_index(name='Audit Count')
    .sort_values(by=['Audit Count'], ascending=False)
)

st.markdown(f"**Audit frequency by Site and Month – `{selected_auditor}`**")
st.dataframe(audit_period)

# Bar chart: audit counts by site
site_counts = filtered_audit_df['Site'].value_counts()

fig4, ax4 = plt.subplots(figsize=(10, max(5, len(site_counts) * 0.3)))
sns.barplot(y=site_counts.index, x=site_counts.values, palette="Blues_d", ax=ax4)
ax4.set_xlabel("Number of Audits")
ax4.set_ylabel("Site")
ax4.set_title(f"Sites audited by {selected_auditor}")
plt.xticks(rotation=0)
plt.tight_layout()
st.pyplot(fig4)


# --- DOWNLOAD CLEANED FILE ---
st.download_button(
    label="Download Processed CSV",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_cleaned_audit.csv",
    mime="text/csv"
)
