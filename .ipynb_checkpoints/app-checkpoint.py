# Import libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import unicodedata
import re

# --- Streamlit page setup ---
st.set_page_config(page_title="Cleaning Audit Insights", layout="wide")
st.title("Data-Driven Cleaning Audit Insights")

# --- Introduction ---
st.header("Understanding Our Operational Performance Through Data")
st.markdown("""
This dashboard provides a clear and visual overview of our cleaning audit performance over the last three months. By analyzing this data, we can identify key trends, areas of strength, and opportunities for improvement, ultimately leading to cost savings and enhanced efficiency.
""")

# --- Sidebar: File uploader ---
st.sidebar.header("Upload Audit Data")
uploaded_file = st.sidebar.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

# --- Function: Standardize column names ---
def standardize_column_names(df):
    df.columns = [
        re.sub(r'\W+', '_',
               unicodedata.normalize('NFKD', col)
               .encode('ascii', 'ignore')
               .decode('utf-8')
               .strip()
               .lower()
        )
        for col in df.columns
    ]
    return df

# --- Function: Rename common variations ---
def rename_columns(df):
    col_map = {
        'date_completed': ['date_completed', 'completed_date', 'data_finalizada'],
        'score': ['score', 'pontuacao', 'nota'],
        'questionarie_result': ['questionarie_result', 'resultado_questionario', 'result'],
        'site': ['site', 'location', 'local']
    }
    for standard_name, aliases in col_map.items():
        for alias in aliases:
            if alias in df.columns:
                df.rename(columns={alias: standard_name}, inplace=True)
                break
    return df

# --- Function: Load and merge data ---
def load_data(file):
    try:
        if file.name.endswith('.xlsx'):
            sheets = {
                'Jan': pd.read_excel(file, sheet_name='January 25', header=1),
                'Feb': pd.read_excel(file, sheet_name='Feb 25', header=1),
                'Mar': pd.read_excel(file, sheet_name='March 25', header=1)
            }
            df = pd.concat(sheets.values(), keys=sheets.keys(), names=['month'])
            df.reset_index(inplace=True)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
            df['month'] = 'Uploaded'
        else:
            st.error("Unsupported file format.")
            return None

        df = standardize_column_names(df)
        df = rename_columns(df)

        if 'date_completed' in df.columns:
            df['date'] = pd.to_datetime(df['date_completed'], errors='coerce')
        if 'score' in df.columns:
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
        if 'questionarie_result' in df.columns:
            df['questionarie_result'] = df['questionarie_result'].astype(str).str.strip()
        return df

    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# --- Load data if uploaded ---
if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        # KPIs
        st.header("📊 Overall Audit Performance")
        col1, col2, col3 = st.columns(3)

        with col1:
            total_audits = len(df)
            st.metric("Total Audits Conducted", total_audits)

        with col2:
            if 'questionarie_result' in df.columns:
                pass_rate = (df['questionarie_result'].str.lower() == 'pass').mean() * 100
                st.metric("Overall Pass Rate", f"{pass_rate:.2f}%")
            else:
                st.metric("Overall Pass Rate", "Data Missing")

        with col3:
            if 'score' in df.columns:
                avg_score = df['score'].mean()
                st.metric("Average Score", f"{avg_score:.2f}")
            else:
                st.metric("Average Score", "Data Missing")

        # Visuals
        st.header("📈 Key Trends & Patterns")

        if 'questionarie_result' in df.columns and 'month' in df.columns:
            st.subheader("Monthly Audit Performance")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(data=df, x='month', hue='questionarie_result', palette='viridis', ax=ax)
            ax.set_title("Monthly Audit Performance")
            st.pyplot(fig)

        if 'questionarie_result' in df.columns and 'site' in df.columns:
            st.subheader("Top Sites with Most Failures")
            fail_counts = df[df['questionarie_result'].str.lower() == 'failure']['site'].value_counts().head(5)
            if not fail_counts.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=fail_counts.values, y=fail_counts.index, palette='Reds_r', ax=ax)
                ax.set_title("Top 5 Sites with Highest Failure Rates")
                st.pyplot(fig)
            else:
                st.info("No failures recorded in the selected data.")

        if 'score' in df.columns:
            st.subheader("Distribution of Audit Scores")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(data=df, x='score', kde=True, color='skyblue', ax=ax)
            ax.set_title("Distribution of Audit Scores")
            st.pyplot(fig)

        # Optional Filter
        st.subheader("🔍 Explore Data by Month (Optional)")
        available_months = sorted(df['month'].dropna().unique())
        selected_month = st.selectbox("Select Month:", ['All'] + list(available_months))
        if selected_month != 'All':
            filtered_df = df[df['month'] == selected_month]
            st.write(f"**Details for {selected_month} Audits**")
            st.dataframe(filtered_df[['site', 'questionarie_result', 'score']].style.highlight_max(subset=['score'], axis=0))
        else:
            st.write("**Overall Data Table**")
            st.dataframe(df[['month', 'site', 'questionarie_result', 'score']].head())

        # Conclusion
        st.header("🎯 Conclusion: Embracing Data for a Brighter Future")
        st.markdown("""
        This initial exploration highlights the significant potential of data analysis to drive operational excellence within our company.
        By moving towards a data-driven culture, we can make more informed decisions, optimize our resources, and ultimately deliver greater value.
        """)
    else:
        st.warning("The file could not be processed. Please check its structure and try again.")
else:
    st.sidebar.info("Please upload a file to display the dashboard.")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("Developed for Cleaning Audit Analysis – compatible with Excel (.xlsx) and CSV (.csv) files.")
