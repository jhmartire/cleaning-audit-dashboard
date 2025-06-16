# Cleaning Audit Dashboard 🧹

A Streamlit dashboard for visualising and analysing cleaning audit data from an Excel spreadsheet. Built to provide clear, interactive insights for facility managers and stakeholders.

---

## 🚀 Key Features

* 📥 **Excel Upload**: Import your audit data via a simple file uploader.
* 📊 **Interactive Visualisations**: Dynamic charts and heatmaps display scores, trends, and correlations.
* 🗓 **Filtering Options**: Filter audits by date range, site status, and other criteria.
* 📈 **Performance Metrics**: View top-performing and underperforming sites at a glance.
* 📂 **Export Results**: Download filtered data back to Excel for reporting or further analysis.

---

## 🛠️ Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/username/cleaning-audit-dashboard.git
cd cleaning-audit-dashboard
```

2. **Create and activate a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```
├── app.py                 # Streamlit application entrypoint
├── requirements.txt       # Python dependencies
├── data/                  # Sample audit Excel files
├── utils/                 # Helper functions for data processing
├── images/                # Static assets (e.g., Cleaning.png banner)
└── README.md              # Project documentation
```

---

## ☁️ Deployment on Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and select **New App**.
3. Connect your GitHub repo and set the main file to `app.py`.
4. Add any required secrets (e.g., none for this public dashboard).
5. Deploy and share the live link with stakeholders.

---

## 📋 Requirements

* Python >= 3.8
* Streamlit
* pandas, numpy, matplotlib, seaborn, plotly (specified in `requirements.txt`)

---

## 🔐 Security & Privacy

* **Do not** upload sensitive or personally identifiable information to this repository.
* Ensure that any audit data you upload is anonymised before sharing.

---

## 👤 Author

Created with ❤️ by [@jhmartire](https://github.com/jhmartire)
Assistant Manager & Data Science Trainee at Andron FM
