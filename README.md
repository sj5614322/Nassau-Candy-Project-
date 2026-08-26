# 🍬 Nassau Candy — Factory-to-Customer Shipping Route Efficiency Dashboard

An interactive Streamlit analytics dashboard that analyzes shipping route efficiency for **Nassau Candy Distributor**, identifying bottlenecks, comparing factory-to-state routes, and benchmarking ship-mode performance — with a one-click exportable Word report.

**🔗 Live App:** https://candy-logistics-dashboard.streamlit.app/
**📄 Research Paper:** [Add your paper link here]
**🎥 Demo Video:** [Add your feedback/demo video link here]

---

## 📌 Project Overview

Nassau Candy ships products from four factories (Lot's O' Nuts, Wicked Choccy's, Sugar Shack, The Other Factory) to customers across the United States. This project analyzes ~10,000 historical order records to answer:

- Which **factory → state routes** are most/least efficient?
- Which **states** are geographic bottlenecks (high volume + high lead time)?
- How do different **ship modes** compare on speed, delay rate, and cost?
- Which individual **orders are statistical outliers** on their route?
- How can regional/route-level performance be benchmarked using a normalized **Efficiency Score**?

## ✨ Features

| Module | Description |
|---|---|
| **Overview KPIs** | Total shipments, average lead time, delay %, active routes |
| **Route Rankings** | Sortable table of all Factory → State routes with Efficiency Score |
| **Geographic View** | US choropleth map highlighting congestion-prone states |
| **Ship Mode Analysis** | Lead time, delay rate, and cost-time tradeoff by ship mode |
| **Route Comparison** | Head-to-head comparison of any two routes with distribution plots |
| **Route Drill-Down** | State-level factory breakdown + order timeline |
| **Methodology Tab** | Full transparency on data cleaning, feature engineering & formulas |
| **Word Report Export** | One-click `.docx` executive summary of the current filtered view |

## 🏗️ Tech Stack

- **Frontend/App:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly Express
- **Reporting:** python-docx
- **Language:** Python 3.10+

## 📁 Repository Structure

```
├── app.py                 # Streamlit UI, layout, tabs, styling
├── data_processing.py      # Cleaning, feature engineering, aggregation, report generation
├── data/
│   └── Nassau Candy Distributor.csv
├── assets/
│   └── logo_icon.png, logo_sidebar.png
├── requirements.txt
└── README.md
```

## ⚙️ Methodology (Summary)

1. **Data Cleaning:** Trim text fields; parse `dd-mm-yyyy` dates; drop rows with missing dates or negative lead time.
2. **Feature Engineering:** `Lead_Time = Ship Date − Order Date`; `Factory` derived from `Product Name`; `Route = Factory → State`.
3. **Aggregation:** Group by route for total shipments, average lead time, and lead-time standard deviation.
4. **Efficiency Score:** Min-max normalized average lead time, inverted and scaled 0–100 (higher = better), relative to routes in the current filtered view.
5. **Bottleneck Flagging:** A state is "congestion-prone" when both its average lead time and shipment volume are above the median across all states.

**Data quality note:** `Ship Date` values in the source dataset fall several years after `Order Date` and barely vary by ship mode — a known data-generation artifact, not real-world performance. All figures should be read as **relative comparisons** between routes/states/modes rather than calendar-accurate day counts. This is documented transparently inside the app's Methodology tab.

## 🚀 Running Locally

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## ☁️ Deployment

This app is deployed on **Streamlit Community Cloud**. See the deployment steps in the project research paper / submission notes.

## 📊 Dataset

`Nassau Candy Distributor.csv` — ~10,000 order-level records with Order Date, Ship Date, Ship Mode, Product Name, geography (City/State/Region), Sales, Units, Gross Profit, and Cost.

## 👤 Author

[Your Name] — [Your Email / LinkedIn]

## 📄 License

This project is submitted as part of an internship/academic project. [Add license if applicable, e.g., MIT.]
