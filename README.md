💹 Financial KPI Analytics Dashboard
📌 Overview

This project demonstrates a full-cycle data analytics and machine learning workflow for evaluating the financial health of over 260 publicly listed companies.
It includes data cleaning and transformation with Python, KPI calculation and validation using Excel, ML model building with PyTorch, and the creation of an interactive HTML dashboard for visualization.

🌐 Interactive Dashboard

https://akira23456.github.io/Financial_KPI/

The dashboard allows users to explore historical financial metrics and machine learning predictions across multiple stocks — including ROE (Return on Equity), Debt-to-Equity Ratio, Current Ratio, and other key performance indicators.

🔎 Project Workflow
1. Data Cleaning (Python + Pandas)

Imported and combined multiple financial CSV files.

Handled missing values, standardized data types, and removed invalid entries.

Calculated essential financial ratios such as:

current_ratio = current_assets / current_liabilities

debt_to_equity = total_debt / total_equity

roe = net_income / shareholders_equity

debt_to_assets = total_debt / total_assets

cash_to_assets = cash / total_assets

inventory_to_assets = inventory / total_assets

receivables_to_assets = net_receivables / total_assets

Applied IQR-based outlier removal and validated consistency across time periods.

2. Data Organization (Excel)

Used Excel for data review and quick KPI comparisons across companies.

Verified ratios and calculations through pivot tables and filters.

Prepared cleaned CSVs for ML model input and dashboard integration.

3. Machine Learning (PyTorch)

Built and trained a PyTorch regression model to predict financial KPIs, including ROE and Debt-to-Equity ratios.

Evaluated performance using Mean Absolute Error (MAE) across 200+ predictions.

Generated predicted vs. actual comparison datasets for dashboard visualization.

4. Visualization (Interactive HTML Dashboard)

Used Claude to use my data and create a fully interactive web-based dashboard using HTML, JavaScript, Recharts, and Tailwind CSS.

Added dynamic views for:

Average financial ratios across companies

Prediction accuracy comparisons

Asset composition breakdowns

Debt vs. equity and liquidity trends

Integrated CSV-based data loading with PapaParse for real-time updates.

🛠 Tools Used

Python & Pandas → Data cleaning, ratio calculations, and transformation

Excel → KPI organization and manual validation

PyTorch → Machine learning model for ROE and Debt/Equity predictions

HTML + Recharts + Tailwind CSS → Interactive dashboard development

Git & GitHub Pages → Version control and live project hosting

🔑 Key Insights

Financial performance varies significantly — firms with higher Current Ratios generally show better ROE.

Debt-to-Equity serves as a major differentiator between high- and low-performing companies.

ROE predictions achieved low mean absolute error, showing strong model generalization.

Interactive visualizations enable quick insights into trends across 260+ stocks, helping identify both risk and growth opportunities.

📈 Dashboard Highlights

Overview Tab → Company-wide KPIs, trends, and ratio distributions.

Predictions Tab → ML model performance and prediction accuracy plots.

Assets Tab → Asset composition, debt levels, and liquidity indicators.

🚀 Future Enhancements

Real-time financial data integration via APIs

Additional KPIs (e.g., EPS, P/E Ratio, EBITDA Margin)

Sector-based and time-period filtering

Export options for dashboard visuals and data

Enhanced ML model comparison (XGBoost, LSTM, etc.)
