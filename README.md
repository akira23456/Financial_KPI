Financial KPI Analytics Dashboard
A comprehensive interactive dashboard for analyzing financial key performance indicators (KPIs) across multiple stocks, featuring machine learning predictions and historical financial metrics.
Overview
This project provides an interactive web-based dashboard that visualizes financial data and ML model predictions for 260+ stocks. The dashboard displays ROE (Return on Equity), Debt-to-Equity ratios, Current Ratios, and other key financial metrics with interactive charts and filtering capabilities.
Features
📊 Three Main Views
Overview Tab

Average ROE, Debt/Equity, and Current Ratio metrics
Time series analysis of financial metrics
ROE distribution across stocks
Debt-to-Equity analysis with color-coded visualizations

Predictions Tab

Model performance metrics (MAE for ROE and Debt/Equity predictions)
Top 10 prediction errors with stock details
Actual vs Predicted scatter plots for both ROE and Debt/Equity
Comprehensive model evaluation statistics

Assets Tab

Asset composition breakdown (Cash, Inventory, Net Receivables)
Total Assets vs Long-term Debt correlation analysis
Current Ratio distribution with health indicators

🎯 Interactive Features

Stock-specific filtering (260+ stocks available)
Dynamic chart updates
Color-coded metrics (green for positive, red for negative)
Responsive design for all screen sizes
Hover tooltips with detailed information

Data Pipeline
1. Data Collection
Raw financial data was collected for 260+ stocks including:

Balance sheet metrics
Income statement data
Financial ratios
Historical time series data

2. Data Cleaning Process
Initial Data Processing:
- Removed null and invalid entries
- Standardized column names and data types
- Handled missing values appropriately
- Filtered out stocks with insufficient data
Data Transformation:

Calculated financial ratios:

current_ratio = current_assets / current_liabilities
debt_to_equity = total_debt / total_equity
roe = net_income / shareholders_equity
debt_to_assets = total_debt / total_assets
cash_to_assets = cash / total_assets
inventory_to_assets = inventory / total_assets
receivables_to_assets = net_receivables / total_assets



Outlier Removal:

Applied IQR method to remove extreme outliers
Capped ratios at reasonable thresholds
Validated data consistency across time periods

3. Data Combination
File Structure:
FKQ.csv / ratios_for_ml.csv (1000 rows)
├── Core financial metrics
├── Historical time series data
└── Calculated financial ratios

combined_financial_predictions.csv (200 rows)
├── Actual ROE and Debt/Equity values
├── Predicted ROE and Debt/Equity values
└── Stock identifiers and dates

debt_to_equity_predictions.csv (200 rows)
├── Actual Debt/Equity values
└── Predicted Debt/Equity values

financial_kpi_predictions.csv (200 rows)
├── Actual ROE values
└── Predicted ROE values
Combination Logic:

Merged datasets on stock ticker and end date
Aligned prediction data with historical actuals
Ensured data consistency across all files
Created unified data structure for dashboard consumption

4. Dashboard Creation
The HTML dashboard was built using:

React for component-based UI
Recharts for data visualizations
Tailwind CSS for modern styling
PapaParse for CSV parsing
Vanilla JavaScript for data loading and processing

Project Structure
financial-kpi-dashboard/
├── dashboard.html                          # Main dashboard file
├── data/
│   ├── ratios_for_ml.csv                  # Main financial data (996 rows)
│   ├── combined_financial_predictions.csv  # Combined predictions (200 rows)
│   ├── debt_to_equity_predictions.csv     # Debt predictions (200 rows)
│   └── financial_kpi_predictions.csv      # ROE predictions (200 rows)
└── README.md                               # This file
Installation & Setup
Local Development

Clone the repository:

bashgit clone https://github.com/yourusername/financial-kpi-dashboard.git
cd financial-kpi-dashboard

Add your CSV files to the data folder:

bashmkdir data
# Copy your 4 CSV files into the data/ folder

Serve the HTML file:

bash# Using Python
python -m http.server 8000

# Using Node.js
npx serve

# Using PHP
php -S localhost:8000

Open in browser:

http://localhost:8000/dashboard.html
GitHub Pages Deployment

Push to GitHub:

bashgit add .
git commit -m "Add financial dashboard"
git push origin main

Enable GitHub Pages:

Go to repository Settings
Navigate to Pages section
Source: Deploy from branch
Branch: main
Click Save


Access your dashboard:

https://yourusername.github.io/your-repo-name/dashboard.html
Data Files Description
ratios_for_ml.csv (996 rows, 14 columns)
Main dataset containing historical financial data and calculated ratios.
Columns:

stock - Stock ticker symbol
enddate - Reporting period end date
current_ratio - Current assets / Current liabilities
debt_to_equity - Total debt / Total equity
roe - Return on equity (net income / equity)
totalassets - Total assets (in dollars)
cash - Cash and equivalents
inventory - Inventory value
netreceivables - Net receivables
longtermdebt - Long-term debt
debt_to_assets - Total debt / Total assets
cash_to_assets - Cash / Total assets
inventory_to_assets - Inventory / Total assets
receivables_to_assets - Receivables / Total assets

combined_financial_predictions.csv (200 rows, 6 columns)
ML model predictions compared with actual values.
Columns:

stock - Stock ticker symbol
enddate - Prediction date
Actual_ROE - Actual return on equity
Actual_Debt_to_Equity - Actual debt-to-equity ratio
Predicted_ROE - ML predicted ROE
Predicted_Debt_to_Equity - ML predicted debt-to-equity

debt_to_equity_predictions.csv (200 rows, 2 columns)
Focused predictions for debt-to-equity ratios.
Columns:

Actual_Debt_to_Equity - Actual debt-to-equity ratio
Predicted_Debt_to_Equity - Predicted debt-to-equity ratio

financial_kpi_predictions.csv (200 rows, 2 columns)
Focused predictions for return on equity.
Columns:

Actual_ROE - Actual return on equity
Predicted_ROE - Predicted return on equity

Model Performance
The dashboard displays model performance metrics including:
Mean Absolute Error (MAE):

ROE Predictions: Calculated from 200 prediction data points
Debt/Equity Predictions: Calculated from 200 prediction data points

Visualization Methods:

Scatter plots showing actual vs predicted values
Error distribution analysis
Top 10 worst predictions for model improvement insights

Technology Stack

Frontend Framework: React 18
Charting Library: Recharts 2.5
CSS Framework: Tailwind CSS
Data Parsing: PapaParse 5.4
Build Tool: Babel Standalone (for JSX transformation)
Hosting: GitHub Pages compatible

Usage
Filtering Data
Use the stock dropdown to filter data by specific company or view aggregate metrics across all stocks.
Navigating Tabs

Overview: General financial health metrics and trends
Predictions: Model accuracy and prediction analysis
Assets: Deep dive into asset composition and debt analysis

Interpreting Metrics
ROE (Return on Equity):

Positive values (green) indicate profitability
Negative values (red) indicate losses
Higher values generally indicate better performance

Debt-to-Equity Ratio:

Lower values indicate less leverage
Higher values indicate more debt relative to equity
Industry benchmarks vary

Current Ratio:

Values > 1.5 (green) indicate good liquidity
Values < 1.5 (orange) may indicate liquidity concerns
Measures ability to pay short-term obligations

Data Quality & Limitations
Strengths:

996 historical data points across 260+ stocks
200 prediction data points for model evaluation
Comprehensive financial metrics coverage

Limitations:

Historical data only (not real-time)
Limited to available reporting periods
Prediction model performance varies by stock
Some stocks may have incomplete data

Future Enhancements

 Real-time data integration
 Additional financial metrics (P/E ratio, EPS, etc.)
 Sector-based analysis and filtering
 Export functionality for charts and data
 Advanced filtering options (date ranges, metric thresholds)
 Mobile app version
 API integration for live data updates
 User authentication for personalized views
 Comparison tools for multiple stocks
