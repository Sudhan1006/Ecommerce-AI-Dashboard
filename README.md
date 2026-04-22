E-Commerce AI Dashboard
This is an end-to-end data science and machine learning application I built as part of a 14-day data engineering and analytics challenge.

Instead of just leaving my machine learning models in a Jupyter Notebook, I deployed them into an interactive Streamlit web app so non-technical stakeholders (like a marketing or sales team) can actually use them to make business decisions.

Live Demo: https://ecommerce-ai-dashboard-atgujzmaxezqhfkovuy6yu.streamlit.app/

What it does
The app processes over 500,000 rows of transactional e-commerce data and features two main sections:

Executive Dashboard: A real-time overview of top-line business metrics (Total Revenue, Unique Orders, Total Customers) along with dynamic charts showing revenue over time and top-performing products. Everything can be filtered by country.

AI Customer Segmentation: A live K-Means clustering algorithm that automatically groups customers based on their spending habits (Recency, Frequency, Monetary value). If you filter the dashboard by a specific country, the AI recalculates the RFM matrix and re-trains the model on the fly to show you exactly who the VIPs and churn-risk customers are in that specific region.

Tech Stack
Language: Python

Data Processing: Pandas

Machine Learning: Scikit-Learn (K-Means Clustering, Random Forest)

Frontend/Deployment: Streamlit

Running it locally
If you want to pull this down and run it on your own machine:

Clone the repository:

Bash
git clone https://github.com/yourusername/ecommerce-ai-dashboard.git
cd ecommerce-ai-dashboard
Install the required dependencies:

Bash
pip install -r requirements.txt
Run the Streamlit app:

Bash
streamlit run app.py
