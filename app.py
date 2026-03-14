import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="E-Commerce Analytics", layout="wide")
st.title("🛍️ AI-Powered Business Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_ecommerce_sales.csv.gz', compression='gzip')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[df['CustomerID'] != 'Guest_Checkout']
    return df

df = load_data()

st.sidebar.header("⚙️ Dashboard Filters")
all_countries = df['Country'].unique()
selected_countries = st.sidebar.multiselect("Select Country", all_countries, default=['United Kingdom'])

if not selected_countries:
    filtered_df = df
else:
    filtered_df = df[df['Country'].isin(selected_countries)]

tab1, tab2 = st.tabs(["📊 Executive Dashboard", "🤖 AI Customer Segmentation"])

with tab1:
    total_revenue = filtered_df['Total_Revenue'].sum()
    total_orders = filtered_df['InvoiceNo'].nunique()
    total_customers = filtered_df['CustomerID'].nunique()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Unique Orders", f"{total_orders:,}")
    col3.metric("Total Customers", f"{total_customers:,}")

    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("📈 Revenue Over Time")
        daily_rev = filtered_df.groupby(filtered_df['InvoiceDate'].dt.date)['Total_Revenue'].sum()
        st.area_chart(daily_rev)
    with col_chart2:
        st.subheader("🔥 Top 10 Products")
        top_products = filtered_df.groupby('Description')['Total_Revenue'].sum().sort_values(ascending=False).head(10)
        st.bar_chart(top_products)

with tab2:
    st.subheader("Machine Learning: VIPs vs. At-Risk Customers")
    st.markdown("This AI algorithm dynamically groups customers based on their spending habits (RFM) using the filters selected on the left.")

    reference_date = filtered_df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = filtered_df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Total_Revenue': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled).astype(str)

    st.scatter_chart(
        data=rfm,
        x='Recency',
        y='Monetary',
        color='Cluster',
        use_container_width=True
    )

    st.markdown("**Cluster Averages (What do these groups mean?)**")
    cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
    st.dataframe(cluster_summary, use_container_width=True)
