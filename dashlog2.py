import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import numpy as np

# --- Authentication ---
def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username == "SalesEmployee" and password == "SalesDash":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")
        return False
    return True

# --- Page Configuration ---
st.set_page_config(layout="wide")  # Use the full width of the screen

# Inject custom CSS to control vertical height, spacing, and sidebar styling
st.markdown(
    """
    <style>
    /* Set the main container to fit within viewport height */
    .executive-container, .sales-manager-container, .marketing-container {
        max-height: calc(100vh - 100px);  /* Account for Streamlit header */
        overflow-y: auto;  /* Allow internal scrolling if needed */
        padding: 0px;
        margin: 0px;
    }
    /* Reduce padding and margins for all elements */
    .stMetric, .stPlotlyChart, .stMarkdown {
        padding: 2px !important;
        margin: 2px !important;
    }
    /* Ensure columns have minimal spacing */
    .stColumns > div {
        padding: 2px !important;
    }
    /* Reduce subheader font size */
    h3 {
        font-size: 1.2rem !important;
        margin-bottom: 5px !important;
    }
    /* Ensure charts fit tightly */
    .plotly-chart {
        margin: 0px !important;
    }
    /* Increase font size for sidebar elements */
    [data-testid="stSidebar"] {
        font-size: 1.8rem !important;  /* Base font size for sidebar text */
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.8rem !important;  /* Larger font for titles */
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio div {
        font-size: 1.4rem !important;  /* Font size for widget labels and options */
    }
    [data-testid="stSidebar"] .stMultiSelect label {
        font-size: 1.4rem !important;  /* Font size for multiselect label */
    }
    /* Style the multiselect options (product names) */
    [data-testid="stSidebar"] .stMultiSelect div[role="option"] {
        color: blue !important;  /* Set product options text to blue */
    }
    [data-testid="stSidebar"] .stSelectbox span {
        font-size: 1.4rem !important;  /* Font size for selectbox labels and options */
    }
    [data-testid="stSidebar"] .stDateInput label,
    [data-testid="stSidebar"] .stDateInput div {
        font-size: 1.4rem !important;  /* Font size for date input labels and options */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Check authentication before proceeding
if not check_authentication():
    st.stop()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("ai_solution_transactions.csv", parse_dates=['Timestamp', 'Login Timestamp'])
    return df

df = load_data()

# --- Sidebar - Filters ---
try:
    st.sidebar.image("logo.png", width=150)  # Adjust width for sidebar
except FileNotFoundError:
    st.sidebar.warning("Logo file 'logo.png' not found. Please ensure the file is in the correct directory.")

# Date Range Filter
min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()
start_date, end_date = st.sidebar.date_input("Select Date Range:", (min_date, max_date), min_date, max_date)
df_filtered_date = df[(df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)]

# Product Name Filter
product_names = df_filtered_date['Product Name'].unique()
selected_products = st.sidebar.multiselect("Filter by Product:", product_names, default=product_names)
df_filtered = df_filtered_date[df_filtered_date['Product Name'].isin(selected_products)]

st.sidebar.markdown("---")

# --- Sidebar - User Role Selector ---
st.sidebar.title("User Role")
user_role = st.sidebar.radio("Select your role:", ("Executive", "Sales Manager", "Sales Agent", "Marketing Team Leader"))

# --- Filter by Referral Source if Marketing Team Leader role ---
if user_role == "Marketing Team Leader":
    referral_sources = df_filtered['Referral Source'].unique()
    selected_referrals = st.sidebar.multiselect("Filter by Referral Source:", referral_sources, default=referral_sources)
    df_filtered = df_filtered[df_filtered['Referral Source'].isin(selected_referrals)]

# Logout Button
st.sidebar.markdown("---")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

# --- Main App ---
st.title("AI-Solution Sales Dashboard")

# --- EXECUTIVE VIEW ---
if user_role == "Executive":
    with st.container():
        st.markdown('<div class="executive-container">', unsafe_allow_html=True)

        # KPIs
        total_revenue = df_filtered['Final Sale Amount'].sum()
        total_visits = df_filtered['Total Number of Visits'].sum()
        total_transactions = df_filtered['Number of Completed Transactions'].sum()
        avg_conversion = df_filtered['Conversion Rate'].mean() / 100 if not df_filtered['Conversion Rate'].empty else 0
        total_discount = df_filtered['Discount Amount'].sum()

        revenue_target = 5_000_000
        conversion_target = 0.30

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"${total_revenue:,.2f}",
                    delta=f"{((total_revenue/revenue_target)-1)*100:.1f}%" if total_revenue < revenue_target else f"🟢 +{((total_revenue/revenue_target)-1)*100:.1f}%")
        col2.metric("Total Discount", f"${total_discount:,.2f}")
        col3.metric("Total Visits", f"{total_visits:,.0f}")
        col4.metric("Total Transactions", f"{total_transactions:,.0f}")

        st.subheader("Performance Overview")

        col_g1, col_g2, col_pred = st.columns(3)

        with col_g1:
            fig_g1 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=total_revenue,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Revenue vs Target<br><span style='font-size:0.8em; color:red'>Red Line: Target</span>"},
                delta={'reference': revenue_target},
                gauge={
                    'axis': {'range': [None, revenue_target * 1.5]},
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, revenue_target * 0.5], 'color': "red"},
                        {'range': [revenue_target * 0.5, revenue_target], 'color': "yellow"},
                        {'range': [revenue_target, revenue_target * 1.5], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': revenue_target
                    }
                }
            ))
            fig_g1.add_annotation(
                x=0.5,
                y=-0.3,
                xref="paper",
                yref="paper",
                text="Legend: Red: Low Sales, Yellow: Moderate Sales, Green: Good Sales",
                showarrow=False,
                font=dict(size=10),
                align="center"
            )
            fig_g1.update_layout(
                height=280,
                margin=dict(b=50)
            )
            st.plotly_chart(fig_g1, use_container_width=False)

        with col_g2:
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_conversion * 100,
                number={'suffix': "%", 'valueformat': ".1f"},
                delta={'reference': conversion_target * 100, 'relative': True, 'valueformat': ".1f"},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg Conversion Rate<br><span style='font-size:0.8em; color:red'>Red Line: 30% Target</span>"},
                gauge={
                    'axis': {'range': [0, 100], 'tickformat': ".0f"},
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 20], 'color': "red"},
                        {'range': [20, 30], 'color': "yellow"},
                        {'range': [30, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': conversion_target * 100
                    }
                }
            ))
            fig_g2.add_annotation(
                x=0.5,
                y=-0.3,
                xref="paper",
                yref="paper",
                text="Legend: Red: Low Sales, Yellow: Moderate Sales, Green: Good Sales",
                showarrow=False,
                font=dict(size=10),
                align="center"
            )
            fig_g2.update_layout(
                height=280,
                margin=dict(b=50)
            )
            st.plotly_chart(fig_g2, use_container_width=False)

        with col_pred:
            monthly_sales = df_filtered.groupby(pd.Grouper(key='Timestamp', freq='M'))["Final Sale Amount"].sum().reset_index()

            if len(monthly_sales) >= 2:
                monthly_sales['Month_Number'] = range(1, len(monthly_sales) + 1)
                model = LinearRegression()
                X = monthly_sales[['Month_Number']]
                y = monthly_sales['Final Sale Amount']
                model.fit(X, y)
                num_future_months = 6
                future_month_numbers = np.array(range(len(monthly_sales) + 1, len(monthly_sales) + num_future_months + 1)).reshape(-1, 1)
                future_predictions = model.predict(future_month_numbers)
                last_date = monthly_sales['Timestamp'].max()
                future_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(num_future_months)]
                predicted_sales = pd.DataFrame({
                    'Timestamp': future_dates,
                    'Sales': future_predictions,
                    'Month': [d.strftime('%Y-%m') for d in future_dates],
                    'Type': ['Predicted Sales'] * num_future_months
                })
                fig_pred = px.line(
                    predicted_sales,
                    x='Month',
                    y='Sales',
                    color='Type',
                    title='Forecasted Monthly Sales',
                    labels={'Sales': 'Sales Amount ($)'},
                    markers=True
                )
                fig_pred.update_layout(
                    height=250,
                    xaxis_title='Month',
                    yaxis_title='Sales Amount ($)',
                    legend_title='Data Type'
                )
                st.plotly_chart(fig_pred, use_container_width=False)
            else:
                st.info("Insufficient data for sales prediction. At least 2 months of data are required.")

        c1, c2, c3 = st.columns(3)

        with c1:
            monthly_target = 40_000_000
            monthly_revenue = df_filtered.groupby(pd.Grouper(key='Timestamp', freq='M'))["Final Sale Amount"].sum().reset_index()
            monthly_revenue["Month"] = monthly_revenue['Timestamp'].dt.strftime('%Y-%m')
            monthly_revenue["Target Revenue"] = monthly_target / monthly_revenue["Month"].nunique() if not monthly_revenue.empty else 0

            if not monthly_revenue.empty:
                fig1 = px.line(monthly_revenue, x="Month", y=["Final Sale Amount", "Target Revenue"],
                               title="Monthly Revenue Trend")
                fig1.update_layout(height=250)
                st.plotly_chart(fig1, use_container_width=False)
            else:
                st.info("No data available for the selected date range.")

        with c2:
            product_sales = df_filtered.groupby("Product Name")["Final Sale Amount"].sum().reset_index().sort_values(by="Final Sale Amount", ascending=False)
            fig2 = px.bar(product_sales, x="Product Name", y="Final Sale Amount", title="Revenue by Product")
            fig2.update_layout(height=250)
            st.plotly_chart(fig2, use_container_width=False)

        with c3:
            country_sales = df_filtered.groupby("Country")["Final Sale Amount"].sum().reset_index()
            fig_map = px.choropleth(
                country_sales,
                locations="Country",
                locationmode="country names",
                color="Final Sale Amount",
                color_continuous_scale="Viridis",
                title="Revenue by Country"
            )
            fig_map.update_geos(showcoastlines=True, projection_type="equirectangular")
            fig_map.update_layout(height=350)
            st.plotly_chart(fig_map, use_container_width=False)

        st.markdown('</div>', unsafe_allow_html=True)

# --- SALES MANAGER VIEW ---
elif user_role == "Sales Manager":
    with st.container():
        st.markdown('<div class="sales-manager-container">', unsafe_allow_html=True)
        st.header("Team & Product Insights")

        if "AI Debugging Suite" in df_filtered['Product Name'].unique():
            st.success("The top performing product is the AI Debugging Suite")
        else:
            st.warning("Note: AI Debugging Suite is not in the filtered data, but it is considered the top performing product.")

        with st.container():
            st.markdown('<div class="sub-container">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            with col1:
                agent_sales = df_filtered.groupby("Sales Agent Name")["Final Sale Amount"].sum().reset_index()
                fig1 = px.bar(agent_sales, x="Sales Agent Name", y="Final Sale Amount",
                              title="Sales by Agent", text_auto=".2s")
                fig1.update_layout(xaxis_title="Sales Agent", yaxis_title="Total Revenue ($)", height=250)
                st.plotly_chart(fig1, use_container_width=False)

            with col2:
                transactions_per_agent = df_filtered.groupby("Sales Agent Name")["Number of Completed Transactions"].sum().reset_index()
                fig_trans = px.bar(transactions_per_agent, x="Sales Agent Name", y="Number of Completed Transactions",
                                   title="Transactions per Agent", text_auto=".0f")
                fig_trans.update_layout(xaxis_title="Sales Agent", yaxis_title="Transactions", height=250)
                st.plotly_chart(fig_trans, use_container_width=False)

            with col3:
                monthly_data = df_filtered.groupby(pd.Grouper(key='Timestamp', freq='M')).agg({
                    "Total Number of Visits": "sum",
                    "Final Sale Amount": "sum"
                }).reset_index()
                if not monthly_data.empty:
                    monthly_data["Month"] = monthly_data['Timestamp'].dt.strftime('%Y-%m')
                    fig_visits_sales = go.Figure()
                    fig_visits_sales.add_trace(
                        go.Scatter(
                            x=monthly_data["Month"],
                            y=monthly_data["Total Number of Visits"],
                            name="Total Visits",
                            yaxis="y1",
                            mode="lines+markers"
                        )
                    )
                    fig_visits_sales.add_trace(
                        go.Scatter(
                            x=monthly_data["Month"],
                            y=monthly_data["Final Sale Amount"],
                            name="Sales ($)",
                            yaxis="y2",
                            mode="lines+markers"
                        )
                    )
                    fig_visits_sales.update_layout(
                        title="Total Visits vs. Sales",
                        xaxis=dict(title="Month"),
                        yaxis=dict(title="Total Visits", side="left"),
                        yaxis2=dict(title="Sales ($)", side="right", overlaying="y"),
                        height=250
                    )
                    st.plotly_chart(fig_visits_sales, use_container_width=False)
                else:
                    st.info("No visits or sales data available.")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="sub-container">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            with col1:
                product_sales = df_filtered.groupby("Product Name")["Final Sale Amount"].sum().sort_values(ascending=False).reset_index()
                fig2 = px.pie(product_sales, names="Product Name", values="Final Sale Amount",
                              title="Sales by Product", hole=0.3)
                fig2.update_traces(textinfo='percent+label')
                fig2.update_layout(height=250)
                st.plotly_chart(fig2, use_container_width=False)

            with col2:
                referral_sales = df_filtered.groupby("Referral Source")["Final Sale Amount"].sum().reset_index()
                if not referral_sales.empty:
                    fig_ref = px.bar(referral_sales, x="Referral Source", y="Final Sale Amount",
                                     title="Sales by Referral Source", text_auto=".2s")
                    fig_ref.update_layout(height=250, yaxis_title="Sales ($)")
                    st.plotly_chart(fig_ref, use_container_width=False)
                else:
                    st.info("No referral source data available.")

            with col3:
                country_sales = df_filtered.groupby("Country")["Final Sale Amount"].sum().reset_index()
                if not country_sales.empty:
                    fig_loc = px.bar(country_sales, x="Country", y="Final Sale Amount",
                                     title="Sales by Location", text_auto=".2s")
                    fig_loc.update_layout(height=250, yaxis_title="Sales ($)")
                    st.plotly_chart(fig_loc, use_container_width=False)
                else:
                    st.info("No location data available.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# --- SALES AGENT VIEW ---
elif user_role == "Sales Agent":
    # Define agent names and targets
    agent_targets = {
        "Alice Smith": 2_500_000,
        "Bob Johnson": 2_000_000,
        "Charlie Lee": 1_800_000,
        "Dana Kim": 2_200_000,
        "Charlie Brown": 2_000_000
    }

    # Dropdown for selecting agent
    selected_agent_personal = st.selectbox(
        "Select Sales Agent",
        list(agent_targets.keys()),
        index=0  # Default to the first agent
    )

    st.header(f"Welcome, {selected_agent_personal}")

    # Rest of the existing code for sales target and metrics
    if selected_agent_personal in agent_targets:
        personal_target = agent_targets[selected_agent_personal]
    else:
        st.warning(f"No sales target found for {selected_agent_personal}. Please check the configuration.")
        personal_target = 0

    total_sales = df_filtered['Final Sale Amount'].sum()
    total_commission = df_filtered['Commission Amount'].sum()
    conversion_rate = df_filtered['Number of Completed Transactions'].sum() / df_filtered['Total Number of Visits'].sum() if df_filtered['Total Number of Visits'].sum() > 0 else 0

    st.info(f"Your Annual Sales Target: **${personal_target:,.2f}**")
    if total_sales >= personal_target and personal_target > 0:
        st.success("Congratulations! You've met your annual sales target!")
    elif personal_target > 0:
        st.info(f"You're at **{(total_sales / personal_target):.1%}** of your annual target. Keep going!")

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Your Sales",
        f"${total_sales:,.2f}",
        delta=f"{((total_sales / personal_target) - 1) * 100:.1f}%" if personal_target > 0 else "N/A"
    )
    col2.metric("Commission Earned", f"${total_commission:,.2f}")
    col3.metric("Avg. Conversion Rate", f"{conversion_rate:.2%}")

    chart_col1, chart_col2, chart_col3 = st.columns(3)

    with chart_col1:
        monthly = df_filtered.groupby(pd.Grouper(key='Timestamp', freq='M'))["Final Sale Amount"].sum().reset_index()
        if not monthly.empty:
            monthly["Month"] = monthly['Timestamp'].dt.strftime('%Y-%m')
            monthly_target = personal_target / 12 if personal_target > 0 else 0
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=monthly["Month"],
                y=monthly["Final Sale Amount"],
                mode="lines+markers",
                name="Monthly Sales",
                line=dict(color="blue")
            ))
            fig1.add_hline(
                y=monthly_target,
                line_color="red",
                annotation_text=f"Monthly Target (${monthly_target:,.0f})",
                annotation_position="top left"
            )
            fig1.update_layout(
                title="Your Monthly Sales vs Target",
                xaxis_title="Month",
                yaxis_title="Sales Amount",
                height=250
            )
            st.plotly_chart(fig1, use_container_width=False)
        else:
            st.info("No monthly sales data available for the current filters.")

    with chart_col2:
        product = df_filtered.groupby("Product Name")["Final Sale Amount"].sum().reset_index()
        fig2 = px.bar(product, x="Product Name", y="Final Sale Amount", title="Your Product Sales")
        fig2.update_layout(height=250)
        st.plotly_chart(fig2, use_container_width=False)

    with chart_col3:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=total_sales,
            delta={'reference': personal_target},
            title={'text': f"{selected_agent_personal}'s Sales Target"},
            gauge={
                'axis': {'range': [None, personal_target * 1.5 if personal_target > 0 else 1]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, personal_target * 0.5], 'color': "red"},
                    {'range': [personal_target * 0.5, personal_target], 'color': "yellow"},
                    {'range': [personal_target, personal_target * 1.5], 'color': "green"}
                ] if personal_target > 0 else [],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': personal_target
                } if personal_target > 0 else {}
            }
        ))
        gauge_fig.add_annotation(
            x=0.5,
            y=-0.3,
            xref="paper",
            yref="paper",
            text="Legend: Red: Low Sales, Yellow: Moderate Sales, Green: Good Sales",
            showarrow=False,
            font=dict(size=10),
            align="center"
        )
        gauge_fig.update_layout(
            height=300,
        )
        st.plotly_chart(gauge_fig, use_container_width=False)

# --- MARKETING TEAM LEADER VIEW ---
elif user_role == "Marketing Team Leader":
    with st.container():
        st.markdown('<div class="marketing-container">', unsafe_allow_html=True)

        # Alert for top-performing product
        product_sales = df_filtered.groupby("Product Name")["Final Sale Amount"].sum().reset_index()
        top_product = product_sales.loc[product_sales["Final Sale Amount"].idxmax(), "Product Name"] if not product_sales.empty else "N/A"
        if top_product == "AI Debugging Suite":
            st.success("The top-performing product is the AI Debugging Suite!")
        elif top_product != "N/A":
            st.info(f"Top-performing product: {top_product}")

        # KPIs
        total_visits = df_filtered['Total Number of Visits'].sum()
        total_revenue = df_filtered['Final Sale Amount'].sum()
        avg_conversion = df_filtered['Conversion Rate'].mean() / 100 if not df_filtered['Conversion Rate'].empty else 0
        total_transactions = df_filtered['Number of Completed Transactions'].sum()
        visits_to_sales_ratio = total_visits / total_transactions if total_transactions > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Visits", f"{total_visits:,.0f}")
        col2.metric("Avg. Conversion Rate", f"{avg_conversion:.2%}")
        col3.metric("Total Revenue", f"${total_revenue:,.2f}")
        col4.metric("Visits per Transaction", f"{visits_to_sales_ratio:.1f}")

        # First Container: Campaign and Product Insights
        with st.container():
            st.markdown('<div class="sub-container">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)

            with col1:
                # Visits and Sales Trends Over Time
                monthly_data = df_filtered.groupby(pd.Grouper(key='Timestamp', freq='M')).agg({
                    "Total Number of Visits": "sum",
                    "Final Sale Amount": "sum"
                }).reset_index()
                if not monthly_data.empty:
                    monthly_data["Month"] = monthly_data['Timestamp'].dt.strftime('%Y-%m')
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=monthly_data["Month"],
                        y=monthly_data["Total Number of Visits"],
                        name="Total Visits",
                        yaxis="y1",
                        mode="lines+markers",
                        line=dict(color="blue")
                    ))
                    fig_trend.add_trace(go.Scatter(
                        x=monthly_data["Month"],
                        y=monthly_data["Final Sale Amount"],
                        name="Sales ($)",
                        yaxis="y2",
                        mode="lines+markers",
                        line=dict(color="green")
                    ))
                    fig_trend.update_layout(
                        title="Visits vs. Sales Trend",
                        xaxis=dict(title="Month"),
                        yaxis=dict(title="Total Visits", side="left"),
                        yaxis2=dict(title="Sales ($)", side="right", overlaying="y"),
                        height=250
                    )
                    st.plotly_chart(fig_trend, use_container_width=False)
                else:
                    st.info("No visits or sales data available.")

            with col2:
                # Sales by Referral Source
                referral_sales = df_filtered.groupby("Referral Source")["Final Sale Amount"].sum().reset_index()
                if not referral_sales.empty:
                    fig_ref = px.bar(referral_sales, x="Referral Source", y="Final Sale Amount",
                                     title="Sales by Referral Source", text_auto=".2s")
                    fig_ref.update_layout(height=250, yaxis_title="Sales ($)")
                    st.plotly_chart(fig_ref, use_container_width=False)
                else:
                    st.info("No referral source data available.")

            with col3:
                # Conversion Rate by Product
                product_conversion = df_filtered.groupby("Product Name")["Conversion Rate"].mean().reset_index()
                fig_conv = px.pie(product_conversion, names="Product Name", values="Conversion Rate",
                                  title="Conversion Rate by Product", hole=0.3)
                fig_conv.update_traces(textinfo='percent+label')
                fig_conv.update_layout(height=250)
                st.plotly_chart(fig_conv, use_container_width=False)
            st.markdown('</div>', unsafe_allow_html=True)

        # Second Container: Regional and Discount Insights
        with st.container():
            st.markdown('<div class="sub-container">', unsafe_allow_html=True)
            col1, col3 = st.columns(2)

            with col1:
                # Sales by Country
                country_sales = df_filtered.groupby("Country")["Final Sale Amount"].sum().reset_index()
                if not country_sales.empty:
                    fig_map = px.choropleth(
                        country_sales,
                        locations="Country",
                        locationmode="country names",
                        color="Final Sale Amount",
                        color_continuous_scale="Viridis",
                        title="Sales by Country"
                    )
                    fig_map.update_geos(showcoastlines=True, projection_type="equirectangular")
                    fig_map.update_layout(height=350)
                    st.plotly_chart(fig_map, use_container_width=False)
                else:
                    st.info("No country sales data available.")

            with col3:
                # Marketing Funnel
                funnel_data = pd.DataFrame({
                    "Stage": ["Visits", "Transactions"],
                    "Value": [total_visits, total_transactions]
                })
                fig_funnel = px.funnel(funnel_data, x="Value", y="Stage", title="Marketing Funnel")
                fig_funnel.update_layout(height=250)
                st.plotly_chart(fig_funnel, use_container_width=False)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)