import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import gspread
from google.oauth2.service_account import Credentials
from io import StringIO
import os

# Page configuration
st.set_page_config(
    page_title="Advanced Sales Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Success metric card */
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Warning metric card */
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Info metric card */
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Data table styling */
    .stDataFrame {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        overflow: hidden;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Form styling */
    .stForm {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e0e0e0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'gsheet_client' not in st.session_state:
    st.session_state.gsheet_client = None
if 'sheet_url' not in st.session_state:
    st.session_state.sheet_url = ""
if 'data_source' not in st.session_state:
    st.session_state.data_source = "csv"
if 'selected_reps' not in st.session_state:
    st.session_state.selected_reps = []
if 'filter_state' not in st.session_state:
    st.session_state.filter_state = {}

# Helper Functions
def calculate_performance_score(row):
    """Calculate overall performance score based on multiple metrics"""
    try:
        goal_weight = 0.3
        win_rate_weight = 0.25
        activity_weight = 0.25
        csat_weight = 0.2
        
        goal_score = min(float(row.get('% to Goal', 0)) / 100, 1.5) * 100
        win_rate_score = float(row.get('Win Rate (%)', 0))
        activity_score = float(row.get('Daily Activity Score', 0))
        csat_score = (float(row.get('Customer Satisfaction Score', 0)) / 5) * 100
        
        total_score = (
            goal_score * goal_weight +
            win_rate_score * win_rate_weight +
            activity_score * activity_weight +
            csat_score * csat_weight
        )
        
        return round(total_score, 2)
    except:
        return 0

def get_performance_level(score):
    """Get performance level based on score"""
    if score >= 90:
        return "Excellent", "🌟"
    elif score >= 75:
        return "Good", "✅"
    elif score >= 60:
        return "Average", "⚠️"
    else:
        return "Needs Improvement", "❌"

def sync_to_google_sheets(df):
    """Sync dataframe to Google Sheets"""
    if st.session_state.data_source == "gsheet" and st.session_state.gsheet_client:
        try:
            st.session_state.gsheet_client.clear()
            data_to_upload = [df.columns.tolist()] + df.values.tolist()
            st.session_state.gsheet_client.update('A1', data_to_upload)
            return True, "Successfully synced to Google Sheets!"
        except Exception as e:
            return False, f"Error syncing to Google Sheets: {str(e)}"
    return False, "Google Sheets not connected"

# Sidebar - Configuration
st.sidebar.title("🔧 Configuration & Settings")

# Data Source Selection
with st.sidebar.expander("📂 Data Source", expanded=True):
    data_source = st.radio(
        "Select Data Source:",
        ["CSV File", "Google Sheets"],
        key="data_source_radio"
    )
    
    # CSV Upload
    if data_source == "CSV File":
        st.subheader("📁 Upload CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.data_source = "csv"
            st.success("✅ CSV loaded successfully!")
            st.info(f"Loaded {len(st.session_state.df)} records")
    
    # Google Sheets Integration
    else:
        st.subheader("🔐 Google Sheets Login")
        
        service_account_file = st.file_uploader(
            "Upload Service Account JSON",
            type=['json'],
            help="Upload your Google Cloud service account JSON file"
        )
        
        sheet_url = st.text_input(
            "Google Sheets URL:",
            value="https://docs.google.com/spreadsheets/d/10eTuYCcJ7qavCwqT27HRI-vexCwwwKSMDJTv6B6XOr0/edit?usp=sharing",
            help="Paste your Google Sheets URL here"
        )
        
        if service_account_file and sheet_url:
            if st.button("🔗 Connect to Google Sheets", use_container_width=True):
                with st.spinner("Connecting to Google Sheets..."):
                    try:
                        service_account_info = json.load(service_account_file)
                        scope = [
                            'https://spreadsheets.google.com/feeds',
                            'https://www.googleapis.com/auth/drive'
                        ]
                        credentials = Credentials.from_service_account_info(
                            service_account_info,
                            scopes=scope
                        )
                        client = gspread.authorize(credentials)
                        spreadsheet = client.open_by_url(sheet_url)
                        worksheet = spreadsheet.get_worksheet(0)
                        data = worksheet.get_all_values()
                        
                        if len(data) > 1:
                            st.session_state.df = pd.DataFrame(data[1:], columns=data[0])
                            st.session_state.gsheet_client = worksheet
                            st.session_state.sheet_url = sheet_url
                            st.session_state.data_source = "gsheet"
                            st.success("✅ Connected to Google Sheets!")
                            st.info(f"Loaded {len(st.session_state.df)} records")
                        else:
                            st.error("❌ Sheet is empty!")
                            
                    except Exception as e:
                        st.error(f"❌ Connection Error: {str(e)}")

# Sidebar - Quick Actions
if st.session_state.df is not None:
    with st.sidebar.expander("⚡ Quick Actions"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("💾 Save All", use_container_width=True):
                success, message = sync_to_google_sheets(st.session_state.df)
                if success:
                    st.success(message)
                else:
                    st.warning(message)
        
        # Export options
        st.subheader("📥 Export Options")
        
        if st.session_state.df is not None:
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="Download Full CSV",
                data=csv,
                file_name=f"sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

# Sidebar - Analytics Settings
with st.sidebar.expander("📊 Analytics Settings"):
    show_trends = st.checkbox("Show Trends", value=True)
    show_predictions = st.checkbox("Show Predictions", value=False)
    comparison_period = st.selectbox(
        "Comparison Period",
        ["Current Month", "Last 3 Months", "Last 6 Months", "Year to Date"]
    )

# Sidebar - Instructions
with st.sidebar.expander("ℹ️ Help & Instructions"):
    st.markdown("""
    ### Getting Started
    
    **Connect Data Source:**
    1. Upload a CSV file, OR
    2. Connect to Google Sheets with service account JSON
    
    **Google Sheets Setup:**
    1. Go to [Google Cloud Console](https://console.cloud.google.com/)
    2. Create a project and enable Google Sheets API
    3. Create a Service Account
    4. Download the JSON key file
    5. Share your Google Sheet with the service account email
    6. Upload the JSON file and connect
    
    **Features:**
    - 📊 Real-time analytics and visualizations
    - ✏️ Live editing with instant sync
    - 🔍 Advanced filtering and search
    - 📈 Performance tracking and trends
    - 📄 Invoice management
    - 📥 Export data anytime
    
    **Support:**
    For issues, check connection status and ensure proper permissions.
    """)

# Main Content
if st.session_state.df is not None:
    df = st.session_state.df.copy()
    
    # Data preprocessing
    numeric_columns = [
        'Monthly Sales Target', 'Actual Sales (Month)', '% to Goal', 
        'Total Deals Closed (Month)', 'Average Deal Size', 'Pipeline Value', 
        'Win Rate (%)', 'Outbound Calls Made', 'Outbound Emails Sent',
        'Meetings Booked', 'Meetings Completed', 'New Leads Assigned',
        'Leads Contacted', 'Follow-Ups Due', 'Opportunities Created',
        'Opportunities in Progress', 'Opportunities Lost', 'Opportunities Won',
        'Base Salary', 'Commission Earned (Month)', 'Bonuses Earned',
        'Total Compensation to Date', 'Daily Activity Score', 'Notes Logged (Count)',
        'Tasks Completed', 'Tasks Overdue', 'Customer Satisfaction Score'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate additional metrics
    df['Performance Score'] = df.apply(calculate_performance_score, axis=1)
    df['Conversion Rate'] = (df['Opportunities Won'] / df['Opportunities Created'].replace(0, 1) * 100).round(2)
    df['Meeting Completion Rate'] = (df['Meetings Completed'] / df['Meetings Booked'].replace(0, 1) * 100).round(2)
    df['Lead Contact Rate'] = (df['Leads Contacted'] / df['New Leads Assigned'].replace(0, 1) * 100).round(2)
    df['Productivity Index'] = ((df['Outbound Calls Made'] + df['Outbound Emails Sent']) / 2).round(0)
    
    # Header
    st.markdown('<div class="main-header">📊 Advanced Sales Management Dashboard</div>', unsafe_allow_html=True)
    
    # Connection status
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.session_state.data_source == "gsheet":
            st.success(f"🔗 Connected to Google Sheets | {len(df)} records loaded")
        else:
            st.info(f"📁 CSV Mode | {len(df)} records loaded")
    with col2:
        st.info(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col3:
        auto_sync = st.checkbox("Auto-sync", value=False, help="Automatically sync changes to Google Sheets")
    
    st.divider()
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Executive Dashboard",
        "📊 Analytics & Insights",
        "👥 Team Overview",
        "🎯 Performance Tracking",
        "✏️ Data Management",
        "➕ Add New Rep",
        "🔧 Edit Rep",
        "📄 Invoice Management",
        "📑 Reports & Export"
    ])
    
    # TAB 1: Executive Dashboard
    with tab1:
        st.header("Executive Overview - Key Metrics")
        
        # Top KPI Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_reps = len(df)
            active_reps = len(df[df['Employment Status'] == 'Active'])
            st.metric(
                "Total Reps",
                total_reps,
                f"{active_reps} Active",
                delta_color="normal"
            )
        
        with col2:
            total_sales = df['Actual Sales (Month)'].sum()
            target_sales = df['Monthly Sales Target'].sum()
            sales_diff = ((total_sales - target_sales) / target_sales * 100) if target_sales > 0 else 0
            st.metric(
                "Total Sales",
                f"${total_sales:,.0f}",
                f"{sales_diff:+.1f}% vs Target"
            )
        
        with col3:
            avg_goal = df['% to Goal'].mean()
            st.metric(
                "Avg % to Goal",
                f"{avg_goal:.1f}%",
                "Overall Performance"
            )
        
        with col4:
            total_pipeline = df['Pipeline Value'].sum()
            st.metric(
                "Total Pipeline",
                f"${total_pipeline:,.0f}",
                "Active Opportunities"
            )
        
        with col5:
            avg_win_rate = df['Win Rate (%)'].mean()
            st.metric(
                "Avg Win Rate",
                f"{avg_win_rate:.1f}%",
                "Conversion Success"
            )
        
        st.divider()
        
        # Secondary KPIs
        st.subheader("Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_deals = df['Total Deals Closed (Month)'].sum()
            avg_deal_size = df['Average Deal Size'].mean()
            st.metric("Total Deals Closed", f"{int(total_deals)}", f"Avg: ${avg_deal_size:,.0f}")
        
        with col2:
            total_commission = df['Commission Earned (Month)'].sum()
            st.metric("Total Commission", f"${total_commission:,.0f}", "This Month")
        
        with col3:
            avg_activity = df['Daily Activity Score'].mean()
            st.metric("Avg Activity Score", f"{avg_activity:.0f}", "Team Engagement")
        
        with col4:
            avg_csat = df['Customer Satisfaction Score'].mean()
            st.metric("Avg CSAT Score", f"{avg_csat:.2f}/5.0", "Customer Feedback")
        
        st.divider()
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales Performance Comparison")
            fig1 = go.Figure()
            
            fig1.add_trace(go.Bar(
                name='Target',
                x=df['Full Name'].head(15),
                y=df['Monthly Sales Target'].head(15),
                marker_color='#636EFA'
            ))
            
            fig1.add_trace(go.Bar(
                name='Actual',
                x=df['Full Name'].head(15),
                y=df['Actual Sales (Month)'].head(15),
                marker_color='#00CC96'
            ))
            
            fig1.update_layout(
                barmode='group',
                xaxis_tickangle=-45,
                height=400,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Territory Sales Distribution")
            territory_sales = df.groupby('Territory / Region').agg({
                'Actual Sales (Month)': 'sum',
                'Pipeline Value': 'sum'
            }).reset_index()
            
            fig2 = px.pie(
                territory_sales,
                values='Actual Sales (Month)',
                names='Territory / Region',
                title="Sales by Territory",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig2.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Win Rate by Role")
            role_metrics = df.groupby('Role / Position').agg({
                'Win Rate (%)': 'mean',
                'Total Deals Closed (Month)': 'sum',
                'Performance Score': 'mean'
            }).reset_index()
            
            fig3 = px.bar(
                role_metrics,
                x='Role / Position',
                y='Win Rate (%)',
                color='Performance Score',
                title="Average Win Rate & Performance by Position",
                color_continuous_scale='Viridis',
                text='Win Rate (%)'
            )
            fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.subheader("Activity & Engagement")
            
            fig4 = go.Figure()
            
            fig4.add_trace(go.Scatter(
                x=df['Full Name'].head(15),
                y=df['Daily Activity Score'].head(15),
                mode='lines+markers',
                name='Activity Score',
                line=dict(color='#EF553B', width=3),
                marker=dict(size=10)
            ))
            
            fig4.update_layout(
                xaxis_tickangle=-45,
                height=400,
                hovermode='x unified',
                yaxis_title="Activity Score"
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        # Top Performers Section
        st.divider()
        st.subheader("🏆 Top Performers")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Top by Sales**")
            top_sales = df.nlargest(5, 'Actual Sales (Month)')[
                ['Full Name', 'Actual Sales (Month)', '% to Goal']
            ]
            top_sales['Actual Sales (Month)'] = top_sales['Actual Sales (Month)'].apply(lambda x: f"${x:,.0f}")
            top_sales['% to Goal'] = top_sales['% to Goal'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_sales, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**Top by Win Rate**")
            top_win_rate = df.nlargest(5, 'Win Rate (%)')[
                ['Full Name', 'Win Rate (%)', 'Total Deals Closed (Month)']
            ]
            top_win_rate['Win Rate (%)'] = top_win_rate['Win Rate (%)'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_win_rate, hide_index=True, use_container_width=True)
        
        with col3:
            st.markdown("**Top by Performance Score**")
            top_performance = df.nlargest(5, 'Performance Score')[
                ['Full Name', 'Performance Score', 'Daily Activity Score']
            ]
            top_performance['Performance Score'] = top_performance['Performance Score'].apply(lambda x: f"{x:.1f}")
            st.dataframe(top_performance, hide_index=True, use_container_width=True)
        
        # Bottom Performers / Needs Attention
        st.divider()
        st.subheader("⚠️ Needs Attention")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Below Target Performance**")
            below_target = df[df['% to Goal'] < 60].nsmallest(5, '% to Goal')[
                ['Full Name', 'Role / Position', '% to Goal', 'Actual Sales (Month)']
            ]
            if not below_target.empty:
                below_target['% to Goal'] = below_target['% to Goal'].apply(lambda x: f"{x:.1f}%")
                below_target['Actual Sales (Month)'] = below_target['Actual Sales (Month)'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(below_target, hide_index=True, use_container_width=True)
            else:
                st.success("✅ All reps are performing above 60% of target!")
        
        with col2:
            st.markdown("**Overdue Tasks**")
            overdue_tasks = df[df['Tasks Overdue'] > 0].nlargest(5, 'Tasks Overdue')[
                ['Full Name', 'Tasks Overdue', 'Tasks Completed', 'Follow-Ups Due']
            ]
            if not overdue_tasks.empty:
                st.dataframe(overdue_tasks, hide_index=True, use_container_width=True)
            else:
                st.success("✅ No overdue tasks!")
    
    # TAB 2: Analytics & Insights
    with tab2:
        st.header("Advanced Analytics & Insights")
        
        # Analysis Options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Correlation Analysis", "Trend Analysis", "Comparative Analysis", "Predictive Insights"]
            )
        
        with col2:
            metric_focus = st.selectbox(
                "Focus Metric",
                ["Sales Performance", "Activity Metrics", "Win Rates", "Customer Satisfaction"]
            )
        
        with col3:
            group_by = st.selectbox(
                "Group By",
                ["Territory / Region", "Role / Position", "Employment Status", "Manager Name"]
            )
        
        st.divider()
        
        # Correlation Analysis
        if analysis_type == "Correlation Analysis":
            st.subheader("📊 Correlation Matrix")
            
            correlation_columns = [
                'Actual Sales (Month)', 'Win Rate (%)', 'Daily Activity Score',
                'Outbound Calls Made', 'Outbound Emails Sent', 'Meetings Completed',
                'Customer Satisfaction Score', 'Performance Score'
            ]
            
            available_cols = [col for col in correlation_columns if col in df.columns]
            corr_matrix = df[available_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=available_cols,
                y=available_cols,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Correlation between Key Performance Metrics"
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Key Insights
            st.subheader("🔍 Key Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                # Find strongest positive correlation
                mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                masked_corr = corr_matrix.where(mask)
                max_corr = masked_corr.max().max()
                max_corr_idx = np.where(masked_corr == max_corr)
                if len(max_corr_idx[0]) > 0:
                    var1 = available_cols[max_corr_idx[0][0]]
                    var2 = available_cols[max_corr_idx[1][0]]
                    st.success(f"**Strongest Positive Correlation:** {var1} ↔ {var2} ({max_corr:.2f})")
            
            with col2:
                # Find strongest negative correlation
                min_corr = masked_corr.min().min()
                min_corr_idx = np.where(masked_corr == min_corr)
                if len(min_corr_idx[0]) > 0:
                    var1 = available_cols[min_corr_idx[0][0]]
                    var2 = available_cols[min_corr_idx[1][0]]
                    st.warning(f"**Strongest Negative Correlation:** {var1} ↔ {var2} ({min_corr:.2f})")
        
        # Trend Analysis
        elif analysis_type == "Trend Analysis":
            st.subheader("📈 Performance Trends")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Activity vs Performance scatter
                fig_scatter = px.scatter(
                    df,
                    x='Daily Activity Score',
                    y='Actual Sales (Month)',
                    size='Win Rate (%)',
                    color='Performance Score',
                    hover_name='Full Name',
                    title="Activity Score vs Sales Performance",
                    labels={'Daily Activity Score': 'Activity Score', 'Actual Sales (Month)': 'Sales ($)'},
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Pipeline vs Closed Deals
                fig_pipeline = px.scatter(
                    df,
                    x='Pipeline Value',
                    y='Total Deals Closed (Month)',
                    size='Win Rate (%)',
                    color='Role / Position',
                    hover_name='Full Name',
                    title="Pipeline Value vs Deals Closed",
                    labels={'Pipeline Value': 'Pipeline ($)', 'Total Deals Closed (Month)': 'Deals Closed'}
                )
                st.plotly_chart(fig_pipeline, use_container_width=True)
            
            # Meeting Effectiveness
            st.subheader("Meeting Effectiveness Analysis")
            df['Meeting Effectiveness'] = (df['Total Deals Closed (Month)'] / df['Meetings Completed'].replace(0, 1)).round(2)
            
            fig_meetings = px.bar(
                df.head(15),
                x='Full Name',
                y=['Meetings Booked', 'Meetings Completed'],
                title="Meetings: Booked vs Completed",
                barmode='group',
                color_discrete_sequence=['#FFA15A', '#19D3F3']
            )
            fig_meetings.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_meetings, use_container_width=True)
        
        # Comparative Analysis
        elif analysis_type == "Comparative Analysis":
            st.subheader(f"📊 Comparison by {group_by}")
            
            grouped_data = df.groupby(group_by).agg({
                'Actual Sales (Month)': ['sum', 'mean'],
                'Win Rate (%)': 'mean',
                'Daily Activity Score': 'mean',
                'Total Deals Closed (Month)': 'sum',
                'Customer Satisfaction Score': 'mean',
                'Performance Score': 'mean'
            }).round(2)
            
            grouped_data.columns = ['Total Sales', 'Avg Sales', 'Avg Win Rate', 'Avg Activity', 'Total Deals', 'Avg CSAT', 'Avg Performance']
            
            st.dataframe(grouped_data, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig_comp1 = px.bar(
                    grouped_data.reset_index(),
                    x=group_by,
                    y='Total Sales',
                    title=f"Total Sales by {group_by}",
                    color='Avg Performance',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_comp1, use_container_width=True)
            
            with col2:
                fig_comp2 = px.bar(
                    grouped_data.reset_index(),
                    x=group_by,
                    y='Avg Win Rate',
                    title=f"Average Win Rate by {group_by}",
                    color='Avg Activity',
                    color_continuous_scale='Plasma'
                )
                st.plotly_chart(fig_comp2, use_container_width=True)
        
        # Predictive Insights
        else:
            st.subheader("🔮 Predictive Insights & Recommendations")
            
            # Calculate predictions based on current performance
            df['Projected Monthly Sales'] = (df['Actual Sales (Month)'] / datetime.now().day * 30).round(0)
            df['Target Gap'] = df['Monthly Sales Target'] - df['Projected Monthly Sales']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Likely to Exceed Target**")
                likely_exceed = df[df['Projected Monthly Sales'] > df['Monthly Sales Target']].nlargest(10, 'Projected Monthly Sales')[
                    ['Full Name', 'Projected Monthly Sales', 'Monthly Sales Target', '% to Goal']
                ]
                if not likely_exceed.empty:
                    likely_exceed['Projected Monthly Sales'] = likely_exceed['Projected Monthly Sales'].apply(lambda x: f"${x:,.0f}")
                    likely_exceed['Monthly Sales Target'] = likely_exceed['Monthly Sales Target'].apply(lambda x: f"${x:,.0f}")
                    likely_exceed['% to Goal'] = likely_exceed['% to Goal'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(likely_exceed, hide_index=True, use_container_width=True)
                else:
                    st.info("No reps currently projected to exceed target")
            
            with col2:
                st.markdown("**At Risk of Missing Target**")
                at_risk = df[df['Target Gap'] > 0].nlargest(10, 'Target Gap')[
                    ['Full Name', 'Projected Monthly Sales', 'Target Gap', '% to Goal']
                ]
                if not at_risk.empty:
                    at_risk['Projected Monthly Sales'] = at_risk['Projected Monthly Sales'].apply(lambda x: f"${x:,.0f}")
                    at_risk['Target Gap'] = at_risk['Target Gap'].apply(lambda x: f"${x:,.0f}")
                    at_risk['% to Goal'] = at_risk['% to Goal'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(at_risk, hide_index=True, use_container_width=True)
                else:
                    st.success("All reps on track to meet targets!")
            
            # Recommendations
            st.divider()
            st.subheader("💡 AI-Powered Recommendations")
            
            recommendations = []
            
            # Low activity reps
            low_activity = df[df['Daily Activity Score'] < 70]
            if not low_activity.empty:
                recommendations.append({
                    'Priority': '🔴 High',
                    'Category': 'Activity',
                    'Issue': f"{len(low_activity)} reps with low activity scores',
                    'Action': 'Schedule coaching sessions and review daily routines'
                })
            
            # Low win rate
            low_win_rate = df[df['Win Rate (%)'] < 20]
            if not low_win_rate.empty:
                recommendations.append({
                    'Priority': '🟡 Medium',
                    'Category': 'Conversion',
                    'Issue': f"{len(low_win_rate)} reps with win rates below 20%",
                    'Action': 'Provide sales training and review qualification process'
                })
            
            # High pipeline, low closes
            high_pipeline_low_close = df[(df['Pipeline Value'] > df['Pipeline Value'].median()) & 
                                          (df['Total Deals Closed (Month)'] < df['Total Deals Closed (Month)'].median())]
            if not high_pipeline_low_close.empty:
                recommendations.append({
                    'Priority': '🟡 Medium',
                    'Category': 'Pipeline',
                    'Issue': f"{len(high_pipeline_low_close)} reps with high pipeline but low closes",
                    'Action': 'Review deal progression and remove stale opportunities'
                })
            
            # Low CSAT
            low_csat = df[df['Customer Satisfaction Score'] < 3.5]
            if not low_csat.empty:
                recommendations.append({
                    'Priority': '🔴 High',
                    'Category': 'Customer Success',
                    'Issue': f"{len(low_csat)} reps with CSAT below 3.5",
                    'Action': 'Customer service training and follow-up process review'
                })
            
            if recommendations:
                rec_df = pd.DataFrame(recommendations)
                st.dataframe(rec_df, hide_index=True, use_container_width=True)
            else:
                st.success("✅ No critical issues detected. Team is performing well!")
    
    # TAB 3: Team Overview
    with tab3:
        st.header("👥 Team Overview & Structure")
        
        # Team Summary Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            active_count = len(df[df['Employment Status'] == 'Active'])
            inactive_count = len(df[df['Employment Status'] == 'Inactive'])
            st.metric("Active Reps", active_count, f"{inactive_count} Inactive")
        
        with col2:
            full_time = len(df[df['Contract Type'] == 'Full-Time'])
            st.metric("Full-Time", full_time, f"{len(df) - full_time} Part-Time/Contract")
        
        with col3:
            trained = len(df[df['Training Completed'] == 'Yes'])
            st.metric("Trained Reps", trained, f"{len(df) - trained} Pending")
        
        with col4:
            avg_tenure_days = (datetime.now() - pd.to_datetime(df['Start Date'], errors='coerce')).dt.days.mean()
            st.metric("Avg Tenure", f"{int(avg_tenure_days)} days", "Team Experience")
        
        st.divider()
        
        # Team Distribution
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Distribution by Role")
            role_dist = df['Role / Position'].value_counts().reset_index()
            role_dist.columns = ['Role', 'Count']
            
            fig_role = px.pie(
                role_dist,
                values='Count',
                names='Role',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_role.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_role, use_container_width=True)
        
        with col2:
            st.subheader("Distribution by Territory")
            territory_dist = df['Territory / Region'].value_counts().reset_index()
            territory_dist.columns = ['Territory', 'Count']
            
            fig_territory = px.pie(
                territory_dist,
                values='Count',
                names='Territory',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_territory.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_territory, use_container_width=True)
        
        with col3:
            st.subheader("Employment Status")
            status_dist = df['Employment Status'].value_counts().reset_index()
            status_dist.columns = ['Status', 'Count']
            
            fig_status = px.bar(
                status_dist,
                x='Status',
                y='Count',
                color='Status',
                color_discrete_map={'Active': '#00CC96', 'Inactive': '#EF553B'}
            )
            fig_status.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_status, use_container_width=True)
        
        # Detailed Team Roster
        st.divider()
        st.subheader("📋 Team Roster with Performance Scores")
        
        # Add performance indicators
        roster_df = df[[
            'Full Name', 'Role / Position', 'Territory / Region', 'Employment Status',
            'Actual Sales (Month)', '% to Goal', 'Win Rate (%)', 'Performance Score',
            'Customer Satisfaction Score', 'Manager Name'
        ]].copy()
        
        roster_df['Performance Level'] = roster_df['Performance Score'].apply(
            lambda x: get_performance_level(x)[0]
        )
        roster_df['Status Icon'] = roster_df['Performance Score'].apply(
            lambda x: get_performance_level(x)[1]
        )
        
        # Format columns
        roster_df['Actual Sales (Month)'] = roster_df['Actual Sales (Month)'].apply(lambda x: f"${x:,.0f}")
        roster_df['% to Goal'] = roster_df['% to Goal'].apply(lambda x: f"{x:.1f}%")
        roster_df['Win Rate (%)'] = roster_df['Win Rate (%)'].apply(lambda x: f"{x:.1f}%")
        roster_df['Performance Score'] = roster_df['Performance Score'].apply(lambda x: f"{x:.1f}")
        roster_df['Customer Satisfaction Score'] = roster_df['Customer Satisfaction Score'].apply(lambda x: f"{x:.1f}/5.0")
        
        # Display with color coding
        st.dataframe(
            roster_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Status Icon': st.column_config.TextColumn('Status'),
                'Performance Level': st.column_config.TextColumn('Level'),
            }
        )
    
    # TAB 4: Performance Tracking
    with tab4:
        st.header("🎯 Detailed Performance Tracking")
        
        # Performance Filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            perf_status = st.multiselect(
                "Employment Status",
                options=df['Employment Status'].unique(),
                default=df['Employment Status'].unique()
            )
        
        with col2:
            perf_role = st.multiselect(
                "Role",
                options=df['Role / Position'].unique(),
                default=df['Role / Position'].unique()
            )
        
        with col3:
            perf_territory = st.multiselect(
                "Territory",
                options=df['Territory / Region'].unique(),
                default=df['Territory / Region'].unique()
            )
        
        with col4:
            perf_threshold = st.slider(
                "Min Performance Score",
                0, 100, 0,
                help="Filter reps by minimum performance score"
            )
        
        # Apply filters
        perf_df = df[
            (df['Employment Status'].isin(perf_status)) &
            (df['Role / Position'].isin(perf_role)) &
            (df['Territory / Region'].isin(perf_territory)) &
            (df['Performance Score'] >= perf_threshold)
        ]
        
        st.info(f"Showing {len(perf_df)} of {len(df)} reps")
        
        st.divider()
        
        # Performance Metrics Grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales Performance Metrics")
            
            sales_metrics = perf_df[[
                'Full Name', 'Monthly Sales Target', 'Actual Sales (Month)',
                '% to Goal', 'Total Deals Closed (Month)', 'Average Deal Size', 'Win Rate (%)'
            ]].copy()
            
            # Format for display
            sales_metrics['Monthly Sales Target'] = sales_metrics['Monthly Sales Target'].apply(lambda x: f"${x:,.0f}")
            sales_metrics['Actual Sales (Month)'] = sales_metrics['Actual Sales (Month)'].apply(lambda x: f"${x:,.0f}")
            sales_metrics['% to Goal'] = sales_metrics['% to Goal'].apply(lambda x: f"{x:.1f}%")
            sales_metrics['Average Deal Size'] = sales_metrics['Average Deal Size'].apply(lambda x: f"${x:,.0f}")
            sales_metrics['Win Rate (%)'] = sales_metrics['Win Rate (%)'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(sales_metrics, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("Activity Performance Metrics")
            
            activity_metrics = perf_df[[
                'Full Name', 'Outbound Calls Made', 'Outbound Emails Sent',
                'Meetings Booked', 'Meetings Completed', 'Daily Activity Score'
            ]].copy()
            
            st.dataframe(activity_metrics, hide_index=True, use_container_width=True)
        
        st.divider()
        
        # Lead & Opportunity Metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Lead Management")
            
            lead_metrics = perf_df[[
                'Full Name', 'New Leads Assigned', 'Leads Contacted',
                'Lead Contact Rate', 'Follow-Ups Due'
            ]].copy()
            
            lead_metrics['Lead Contact Rate'] = lead_metrics['Lead Contact Rate'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(lead_metrics, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("Opportunity Pipeline")
            
            opp_metrics = perf_df[[
                'Full Name', 'Opportunities Created', 'Opportunities in Progress',
                'Opportunities Won', 'Opportunities Lost', 'Conversion Rate'
            ]].copy()
            
            opp_metrics['Conversion Rate'] = opp_metrics['Conversion Rate'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(opp_metrics, hide_index=True, use_container_width=True)
        
        # Individual Performance Deep Dive
        st.divider()
        st.subheader("🔍 Individual Performance Deep Dive")
        
        selected_rep_perf = st.selectbox(
            "Select Rep for Detailed Analysis",
            perf_df['Full Name'].tolist(),
            key="perf_deep_dive"
        )
        
        if selected_rep_perf:
            rep_detail = perf_df[perf_df['Full Name'] == selected_rep_perf].iloc[0]
            
            # Rep header
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Performance Score", f"{rep_detail['Performance Score']:.1f}")
            
            with col2:
                perf_level, perf_icon = get_performance_level(rep_detail['Performance Score'])
                st.metric("Performance Level", f"{perf_icon} {perf_level}")
            
            with col3:
                st.metric("% to Goal", f"{rep_detail['% to Goal']:.1f}%")
            
            with col4:
                st.metric("CSAT Score", f"{rep_detail['Customer Satisfaction Score']:.2f}/5.0")
            
            # Detailed metrics tabs
            detail_tab1, detail_tab2, detail_tab3 = st.tabs(["Sales Breakdown", "Activity Analysis", "Performance Radar"])
            
            with detail_tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Sales Metrics**")
                    sales_data = {
                        'Metric': ['Monthly Target', 'Actual Sales', 'Gap to Target', 'Deals Closed', 'Avg Deal Size', 'Pipeline Value'],
                        'Value': [
                            f"${rep_detail['Monthly Sales Target']:,.0f}",
                            f"${rep_detail['Actual Sales (Month)']:,.0f}",
                            f"${rep_detail['Monthly Sales Target'] - rep_detail['Actual Sales (Month)']:,.0f}",
                            int(rep_detail['Total Deals Closed (Month)']),
                            f"${rep_detail['Average Deal Size']:,.0f}",
                            f"${rep_detail['Pipeline Value']:,.0f}"
                        ]
                    }
                    st.dataframe(pd.DataFrame(sales_data), hide_index=True, use_container_width=True)
                
                with col2:
                    # Funnel chart
                    funnel_data = pd.DataFrame({
                        'Stage': ['Leads Assigned', 'Leads Contacted', 'Opportunities', 'Opportunities Won'],
                        'Count': [
                            rep_detail['New Leads Assigned'],
                            rep_detail['Leads Contacted'],
                            rep_detail['Opportunities Created'],
                            rep_detail['Opportunities Won']
                        ]
                    })
                    
                    fig_funnel = px.funnel(
                        funnel_data,
                        x='Count',
                        y='Stage',
                        title="Sales Funnel"
                    )
                    st.plotly_chart(fig_funnel, use_container_width=True)
            
            with detail_tab2:
                # Activity breakdown
                activity_data = pd.DataFrame({
                    'Activity': ['Calls Made', 'Emails Sent', 'Meetings Booked', 'Meetings Completed', 'Notes Logged', 'Tasks Completed'],
                    'Count': [
                        rep_detail['Outbound Calls Made'],
                        rep_detail['Outbound Emails Sent'],
                        rep_detail['Meetings Booked'],
                        rep_detail['Meetings Completed'],
                        rep_detail['Notes Logged (Count)'],
                        rep_detail['Tasks Completed']
                    ]
                })
                
                fig_activity = px.bar(
                    activity_data,
                    x='Activity',
                    y='Count',
                    title="Activity Breakdown",
                    color='Count',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_activity, use_container_width=True)
                
                # Task status
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tasks Completed", int(rep_detail['Tasks Completed']))
                with col2:
                    st.metric("Tasks Overdue", int(rep_detail['Tasks Overdue']), delta=f"-{int(rep_detail['Tasks Overdue'])}" if rep_detail['Tasks Overdue'] > 0 else "0")
            
            with detail_tab3:
                # Radar chart for performance dimensions
                categories = ['Sales %', 'Win Rate', 'Activity', 'CSAT', 'Conversion']
                values = [
                    min(rep_detail['% to Goal'], 100),
                    rep_detail['Win Rate (%)'],
                    rep_detail['Daily Activity Score'],
                    (rep_detail['Customer Satisfaction Score'] / 5) * 100,
                    rep_detail['Conversion Rate']
                ]
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=selected_rep_perf,
                    line_color='#636EFA'
                ))
                
                # Add team average
                team_avg_values = [
                    min(df['% to Goal'].mean(), 100),
                    df['Win Rate (%)'].mean(),
                    df['Daily Activity Score'].mean(),
                    (df['Customer Satisfaction Score'].mean() / 5) * 100,
                    df['Conversion Rate'].mean()
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=team_avg_values,
                    theta=categories,
                    fill='toself',
                    name='Team Average',
                    line_color='#EF553B',
                    line_dash='dash'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=True,
                    title="Performance vs Team Average"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
    
    # TAB 5: Data Management
    with tab5:
        st.header("✏️ Data Management & Live Editing")
        
        # Filter options
        st.subheader("🔍 Filter & Search")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_filter = st.multiselect(
                "Employment Status",
                options=df['Employment Status'].unique(),
                default=df['Employment Status'].unique(),
                key="data_status_filter"
            )
        
        with col2:
            role_filter = st.multiselect(
                "Role / Position",
                options=df['Role / Position'].unique(),
                default=df['Role / Position'].unique(),
                key="data_role_filter"
            )
        
        with col3:
            territory_filter = st.multiselect(
                "Territory",
                options=df['Territory / Region'].unique(),
                default=df['Territory / Region'].unique(),
                key="data_territory_filter"
            )
        
        with col4:
            search_term = st.text_input("🔎 Search by Name or Email", "")
        
        # Apply filters
        filtered_df = df[
            (df['Employment Status'].isin(status_filter)) &
            (df['Role / Position'].isin(role_filter)) &
            (df['Territory / Region'].isin(territory_filter))
        ]
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df['Full Name'].str.contains(search_term, case=False, na=False) |
                filtered_df['Email'].str.contains(search_term, case=False, na=False)
            ]
        
        st.info(f"📊 Showing {len(filtered_df)} of {len(df)} records")
        
        st.divider()
        
        # Column selector
        st.subheader("📋 Select Columns to Display")
        
        all_columns = df.columns.tolist()
        default_columns = [
            'Rep ID', 'Full Name', 'Email', 'Role / Position', 'Territory / Region',
            'Employment Status', 'Monthly Sales Target', 'Actual Sales (Month)',
            '% to Goal', 'Win Rate (%)', 'Daily Activity Score'
        ]
        
        selected_columns = st.multiselect(
            "Choose columns to display (scroll for more)",
            options=all_columns,
            default=[col for col in default_columns if col in all_columns]
        )
        
        if not selected_columns:
            selected_columns = default_columns
        
        st.divider()
        
        # Editable dataframe
        st.subheader("✏️ Edit Data (All changes are live)")
        
        st.warning("⚠️ Any changes made will update the data. Click 'Save Changes' to sync with Google Sheets.")
        
        edited_df = st.data_editor(
            filtered_df[selected_columns],
            use_container_width=True,
            num_rows="dynamic",
            hide_index=False,
            key="main_data_editor_advanced",
            column_config={
                'Email': st.column_config.TextColumn('Email', help="Email address", max_chars=100),
                'Phone': st.column_config.TextColumn('Phone', help="Phone number"),
                'Employment Status': st.column_config.SelectboxColumn(
                    'Status',
                    options=['Active', 'Inactive'],
                    required=True
                ),
            }
        )
        
        # Save button row
        col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
        
        with col1:
            if st.button("💾 Save Changes", type="primary", use_container_width=True):
                # Update main dataframe
                for col in selected_columns:
                    df.loc[filtered_df.index, col] = edited_df[col].values
                
                st.session_state.df = df
                
                # Sync to Google Sheets
                if auto_sync or st.session_state.data_source == "gsheet":
                    success, message = sync_to_google_sheets(df)
                    if success:
                        st.success("✅ " + message)
                    else:
                        st.warning("⚠️ " + message)
                else:
                    st.success("✅ Changes saved locally!")
                
                st.rerun()
        
        with col2:
            if st.button("🔄 Revert Changes", use_container_width=True):
                st.rerun()
        
        with col3:
            # Bulk update
            if st.button("⚙️ Bulk Update", use_container_width=True):
                st.info("Bulk update feature - Coming soon!")
        
        with col4:
            # Download filtered data
            csv_filtered = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv_filtered,
                file_name=f"filtered_sales_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Bulk operations
        st.divider()
        st.subheader("⚙️ Bulk Operations")
        
        with st.expander("Bulk Update Selected Reps"):
            bulk_col1, bulk_col2 = st.columns(2)
            
            with bulk_col1:
                bulk_field = st.selectbox(
                    "Select Field to Update",
                    ['Employment Status', 'Manager Name', 'Territory / Region', 
                     'Training Completed', 'Contract Type']
                )
            
            with bulk_col2:
                if bulk_field == 'Employment Status':
                    bulk_value = st.selectbox("New Value", ['Active', 'Inactive'])
                elif bulk_field == 'Training Completed':
                    bulk_value = st.selectbox("New Value", ['Yes', 'No'])
                elif bulk_field == 'Contract Type':
                    bulk_value = st.selectbox("New Value", ['Full-Time', 'Part-Time', 'Contract'])
                elif bulk_field == 'Territory / Region':
                    bulk_value = st.selectbox("New Value", df['Territory / Region'].unique())
                else:
                    bulk_value = st.text_input("New Value")
            
            selected_reps_bulk = st.multiselect(
                "Select Reps to Update",
                filtered_df['Full Name'].tolist()
            )
            
            if st.button("Apply Bulk Update"):
                if selected_reps_bulk and bulk_value:
                    df.loc[df['Full Name'].isin(selected_reps_bulk), bulk_field] = bulk_value
                    st.session_state.df = df
                    
                    if auto_sync:
                        sync_to_google_sheets(df)
                    
                    st.success(f"✅ Updated {bulk_field} for {len(selected_reps_bulk)} reps!")
                    st.rerun()
                else:
                    st.error("Please select reps and provide a value")
    
    # TAB 6: Add New Rep
    with tab6:
        st.header("➕ Add New Sales Rep")
        
        with st.form("add_rep_form_advanced"):
            st.subheader("Basic Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rep_id = st.text_input("Rep ID*", value=str(int(df['Rep ID'].max()) + 1 if len(df) > 0 else 1))
                full_name = st.text_input("Full Name*")
                email = st.text_input("Email*")
                phone = st.text_input("Phone*")
            
            with col2:
                role = st.selectbox("Role / Position*", 
                    ["Field Rep", "BDR", "SDR", "Account Executive", "Sales Manager"])
                territory = st.selectbox("Territory / Region*",
                    ["National", "South", "North", "West Coast", "Midwest", "East Coast"])
                employment_status = st.selectbox("Employment Status*", ["Active", "Inactive"])
                contract_type = st.selectbox("Contract Type*", 
                    ["Full-Time", "Part-Time", "Contract"])
            
            with col3:
                start_date = st.date_input("Start Date*")
                manager_name = st.text_input("Manager Name", value="Chris Johnson")
                training_completed = st.selectbox("Training Completed", ["Yes", "No"])
                certifications = st.text_input("Certifications", value="Salesforce, HubSpot")
            
            st.divider()
            st.subheader("Sales Targets & Compensation")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                monthly_target = st.number_input("Monthly Sales Target*", min_value=0, value=50000, step=1000)
            
            with col2:
                base_salary = st.number_input("Base Salary*", min_value=0, value=60000, step=1000)
            
            with col3:
                commission_rate = st.text_input("Commission Rate*", value="10%")
            
            with col4:
                customer_sat_score = st.number_input("Customer Satisfaction Score", 
                    min_value=0.0, max_value=5.0, value=4.0, step=0.1)
            
            st.divider()
            st.subheader("Additional Notes")
            
            hr_notes = st.text_area("HR Notes", placeholder="Enter any relevant notes...")
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                submitted = st.form_submit_button("✅ Add Rep", type="primary", use_container_width=True)
            
            if submitted:
                if full_name and email and phone:
                    # Create new row
                    new_row = {col: "" for col in df.columns}
                    
                    new_row.update({
                        'Rep ID': rep_id,
                        'Full Name': full_name,
                        'Email': email,
                        'Phone': phone,
                        'Role / Position': role,
                        'Territory / Region': territory,
                        'Start Date': start_date.strftime('%Y-%m-%d'),
                        'Employment Status': employment_status,
                        'Contract Type': contract_type,
                        'Monthly Sales Target': monthly_target,
                        'Actual Sales (Month)': 0,
                        '% to Goal': 0,
                        'Total Deals Closed (Month)': 0,
                        'Average Deal Size': 0,
                        'Pipeline Value': 0,
                        'Win Rate (%)': 0,
                        'Outbound Calls Made': 0,
                        'Outbound Emails Sent': 0,
                        'Meetings Booked': 0,
                        'Meetings Completed': 0,
                        'New Leads Assigned': 0,
                        'Leads Contacted': 0,
                        'Follow-Ups Due': 0,
                        'Opportunities Created': 0,
                        'Opportunities in Progress': 0,
                        'Opportunities Lost': 0,
                        'Opportunities Won': 0,
                        'Base Salary': base_salary,
                        'Commission Rate': commission_rate,
                        'Commission Earned (Month)': 0,
                        'Bonuses Earned': 0,
                        'Total Compensation to Date': base_salary,
                        'Daily Activity Score': 0,
                        'Last Login (CRM)': datetime.now().strftime('%Y-%m-%d'),
                        'Notes Logged (Count)': 0,
                        'Tasks Completed': 0,
                        'Tasks Overdue': 0,
                        'Lead Response Time (Avg)': "0 hours",
                        'Customer Satisfaction Score': customer_sat_score,
                        'Manager Name': manager_name,
                        'Training Completed': training_completed,
                        'Certifications': certifications,
                        'Performance Review Date': "",
                        'HR Notes': hr_notes
                    })
                    
                    # Add invoices placeholders
                    for i in range(1, 11):
                        new_row[f'Invoice {i} Number'] = f'INV-{rep_id}{i:03d}'
                        new_row[f'Invoice {i} Date'] = ''
                        new_row[f'Invoice {i} Amount'] = 0
                    
                    st.session_state.df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    if auto_sync:
                        success, message = sync_to_google_sheets(st.session_state.df)
                        if success:
                            st.success(f"✅ Rep '{full_name}' added successfully and synced to Google Sheets!")
                        else:
                            st.warning(f"✅ Rep '{full_name}' added locally. {message}")
                    else:
                        st.success(f"✅ Rep '{full_name}' added successfully!")
                    
                    st.rerun()
                else:
                    st.error("❌ Please fill in all required fields (marked with *)")
    
    # TAB 7: Edit Rep
    with tab7:
        st.header("🔧 Edit Sales Rep")
        
        # Rep selection
        col1, col2 = st.columns([3, 1])
        
        with col1:
            rep_names = df['Full Name'].tolist()
            selected_rep = st.selectbox("Select Rep to Edit", rep_names, key="edit_rep_selector")
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            quick_edit_mode = st.checkbox("Quick Edit Mode", help="Edit only key fields")
        
        if selected_rep:
            rep_data = df[df['Full Name'] == selected_rep].iloc[0]
            rep_index = df[df['Full Name'] == selected_rep].index[0]
            
            # Display current performance
            st.divider()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Sales", f"${rep_data['Actual Sales (Month)']:,.0f}")
            with col2:
                st.metric("% to Goal", f"{rep_data['% to Goal']:.1f}%")
            with col3:
                st.metric("Win Rate", f"{rep_data['Win Rate (%)']:.1f}%")
            with col4:
                st.metric("Activity Score", f"{rep_data['Daily Activity Score']:.0f}")
            
            st.divider()
            
            if quick_edit_mode:
                # Quick edit form - only essential fields
                with st.form("quick_edit_form"):
                    st.subheader("Quick Edit - Key Fields")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        actual_sales = st.number_input("Actual Sales", value=float(rep_data['Actual Sales (Month)']))
                        deals_closed = st.number_input("Deals Closed", value=int(rep_data['Total Deals Closed (Month)']))
                    
                    with col2:
                        win_rate = st.number_input("Win Rate (%)", value=float(rep_data['Win Rate (%)']))
                        activity_score = st.number_input("Activity Score", value=int(rep_data['Daily Activity Score']))
                    
                    with col3:
                        employment_status = st.selectbox("Status", ['Active', 'Inactive'], 
                            index=0 if rep_data['Employment Status'] == 'Active' else 1)
                        csat = st.number_input("CSAT Score", value=float(rep_data['Customer Satisfaction Score']), 
                            min_value=0.0, max_value=5.0, step=0.1)
                    
                    if st.form_submit_button("💾 Save Quick Edit", type="primary", use_container_width=True):
                        df.at[rep_index, 'Actual Sales (Month)'] = actual_sales
                        df.at[rep_index, 'Total Deals Closed (Month)'] = deals_closed
                        df.at[rep_index, 'Win Rate (%)'] = win_rate
                        df.at[rep_index, 'Daily Activity Score'] = activity_score
                        df.at[rep_index, 'Employment Status'] = employment_status
                        df.at[rep_index, 'Customer Satisfaction Score'] = csat
                        
                        # Recalculate % to Goal
                        if df.at[rep_index, 'Monthly Sales Target'] > 0:
                            df.at[rep_index, '% to Goal'] = (actual_sales / df.at[rep_index, 'Monthly Sales Target']) * 100
                        
                        st.session_state.df = df
                        
                        if auto_sync:
                            sync_to_google_sheets(df)
                        
                        st.success("✅ Quick edit saved!")
                        st.rerun()
            
            else:
                # Full edit form
                with st.form("full_edit_form"):
                    st.subheader(f"Editing: {selected_rep}")
                    
                    edit_tabs = st.tabs([
                        "Personal Info", "Sales Metrics", "Activity Metrics", 
                        "Lead & Opportunities", "Compensation", "Other"
                    ])
                    
                    updated_data = {}
                    
                    with edit_tabs[0]:  # Personal Info
                        col1, col2 = st.columns(2)
                        with col1:
                            updated_data['Full Name'] = st.text_input("Full Name", value=rep_data['Full Name'])
                            updated_data['Email'] = st.text_input("Email", value=rep_data['Email'])
                            updated_data['Phone'] = st.text_input("Phone", value=rep_data['Phone'])
                            updated_data['Role / Position'] = st.selectbox("Role", 
                                ["Field Rep", "BDR", "SDR", "Account Executive", "Sales Manager"],
                                index=["Field Rep", "BDR", "SDR", "Account Executive", "Sales Manager"].index(rep_data['Role / Position']) if rep_data['Role / Position'] in ["Field Rep", "BDR", "SDR", "Account Executive", "Sales Manager"] else 0)
                        with col2:
                            updated_data['Territory / Region'] = st.text_input("Territory", value=rep_data['Territory / Region'])
                            updated_data['Employment Status'] = st.selectbox("Status", 
                                ["Active", "Inactive"],
                                index=0 if rep_data['Employment Status'] == 'Active' else 1)
                            updated_data['Contract Type'] = st.text_input("Contract Type", value=rep_data['Contract Type'])
                            updated_data['Manager Name'] = st.text_input("Manager", value=rep_data['Manager Name'])
                    
                    with edit_tabs[1]:  # Sales Metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            updated_data['Monthly Sales Target'] = st.number_input("Monthly Target", 
                                value=float(rep_data['Monthly Sales Target']))
                            updated_data['Actual Sales (Month)'] = st.number_input("Actual Sales", 
                                value=float(rep_data['Actual Sales (Month)']))
                            updated_data['Total Deals Closed (Month)'] = st.number_input("Deals Closed", 
                                value=int(rep_data['Total Deals Closed (Month)']))
                            updated_data['Average Deal Size'] = st.number_input("Avg Deal Size", 
                                value=float(rep_data['Average Deal Size']))
                        with col2:
                            updated_data['Pipeline Value'] = st.number_input("Pipeline Value", 
                                value=float(rep_data['Pipeline Value']))
                            updated_data['Win Rate (%)'] = st.number_input("Win Rate %", 
                                value=float(rep_data['Win Rate (%)']))
                            # Auto-calculate % to Goal
                            if updated_data['Monthly Sales Target'] > 0:
                                updated_data['% to Goal'] = (updated_data['Actual Sales (Month)'] / updated_data['Monthly Sales Target']) * 100
                            else:
                                updated_data['% to Goal'] = 0
                            st.metric("Calculated % to Goal", f"{updated_data['% to Goal']:.1f}%")
                    
                    with edit_tabs[2]:  # Activity Metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            updated_data['Outbound Calls Made'] = st.number_input("Calls Made", 
                                value=int(rep_data['Outbound Calls Made']))
                            updated_data['Outbound Emails Sent'] = st.number_input("Emails Sent", 
                                value=int(rep_data['Outbound Emails Sent']))
                            updated_data['Meetings Booked'] = st.number_input("Meetings Booked", 
                                value=int(rep_data['Meetings Booked']))
                            updated_data['Meetings Completed'] = st.number_input("Meetings Completed", 
                                value=int(rep_data['Meetings Completed']))
                        with col2:
                            updated_data['Daily Activity Score'] = st.number_input("Activity Score", 
                                value=int(rep_data['Daily Activity Score']))
                            updated_data['Notes Logged (Count)'] = st.number_input("Notes Logged", 
                                value=int(rep_data['Notes Logged (Count)']))
                            updated_data['Tasks Completed'] = st.number_input("Tasks Completed", 
                                value=int(rep_data['Tasks Completed']))
                            updated_data['Tasks Overdue'] = st.number_input("Tasks Overdue", 
                                value=int(rep_data['Tasks Overdue']))
                    
                    with edit_tabs[3]:  # Lead & Opportunities
                        col1, col2 = st.columns(2)
                        with col1:
                            updated_data['New Leads Assigned'] = st.number_input("Leads Assigned", 
                                value=int(rep_data['New Leads Assigned']))
                            updated_data['Leads Contacted'] = st.number_input("Leads Contacted", 
                                value=int(rep_data['Leads Contacted']))
                            updated_data['Follow-Ups Due'] = st.number_input("Follow-Ups Due", 
                                value=int(rep_data['Follow-Ups Due']))
                            updated_data['Opportunities Created'] = st.number_input("Opps Created", 
                                value=int(rep_data['Opportunities Created']))
                        with col2:
                            updated_data['Opportunities in Progress'] = st.number_input("Opps In Progress", 
                                value=int(rep_data['Opportunities in Progress']))
                            updated_data['Opportunities Won'] = st.number_input("Opps Won", 
                                value=int(rep_data['Opportunities Won']))
                            updated_data['Opportunities Lost'] = st.number_input("Opps Lost", 
                                value=int(rep_data['Opportunities Lost']))
                    
                    with edit_tabs[4]:  # Compensation
                        col1, col2 = st.columns(2)
                        with col1:
                            updated_data['Base Salary'] = st.number_input("Base Salary", 
                                value=float(rep_data['Base Salary']))
                            updated_data['Commission Rate'] = st.text_input("Commission Rate", 
                                value=str(rep_data['Commission Rate']))
                            updated_data['Commission Earned (Month)'] = st.number_input("Commission Earned", 
                                value=float(rep_data['Commission Earned (Month)']))
                        with col2:
                            updated_data['Bonuses Earned'] = st.number_input("Bonuses", 
                                value=float(rep_data['Bonuses Earned']))
                            updated_data['Total Compensation to Date'] = st.number_input("Total Compensation", 
                                value=float(rep_data['Total Compensation to Date']))
                    
                    with edit_tabs[5]:  # Other
                        col1, col2 = st.columns(2)
                        with col1:
                            updated_data['Customer Satisfaction Score'] = st.number_input("CSAT Score", 
                                value=float(rep_data['Customer Satisfaction Score']),
                                min_value=0.0, max_value=5.0, step=0.1)
                            updated_data['Training Completed'] = st.selectbox("Training", ['Yes', 'No'],
                                index=0 if rep_data['Training Completed'] == 'Yes' else 1)
                            updated_data['Certifications'] = st.text_input("Certifications", 
                                value=rep_data['Certifications'])
                        with col2:
                            updated_data['Lead Response Time (Avg)'] = st.text_input("Avg Response Time", 
                                value=rep_data['Lead Response Time (Avg)'])
                            updated_data['HR Notes'] = st.text_area("HR Notes", 
                                value=rep_data.get('HR Notes', ''))
                    
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        if st.form_submit_button("💾 Save All Changes", type="primary", use_container_width=True):
                            for key, value in updated_data.items():
                                df.at[rep_index, key] = value
                            
                            st.session_state.df = df
                            
                            if auto_sync:
                                success, message = sync_to_google_sheets(df)
                                st.success(f"✅ All changes saved! {message}")
                            else:
                                st.success("✅ All changes saved locally!")
                            
                            st.rerun()
    
    # TAB 8: Invoice Management
    with tab8:
        st.header("📄 Invoice Management System")
        
        # Rep selection
        rep_for_invoice = st.selectbox("Select Sales Rep", df['Full Name'].tolist(), key="invoice_rep_select")
        
        if rep_for_invoice:
            rep_data = df[df['Full Name'] == rep_for_invoice].iloc[0]
            rep_index = df[df['Full Name'] == rep_for_invoice].index[0]
            
            # Rep info banner
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.info(f"**Rep ID:** {rep_data['Rep ID']}")
            with col2:
                st.info(f"**Role:** {rep_data['Role / Position']}")
            with col3:
                st.info(f"**Territory:** {rep_data['Territory / Region']}")
            with col4:
                st.info(f"**Status:** {rep_data['Employment Status']}")
            
            st.divider()
            
            # Extract invoice data
            invoices_list = []
            for i in range(1, 11):
                inv_num_col = f'Invoice {i} Number'
                inv_date_col = f'Invoice {i} Date'
                inv_amt_col = f'Invoice {i} Amount'
                
                if all(col in df.columns for col in [inv_num_col, inv_date_col, inv_amt_col]):
                    invoices_list.append({
                        'ID': i,
                        'Invoice Number': rep_data[inv_num_col],
                        'Date': rep_data[inv_date_col],
                        'Amount': rep_data[inv_amt_col]
                    })
            
            if invoices_list:
                invoice_df = pd.DataFrame(invoices_list)
                
                # Summary metrics
                st.subheader("📊 Invoice Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                try:
                    invoice_df['Amount'] = pd.to_numeric(invoice_df['Amount'], errors='coerce').fillna(0)
                    total_amount = invoice_df['Amount'].sum()
                    avg_amount = invoice_df['Amount'].mean()
                    max_amount = invoice_df['Amount'].max()
                    paid_count = len(invoice_df[invoice_df['Date'] != ''])
                    
                    with col1:
                        st.metric("Total Invoice Amount", f"${total_amount:,.2f}")
                    with col2:
                        st.metric("Average Invoice", f"${avg_amount:,.2f}")
                    with col3:
                        st.metric("Highest Invoice", f"${max_amount:,.2f}")
                    with col4:
                        st.metric("Invoices with Dates", paid_count)
                except Exception as e:
                    st.warning("Unable to calculate invoice metrics. Please check data format.")
                
                st.divider()
                
                # Chart
                if invoice_df['Amount'].sum() > 0:
                    fig_inv = px.bar(
                        invoice_df,
                        x='Invoice Number',
                        y='Amount',
                        title="Invoice Amounts",
                        color='Amount',
                        color_continuous_scale='Blues',
                        text='Amount'
                    )
                    fig_inv.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                    fig_inv.update_layout(height=400)
                    st.plotly_chart(fig_inv, use_container_width=True)
                
                st.divider()
                
                # Editable invoice table
                st.subheader("✏️ Edit Invoices")
                
                edited_invoices = st.data_editor(
                    invoice_df,
                    use_container_width=True,
                    num_rows="fixed",
                    hide_index=True,
                    key="invoice_editor_advanced",
                    column_config={
                        'ID': st.column_config.NumberColumn('ID', disabled=True),
                        'Invoice Number': st.column_config.TextColumn('Invoice #', required=True),
                        'Date': st.column_config.TextColumn('Date (YYYY-MM-DD)'),
                        'Amount': st.column_config.NumberColumn('Amount ($)', format="%.2f")
                    }
                )
                
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    if st.button("💾 Save Invoice Changes", type="primary", use_container_width=True):
                        # Update main dataframe
                        for idx, row in edited_invoices.iterrows():
                            invoice_num = int(row['ID'])
                            df.at[rep_index, f'Invoice {invoice_num} Number'] = row['Invoice Number']
                            df.at[rep_index, f'Invoice {invoice_num} Date'] = row['Date']
                            df.at[rep_index, f'Invoice {invoice_num} Amount'] = row['Amount']
                        
                        st.session_state.df = df
                        
                        if auto_sync:
                            success, message = sync_to_google_sheets(df)
                            st.success(f"✅ Invoice changes saved! {message}")
                        else:
                            st.success("✅ Invoice changes saved locally!")
                        
                        st.rerun()
                
                # Download invoice data
                with col2:
                    invoice_csv = edited_invoices.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Invoice Data",
                        data=invoice_csv,
                        file_name=f"invoices_{rep_for_invoice.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    
    # TAB 9: Reports & Export
    with tab9:
        st.header("📑 Reports & Data Export")
        
        st.subheader("📊 Generate Custom Reports")
        
        # Report type selection
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Full Team Report", "Performance Report", "Sales Report", 
                 "Activity Report", "Compensation Report", "Custom Report"]
            )
        
        with col2:
            report_format = st.selectbox(
                "Format",
                ["CSV", "Excel (XLSX)", "JSON"]
            )
        
        st.divider()
        
        # Report configuration
        if report_type == "Full Team Report":
            st.info("📋 Export all data for all sales reps")
            export_df = df
            filename_base = "full_team_report"
        
        elif report_type == "Performance Report":
            st.info("📈 Export performance metrics and scores")
            perf_columns = [
                'Rep ID', 'Full Name', 'Role / Position', 'Territory / Region',
                'Employment Status', 'Monthly Sales Target', 'Actual Sales (Month)',
                '% to Goal', 'Win Rate (%)', 'Performance Score', 'Daily Activity Score',
                'Customer Satisfaction Score'
            ]
            export_df = df[[col for col in perf_columns if col in df.columns]]
            filename_base = "performance_report"
        
        elif report_type == "Sales Report":
            st.info("💰 Export sales and deal metrics")
            sales_columns = [
                'Rep ID', 'Full Name', 'Role / Position', 'Territory / Region',
                'Monthly Sales Target', 'Actual Sales (Month)', '% to Goal',
                'Total Deals Closed (Month)', 'Average Deal Size', 'Pipeline Value',
                'Win Rate (%)', 'Commission Earned (Month)'
            ]
            export_df = df[[col for col in sales_columns if col in df.columns]]
            filename_base = "sales_report"
        
        elif report_type == "Activity Report":
            st.info("📞 Export activity and engagement metrics")
            activity_columns = [
                'Rep ID', 'Full Name', 'Role / Position',
                'Outbound Calls Made', 'Outbound Emails Sent', 'Meetings Booked',
                'Meetings Completed', 'New Leads Assigned', 'Leads Contacted',
                'Daily Activity Score', 'Tasks Completed', 'Tasks Overdue'
            ]
            export_df = df[[col for col in activity_columns if col in df.columns]]
            filename_base = "activity_report"
        
        elif report_type == "Compensation Report":
            st.info("💵 Export compensation and earnings data")
            comp_columns = [
                'Rep ID', 'Full Name', 'Role / Position',
                'Base Salary', 'Commission Rate', 'Commission Earned (Month)',
                'Bonuses Earned', 'Total Compensation to Date'
            ]
            export_df = df[[col for col in comp_columns if col in df.columns]]
            filename_base = "compensation_report"
        
        else:  # Custom Report
            st.info("🔧 Select custom columns to export")
            selected_cols = st.multiselect(
                "Choose columns for custom report",
                df.columns.tolist(),
                default=['Rep ID', 'Full Name', 'Role / Position', 'Employment Status']
            )
            export_df = df[selected_cols] if selected_cols else df
            filename_base = "custom_report"
        
        # Preview
        st.subheader("📋 Report Preview")
        st.dataframe(export_df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 of {len(export_df)} rows")
        
        # Download buttons
        st.divider()
        st.subheader("📥 Download Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv_data,
                file_name=f"{filename_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # For Excel, we'll use CSV as fallback since openpyxl might not be available
            st.download_button(
                label="📊 Download as Excel (CSV)",
                data=csv_data,
                file_name=f"{filename_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="Downloads as CSV compatible with Excel"
            )
        
        with col3:
            json_data = export_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📦 Download as JSON",
                data=json_data,
                file_name=f"{filename_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Summary statistics
        st.divider()
        st.subheader("📈 Report Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(export_df))
        
        with col2:
            st.metric("Total Columns", len(export_df.columns))
        
        with col3:
            if 'Actual Sales (Month)' in export_df.columns:
                total_sales = pd.to_numeric(export_df['Actual Sales (Month)'], errors='coerce').sum()
                st.metric("Total Sales", f"${total_sales:,.0f}")
        
        with col4:
            if 'Performance Score' in export_df.columns:
                avg_perf = pd.to_numeric(export_df['Performance Score'], errors='coerce').mean()
                st.metric("Avg Performance", f"{avg_perf:.1f}")
        
        # Additional export options
        st.divider()
        st.subheader("⚙️ Advanced Export Options")
        
        with st.expander("Schedule Automated Reports"):
            st.info("🚀 Feature Coming Soon: Schedule automated reports to be emailed daily, weekly, or monthly")
            
            schedule_freq = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])
            recipient_email = st.text_input("Recipient Email")
            
            if st.button("Set Up Scheduled Report"):
                st.warning("Scheduled reports feature will be available in the next update!")

else:
    # Welcome Screen
    st.markdown('<div class="main-header">📊 Advanced Sales Management Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("👈 Please upload a CSV file or connect to Google Sheets from the sidebar to get started")
    
    st.divider()
    
    # Feature showcase
    st.subheader("🚀 Dashboard Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Analytics
        - Executive Dashboard
        - Performance Tracking
        - Trend Analysis
        - Predictive Insights
        - Correlation Analysis
        """)
    
    with col2:
        st.markdown("""
        ### ✏️ Data Management
        - Live Editing
        - Bulk Operations
        - Google Sheets Sync
        - CSV Import/Export
        - Column Customization
        """)
    
    with col3:
        st.markdown("""
        ### 👥 Team Management
        - Add/Edit/Delete Reps
        - Performance Scoring
        - Invoice Tracking
        - Activity Monitoring
        - Custom Reports
        """)
    
    st.divider()
    
    # Getting started guide
    st.subheader("📚 Getting Started")
    
    tab1, tab2 = st.tabs(["CSV Upload", "Google Sheets Setup"])
    
    with tab1:
        st.markdown("""
        ### Upload CSV File
        
        1. Click on **"CSV File"** in the sidebar
        2. Click **"Browse files"** or drag & drop your CSV
        3. Your data will load automatically
        4. Start exploring the dashboard!
        
        **CSV Format:** Ensure your CSV has columns for Rep ID, Name, Email, Sales data, etc.
        """)
    
    with tab2:
        st.markdown("""
        ### Connect to Google Sheets
        
        #### Step 1: Google Cloud Setup
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select existing
        3. Enable **Google Sheets API** for your project
        
        #### Step 2: Create Service Account
        1. Navigate to **IAM & Admin** → **Service Accounts**
        2. Click **Create Service Account**
        3. Give it a name (e.g., "Sales Dashboard")
        4. Grant **Editor** role
        5. Click **Done**
        
        #### Step 3: Create JSON Key
        1. Click on your service account
        2. Go to **Keys** tab
        3. Click **Add Key** → **Create new key**
        4. Select **JSON** format
        5. Download the JSON file
        
        #### Step 4: Share Your Sheet
        1. Open your Google Sheet
        2. Click **Share** button
        3. Add the service account email (found in JSON file)
        4. Give **Editor** permissions
        
        #### Step 5: Connect in Dashboard
        1. Select **"Google Sheets"** in sidebar
        2. Upload your JSON key file
        3. Paste your Google Sheets URL
        4. Click **Connect to Google Sheets**
        
        ✅ You're all set! Changes will sync automatically.
        """)
    
    # Sample data structure
    st.divider()
    st.subheader("📋 Expected Data Structure")
    
    with st.expander("View Sample CSV Structure"):
        st.code("""
Rep ID,Full Name,Email,Phone,Role / Position,Territory / Region,Employment Status,Monthly Sales Target,Actual Sales (Month)
1,John Doe,john@example.com,555-0001,Field Rep,National,Active,50000,35000
2,Jane Smith,jane@example.com,555-0002,BDR,South,Active,60000,58000
        """, language="csv")
    
    # Support info
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Need Help?**
        
        Check the **Help & Instructions** section in the sidebar for detailed guides and troubleshooting tips.
        """)
    
    with col2:
        st.success("""
        **Pro Tip!**
        
        Enable **Auto-sync** to automatically save changes to Google Sheets in real-time.
        """)
