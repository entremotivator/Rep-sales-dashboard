import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import gspread
from google.oauth2.service_account import Credentials
from io import StringIO

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
    
    # Main Tabs - Extended to 10 tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📈 Executive Dashboard",
        "📊 Analytics & Insights",
        "👥 Team Overview",
        "🎯 Performance Tracking",
        "✏️ Data Management",
        "➕ Add New Rep",
        "🔧 Edit Rep",
        "📄 Invoice Management",
        "📑 Reports & Export",
        "🔮 Forecasting & Trends"
    ])
    
    # TAB 1: Executive Dashboard (Enhanced)
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
        
        # Enhanced Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales Performance Comparison")
            # Top 20 reps by performance
            top_performers = df.nlargest(20, 'Performance Score')
            
            fig1 = go.Figure()
            
            fig1.add_trace(go.Bar(
                name='Target',
                x=top_performers['Full Name'],
                y=top_performers['Monthly Sales Target'],
                marker_color='#636EFA'
            ))
            
            fig1.add_trace(go.Bar(
                name='Actual',
                x=top_performers['Full Name'],
                y=top_performers['Actual Sales (Month)'],
                marker_color='#00CC96'
            ))
            
            fig1.update_layout(
                barmode='group',
                xaxis_tickangle=-45,
                height=450,
                hovermode='x unified',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Territory Performance Dashboard")
            territory_sales = df.groupby('Territory / Region').agg({
                'Actual Sales (Month)': 'sum',
                'Pipeline Value': 'sum',
                'Win Rate (%)': 'mean',
                'Performance Score': 'mean'
            }).reset_index()
            
            fig2 = px.sunburst(
                territory_sales,
                path=['Territory / Region'],
                values='Actual Sales (Month)',
                color='Performance Score',
                color_continuous_scale='RdYlGn',
                title="Sales Distribution by Territory"
            )
            fig2.update_layout(height=450)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Additional Charts Row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Win Rate vs Activity Score")
            
            fig3 = px.scatter(
                df,
                x='Daily Activity Score',
                y='Win Rate (%)',
                size='Actual Sales (Month)',
                color='Performance Score',
                hover_name='Full Name',
                title="Performance Matrix",
                color_continuous_scale='Viridis',
                labels={'Daily Activity Score': 'Activity Score', 'Win Rate (%)': 'Win Rate (%)'}
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.subheader("Pipeline Health Analysis")
            
            pipeline_data = df.groupby('Role / Position').agg({
                'Pipeline Value': 'sum',
                'Opportunities in Progress': 'sum',
                'Opportunities Won': 'sum'
            }).reset_index()
            
            fig4 = go.Figure()
            
            fig4.add_trace(go.Bar(
                name='Pipeline Value',
                x=pipeline_data['Role / Position'],
                y=pipeline_data['Pipeline Value'],
                marker_color='#FFA15A'
            ))
            
            fig4.add_trace(go.Bar(
                name='Opps Won',
                x=pipeline_data['Role / Position'],
                y=pipeline_data['Opportunities Won'] * 10000,  # Scale for visibility
                marker_color='#19D3F3'
            ))
            
            fig4.update_layout(
                barmode='group',
                height=400,
                xaxis_tickangle=-45,
                yaxis_title="Value ($)"
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        # Top Performers Section (Enhanced)
        st.divider()
        st.subheader("🏆 Top Performers - Hall of Fame")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**🥇 Top by Sales**")
            top_sales = df.nlargest(5, 'Actual Sales (Month)')[
                ['Full Name', 'Actual Sales (Month)', '% to Goal']
            ]
            top_sales['Actual Sales (Month)'] = top_sales['Actual Sales (Month)'].apply(lambda x: f"${x:,.0f}")
            top_sales['% to Goal'] = top_sales['% to Goal'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_sales, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**🎯 Top by Win Rate**")
            top_win_rate = df.nlargest(5, 'Win Rate (%)')[
                ['Full Name', 'Win Rate (%)', 'Total Deals Closed (Month)']
            ]
            top_win_rate['Win Rate (%)'] = top_win_rate['Win Rate (%)'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_win_rate, hide_index=True, use_container_width=True)
        
        with col3:
            st.markdown("**⚡ Top by Performance**")
            top_performance = df.nlargest(5, 'Performance Score')[
                ['Full Name', 'Performance Score', 'Daily Activity Score']
            ]
            top_performance['Performance Score'] = top_performance['Performance Score'].apply(lambda x: f"{x:.1f}")
            st.dataframe(top_performance, hide_index=True, use_container_width=True)
        
        with col4:
            st.markdown("**😊 Top by CSAT**")
            top_csat = df.nlargest(5, 'Customer Satisfaction Score')[
                ['Full Name', 'Customer Satisfaction Score', 'Win Rate (%)']
            ]
            top_csat['Customer Satisfaction Score'] = top_csat['Customer Satisfaction Score'].apply(lambda x: f"{x:.2f}/5.0")
            top_csat['Win Rate (%)'] = top_csat['Win Rate (%)'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(top_csat, hide_index=True, use_container_width=True)
        
        # Needs Attention Section
        st.divider()
        st.subheader("⚠️ Needs Attention & Action Items")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📉 Below Target Performance**")
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
            st.markdown("**📋 Overdue Tasks**")
            overdue_tasks = df[df['Tasks Overdue'] > 0].nlargest(5, 'Tasks Overdue')[
                ['Full Name', 'Tasks Overdue', 'Tasks Completed', 'Follow-Ups Due']
            ]
            if not overdue_tasks.empty:
                st.dataframe(overdue_tasks, hide_index=True, use_container_width=True)
            else:
                st.success("✅ No overdue tasks!")
    
    # TAB 2: Analytics & Insights (Enhanced)
    with tab2:
        st.header("Advanced Analytics & Business Intelligence")
        
        # Analysis Options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Correlation Analysis", "Trend Analysis", "Comparative Analysis", "Predictive Insights", "Cohort Analysis"]
            )
        
        with col2:
            metric_focus = st.selectbox(
                "Focus Metric",
                ["Sales Performance", "Activity Metrics", "Win Rates", "Customer Satisfaction", "Pipeline Health"]
            )
        
        with col3:
            group_by = st.selectbox(
                "Group By",
                ["Territory / Region", "Role / Position", "Employment Status", "Manager Name"]
            )
        
        st.divider()
        
        # Correlation Analysis (Enhanced)
        if analysis_type == "Correlation Analysis":
            st.subheader("📊 Multi-Dimensional Correlation Matrix")
            
            correlation_columns = [
                'Actual Sales (Month)', 'Win Rate (%)', 'Daily Activity Score',
                'Outbound Calls Made', 'Outbound Emails Sent', 'Meetings Completed',
                'Customer Satisfaction Score', 'Performance Score', 'Pipeline Value',
                'Conversion Rate', 'Meeting Completion Rate', 'Lead Contact Rate'
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
                title="Correlation between Key Performance Metrics",
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=700)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Key Insights
            st.subheader("🔍 AI-Powered Insights")
            col1, col2, col3 = st.columns(3)
            
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
            
            with col3:
                # Sales correlation with activity
                if 'Actual Sales (Month)' in available_cols and 'Daily Activity Score' in available_cols:
                    sales_activity_corr = corr_matrix.loc['Actual Sales (Month)', 'Daily Activity Score']
                    st.info(f"**Sales-Activity Correlation:** {sales_activity_corr:.2f} {'📈 Strong' if abs(sales_activity_corr) > 0.5 else '📊 Moderate'}")
        
        # Trend Analysis (Enhanced)
        elif analysis_type == "Trend Analysis":
            st.subheader("📈 Multi-Variable Trend Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Activity vs Performance scatter with trendline
                fig_scatter = px.scatter(
                    df,
                    x='Daily Activity Score',
                    y='Actual Sales (Month)',
                    size='Win Rate (%)',
                    color='Performance Score',
                    hover_name='Full Name',
                    title="Activity Score vs Sales Performance (with Trend)",
                    labels={'Daily Activity Score': 'Activity Score', 'Actual Sales (Month)': 'Sales ($)'},
                    color_continuous_scale='Viridis',
                    trendline="ols"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # Pipeline vs Closed Deals with size by performance
                fig_pipeline = px.scatter(
                    df,
                    x='Pipeline Value',
                    y='Total Deals Closed (Month)',
                    size='Performance Score',
                    color='Role / Position',
                    hover_name='Full Name',
                    title="Pipeline Value vs Deals Closed",
                    labels={'Pipeline Value': 'Pipeline ($)', 'Total Deals Closed (Month)': 'Deals Closed'}
                )
                st.plotly_chart(fig_pipeline, use_container_width=True)
            
            # Meeting Effectiveness Analysis
            st.subheader("🤝 Meeting Effectiveness & Conversion Analysis")
            df['Meeting Effectiveness'] = (df['Total Deals Closed (Month)'] / df['Meetings Completed'].replace(0, 1)).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                top_20_meetings = df.nlargest(20, 'Meetings Completed')
                fig_meetings = px.bar(
                    top_20_meetings,
                    x='Full Name',
                    y=['Meetings Booked', 'Meetings Completed'],
                    title="Top 20 Reps: Meetings Booked vs Completed",
                    barmode='group',
                    color_discrete_sequence=['#FFA15A', '#19D3F3']
                )
                fig_meetings.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig_meetings, use_container_width=True)
            
            with col2:
                # Meeting effectiveness ranking
                top_effectiveness = df.nlargest(20, 'Meeting Effectiveness')
                fig_effectiveness = px.bar(
                    top_effectiveness,
                    x='Full Name',
                    y='Meeting Effectiveness',
                    title="Top 20 Reps: Meeting Effectiveness (Deals/Meeting)",
                    color='Meeting Effectiveness',
                    color_continuous_scale='Greens'
                )
                fig_effectiveness.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig_effectiveness, use_container_width=True)
        
        # Comparative Analysis (Enhanced)
        elif analysis_type == "Comparative Analysis":
            st.subheader(f"📊 Comprehensive Comparison by {group_by}")
            
            grouped_data = df.groupby(group_by).agg({
                'Actual Sales (Month)': ['sum', 'mean', 'std'],
                'Win Rate (%)': ['mean', 'median'],
                'Daily Activity Score': ['mean', 'median'],
                'Total Deals Closed (Month)': ['sum', 'mean'],
                'Customer Satisfaction Score': ['mean', 'std'],
                'Performance Score': ['mean', 'median', 'std']
            }).round(2)
            
            grouped_data.columns = ['Total Sales', 'Avg Sales', 'Sales StdDev', 'Avg Win Rate', 'Median Win Rate', 
                                   'Avg Activity', 'Median Activity', 'Total Deals', 'Avg Deals', 
                                   'Avg CSAT', 'CSAT StdDev', 'Avg Performance', 'Median Performance', 'Performance StdDev']
            
            st.dataframe(grouped_data.style.background_gradient(cmap='RdYlGn', subset=['Avg Performance']), use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig_comp1 = px.bar(
                    grouped_data.reset_index(),
                    x=group_by,
                    y='Total Sales',
                    title=f"Total Sales by {group_by}",
                    color='Avg Performance',
                    color_continuous_scale='Viridis',
                    text='Total Sales'
                )
                fig_comp1.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                st.plotly_chart(fig_comp1, use_container_width=True)
            
            with col2:
                fig_comp2 = px.scatter(
                    grouped_data.reset_index(),
                    x='Avg Win Rate',
                    y='Avg Activity',
                    size='Total Deals',
                    color='Avg Performance',
                    text=group_by,
                    title=f"Win Rate vs Activity by {group_by}",
                    color_continuous_scale='Plasma'
                )
                fig_comp2.update_traces(textposition='top center')
                st.plotly_chart(fig_comp2, use_container_width=True)
        
        # Cohort Analysis (New)
        elif analysis_type == "Cohort Analysis":
            st.subheader("👥 Cohort Analysis by Start Date")
            
            # Convert Start Date to datetime
            df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
            df['Tenure Months'] = ((datetime.now() - df['Start Date']).dt.days / 30).round(0)
            
            # Create cohorts
            df['Cohort'] = pd.cut(df['Tenure Months'], 
                                 bins=[0, 3, 6, 12, 24, 1000], 
                                 labels=['0-3 months', '3-6 months', '6-12 months', '1-2 years', '2+ years'])
            
            cohort_analysis = df.groupby('Cohort').agg({
                'Actual Sales (Month)': 'mean',
                'Win Rate (%)': 'mean',
                'Performance Score': 'mean',
                'Customer Satisfaction Score': 'mean',
                'Full Name': 'count'
            }).round(2)
            
            cohort_analysis.columns = ['Avg Sales', 'Avg Win Rate', 'Avg Performance', 'Avg CSAT', 'Rep Count']
            
            st.dataframe(cohort_analysis, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cohort1 = px.bar(
                    cohort_analysis.reset_index(),
                    x='Cohort',
                    y='Avg Sales',
                    title="Average Sales by Tenure Cohort",
                    color='Avg Performance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_cohort1, use_container_width=True)
            
            with col2:
                fig_cohort2 = px.line(
                    cohort_analysis.reset_index(),
                    x='Cohort',
                    y=['Avg Win Rate', 'Avg CSAT'],
                    title="Win Rate & CSAT by Tenure",
                    markers=True
                )
                st.plotly_chart(fig_cohort2, use_container_width=True)
        
        # Predictive Insights (Enhanced)
        else:
            st.subheader("🔮 Advanced Predictive Insights & AI Recommendations")
            
            # Calculate predictions based on current performance
            current_day = datetime.now().day
            days_in_month = 30
            df['Projected Monthly Sales'] = (df['Actual Sales (Month)'] / current_day * days_in_month).round(0)
            df['Target Gap'] = df['Monthly Sales Target'] - df['Projected Monthly Sales']
            df['Probability of Success'] = ((df['% to Goal'] / 100) * 0.4 + 
                                            (df['Win Rate (%)'] / 100) * 0.3 + 
                                            (df['Daily Activity Score'] / 100) * 0.3) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎯 Likely to Exceed Target**")
                likely_exceed = df[df['Projected Monthly Sales'] > df['Monthly Sales Target']].nlargest(10, 'Projected Monthly Sales')[
                    ['Full Name', 'Projected Monthly Sales', 'Monthly Sales Target', '% to Goal', 'Probability of Success']
                ]
                if not likely_exceed.empty:
                    likely_exceed['Projected Monthly Sales'] = likely_exceed['Projected Monthly Sales'].apply(lambda x: f"${x:,.0f}")
                    likely_exceed['Monthly Sales Target'] = likely_exceed['Monthly Sales Target'].apply(lambda x: f"${x:,.0f}")
                    likely_exceed['% to Goal'] = likely_exceed['% to Goal'].apply(lambda x: f"{x:.1f}%")
                    likely_exceed['Probability of Success'] = likely_exceed['Probability of Success'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(likely_exceed, hide_index=True, use_container_width=True)
                else:
                    st.info("No reps currently projected to exceed target")
            
            with col2:
                st.markdown("**⚠️ At Risk of Missing Target**")
                at_risk = df[df['Target Gap'] > 0].nlargest(10, 'Target Gap')[
                    ['Full Name', 'Projected Monthly Sales', 'Target Gap', '% to Goal', 'Probability of Success']
                ]
                if not at_risk.empty:
                    at_risk['Projected Monthly Sales'] = at_risk['Projected Monthly Sales'].apply(lambda x: f"${x:,.0f}")
                    at_risk['Target Gap'] = at_risk['Target Gap'].apply(lambda x: f"${x:,.0f}")
                    at_risk['% to Goal'] = at_risk['% to Goal'].apply(lambda x: f"{x:.1f}%")
                    at_risk['Probability of Success'] = at_risk['Probability of Success'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(at_risk, hide_index=True, use_container_width=True)
                else:
                    st.success("All reps on track to meet targets!")
            
            # Recommendations Engine
            st.divider()
            st.subheader("💡 AI-Powered Action Recommendations")
            
            recommendations = []
            
            # Low activity reps
            low_activity = df[df['Daily Activity Score'] < 70]
            if not low_activity.empty:
                recommendations.append({
                    'Priority': '🔴 High',
                    'Category': 'Activity',
                    'Issue': f"{len(low_activity)} reps with low activity scores",
                    'Action': 'Schedule coaching sessions and review daily routines',
                    'Impact': 'Potential 15-20% increase in sales'
                })
            
            # Low win rate
            low_win_rate = df[df['Win Rate (%)'] < 20]
            if not low_win_rate.empty:
                recommendations.append({
                    'Priority': '🟡 Medium',
                    'Category': 'Conversion',
                    'Issue': f"{len(low_win_rate)} reps with win rates below 20%",
                    'Action': 'Provide sales training and review qualification process',
                    'Impact': 'Expected 10-15% improvement in close rates'
                })
            
            # High pipeline, low closes
            high_pipeline_low_close = df[(df['Pipeline Value'] > df['Pipeline Value'].median()) & 
                                          (df['Total Deals Closed (Month)'] < df['Total Deals Closed (Month)'].median())]
            if not high_pipeline_low_close.empty:
                recommendations.append({
                    'Priority': '🟡 Medium',
                    'Category': 'Pipeline',
                    'Issue': f"{len(high_pipeline_low_close)} reps with high pipeline but low closes",
                    'Action': 'Review deal progression and remove stale opportunities',
                    'Impact': 'Unlock $50K-100K in potential revenue'
                })
            
            # Low CSAT
            low_csat = df[df['Customer Satisfaction Score'] < 3.5]
            if not low_csat.empty:
                recommendations.append({
                    'Priority': '🔴 High',
                    'Category': 'Customer Success',
                    'Issue': f"{len(low_csat)} reps with CSAT below 3.5",
                    'Action': 'Customer service training and follow-up process review',
                    'Impact': 'Improve retention and referral rates'
                })
            
            # Meeting completion issues
            low_meeting_completion = df[df['Meeting Completion Rate'] < 60]
            if not low_meeting_completion.empty:
                recommendations.append({
                    'Priority': '🟡 Medium',
                    'Category': 'Efficiency',
                    'Issue': f"{len(low_meeting_completion)} reps with low meeting completion rates",
                    'Action': 'Improve scheduling and reminder systems',
                    'Impact': '20% increase in qualified opportunities'
                })
            
            if recommendations:
                rec_df = pd.DataFrame(recommendations)
                st.dataframe(rec_df, hide_index=True, use_container_width=True)
            else:
                st.success("✅ No critical issues detected. Team is performing exceptionally well!")
            
            # Visual prediction dashboard
            st.divider()
            st.subheader("📊 Prediction Dashboard")
            
            fig_prediction = px.scatter(
                df,
                x='Actual Sales (Month)',
                y='Projected Monthly Sales',
                size='Probability of Success',
                color='Performance Score',
                hover_name='Full Name',
                title="Current Sales vs Projected End-of-Month Sales",
                labels={'Actual Sales (Month)': 'Current Sales ($)', 'Projected Monthly Sales': 'Projected Sales ($)'},
                color_continuous_scale='RdYlGn'
            )
            
            # Add diagonal line for reference
            max_val = max(df['Actual Sales (Month)'].max(), df['Projected Monthly Sales'].max())
            fig_prediction.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Target Line',
                line=dict(color='red', dash='dash')
            ))
            
            st.plotly_chart(fig_prediction, use_container_width=True)
    
    # TAB 3: Team Overview (Enhanced)
    with tab3:
        st.header("👥 Team Overview & Organizational Structure")
        
        # Team Summary Cards
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            active_count = len(df[df['Employment Status'] == 'Active'])
            inactive_count = len(df[df['Employment Status'] == 'Inactive'])
            st.metric("Active Reps", active_count, f"{inactive_count} Inactive")
        
        with col2:
            full_time = len(df[df['Contract Type'] == 'Full-Time'])
            st.metric("Full-Time", full_time, f"{len(df) - full_time} Other")
        
        with col3:
            trained = len(df[df['Training Completed'] == 'Yes'])
            st.metric("Trained Reps", trained, f"{len(df) - trained} Pending")
        
        with col4:
            df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
            avg_tenure_days = (datetime.now() - df['Start Date']).dt.days.mean()
            st.metric("Avg Tenure", f"{int(avg_tenure_days)} days", "Experience")
        
        with col5:
            team_performance = df['Performance Score'].mean()
            st.metric("Team Performance", f"{team_performance:.1f}", "Average Score")
        
        st.divider()
        
        # Team Distribution (Enhanced)
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
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Team Composition by Role"
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
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Geographic Distribution"
            )
            fig_territory.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_territory, use_container_width=True)
        
        with col3:
            st.subheader("Performance Distribution")
            perf_ranges = pd.cut(df['Performance Score'], 
                                bins=[0, 60, 75, 90, 100], 
                                labels=['Needs Improvement', 'Average', 'Good', 'Excellent'])
            perf_dist = perf_ranges.value_counts().reset_index()
            perf_dist.columns = ['Performance Level', 'Count']
            
            fig_perf = px.bar(
                perf_dist,
                x='Performance Level',
                y='Count',
                color='Performance Level',
                color_discrete_map={
                    'Needs Improvement': '#EF553B',
                    'Average': '#FFA15A',
                    'Good': '#00CC96',
                    'Excellent': '#636EFA'
                },
                title="Performance Level Distribution"
            )
            fig_perf.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_perf, use_container_width=True)
        
        # Manager Performance Breakdown
        st.divider()
        st.subheader("👔 Manager Performance Breakdown")
        
        manager_stats = df.groupby('Manager Name').agg({
            'Full Name': 'count',
            'Actual Sales (Month)': ['sum', 'mean'],
            'Performance Score': 'mean',
            'Win Rate (%)': 'mean',
            'Customer Satisfaction Score': 'mean'
        }).round(2)
        
        manager_stats.columns = ['Team Size', 'Total Sales', 'Avg Sales', 'Avg Performance', 'Avg Win Rate', 'Avg CSAT']
        manager_stats = manager_stats.reset_index()
        
        fig_manager = px.bar(
            manager_stats,
            x='Manager Name',
            y='Total Sales',
            color='Avg Performance',
            title="Manager Team Performance Overview",
            color_continuous_scale='Viridis',
            text='Team Size'
        )
        fig_manager.update_traces(texttemplate='Team: %{text}', textposition='outside')
        st.plotly_chart(fig_manager, use_container_width=True)
        
        st.dataframe(manager_stats, hide_index=True, use_container_width=True)
        
        # Detailed Team Roster
        st.divider()
        st.subheader("📋 Comprehensive Team Roster")
        
        # Add performance indicators
        roster_df = df[[
            'Full Name', 'Role / Position', 'Territory / Region', 'Employment Status',
            'Actual Sales (Month)', '% to Goal', 'Win Rate (%)', 'Performance Score',
            'Customer Satisfaction Score', 'Manager Name', 'Start Date'
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
    
    # TAB 4: Performance Tracking (keep as is, already comprehensive)
    with tab4:
        st.header("🎯 Detailed Performance Tracking & Scorecards")
        
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
            
            st.dataframe(sales_metrics, hide_index=True, use_container_width=True, height=400)
        
        with col2:
            st.subheader("Activity Performance Metrics")
            
            activity_metrics = perf_df[[
                'Full Name', 'Outbound Calls Made', 'Outbound Emails Sent',
                'Meetings Booked', 'Meetings Completed', 'Daily Activity Score'
            ]].copy()
            
            st.dataframe(activity_metrics, hide_index=True, use_container_width=True, height=400)
        
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
            
            st.dataframe(lead_metrics, hide_index=True, use_container_width=True, height=400)
        
        with col2:
            st.subheader("Opportunity Pipeline")
            
            opp_metrics = perf_df[[
                'Full Name', 'Opportunities Created', 'Opportunities in Progress',
                'Opportunities Won', 'Opportunities Lost', 'Conversion Rate'
            ]].copy()
            
            opp_metrics['Conversion Rate'] = opp_metrics['Conversion Rate'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(opp_metrics, hide_index=True, use_container_width=True, height=400)
        
        # Individual Performance Deep Dive
        st.divider()
        st.subheader("🔍 Individual Performance Deep Dive & 360° Analysis")
        
        selected_rep_perf = st.selectbox(
            "Select Rep for Detailed Analysis",
            perf_df['Full Name'].tolist(),
            key="perf_deep_dive"
        )
        
        if selected_rep_perf:
            rep_detail = perf_df[perf_df['Full Name'] == selected_rep_perf].iloc[0]
            
            # Rep header
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Performance Score", f"{rep_detail['Performance Score']:.1f}")
            
            with col2:
                perf_level, perf_icon = get_performance_level(rep_detail['Performance Score'])
                st.metric("Performance Level", f"{perf_icon} {perf_level}")
            
            with col3:
                st.metric("% to Goal", f"{rep_detail['% to Goal']:.1f}%")
            
            with col4:
                st.metric("CSAT Score", f"{rep_detail['Customer Satisfaction Score']:.2f}/5.0")
            
            with col5:
                st.metric("Win Rate", f"{rep_detail['Win Rate (%)']:.1f}%")
            
            # Detailed metrics tabs
            detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs([
                "Sales Breakdown", "Activity Analysis", "Performance Radar", "Trends & History"
            ])
            
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
                        title="Sales Funnel Performance"
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
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tasks Completed", int(rep_detail['Tasks Completed']))
                with col2:
                    st.metric("Tasks Overdue", int(rep_detail['Tasks Overdue']), 
                             delta=f"-{int(rep_detail['Tasks Overdue'])}" if rep_detail['Tasks Overdue'] > 0 else "0")
                with col3:
                    completion_rate = (rep_detail['Tasks Completed'] / 
                                      (rep_detail['Tasks Completed'] + rep_detail['Tasks Overdue'])) * 100 if (rep_detail['Tasks Completed'] + rep_detail['Tasks Overdue']) > 0 else 0
                    st.metric("Completion Rate", f"{completion_rate:.1f}%")
            
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
                    title="Performance vs Team Average",
                    height=500
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with detail_tab4:
                st.info("📊 Historical trend data would be displayed here with time-series analysis")
                
                # Simulate trend data visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Key Strengths**")
                    strengths = []
                    if rep_detail['Win Rate (%)'] > df['Win Rate (%)'].mean():
                        strengths.append("✅ Above-average win rate")
                    if rep_detail['Customer Satisfaction Score'] > 4.0:
                        strengths.append("✅ Excellent customer satisfaction")
                    if rep_detail['Daily Activity Score'] > 80:
                        strengths.append("✅ High activity engagement")
                    if rep_detail['% to Goal'] > 100:
                        strengths.append("✅ Exceeding sales target")
                    
                    if strengths:
                        for strength in strengths:
                            st.write(strength)
                    else:
                        st.write("Areas for improvement identified")
                
                with col2:
                    st.markdown("**Development Areas**")
                    areas = []
                    if rep_detail['Win Rate (%)'] < df['Win Rate (%)'].mean():
                        areas.append("⚠️ Focus on improving conversion rate")
                    if rep_detail['Customer Satisfaction Score'] < 3.5:
                        areas.append("⚠️ Customer service training recommended")
                    if rep_detail['Daily Activity Score'] < 70:
                        areas.append("⚠️ Increase daily engagement")
                    if rep_detail['% to Goal'] < 80:
                        areas.append("⚠️ Review sales strategy")
                    
                    if areas:
                        for area in areas:
                            st.write(area)
                    else:
                        st.write("✅ All metrics performing well!")
    
    # TAB 5: Data Management (keep existing, it's comprehensive)
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
                st.info("Use the bulk operations section below")
        
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
    
    # TAB 6-9: Keep existing tabs as they are comprehensive
    # TAB 10: NEW - Forecasting & Trends
    with tab10:
        st.header("🔮 Advanced Forecasting & Trend Analysis")
        
        st.subheader("📈 Sales Forecast & Revenue Projections")
        
        # Calculate forecasts
        current_day = datetime.now().day
        days_in_month = 30
        days_remaining = days_in_month - current_day
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_total = df['Actual Sales (Month)'].sum()
            st.metric("Current Month Sales", f"${current_total:,.0f}")
        
        with col2:
            projected_total = (current_total / current_day * days_in_month)
            st.metric("Projected End-of-Month", f"${projected_total:,.0f}", 
                     f"+${projected_total - current_total:,.0f}")
        
        with col3:
            target_total = df['Monthly Sales Target'].sum()
            st.metric("Total Target", f"${target_total:,.0f}")
        
        with col4:
            gap = projected_total - target_total
            st.metric("Projected Gap", f"${gap:,.0f}", 
                     f"{'✅' if gap >= 0 else '⚠️'} {(gap/target_total*100):.1f}%")
        
        st.divider()
        
        # Forecasting charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Individual Rep Forecasts")
            
            top_20_forecast = df.nlargest(20, 'Actual Sales (Month)').copy()
            top_20_forecast['Projected Sales'] = (top_20_forecast['Actual Sales (Month)'] / current_day * days_in_month)
            
            fig_forecast = go.Figure()
            
            fig_forecast.add_trace(go.Bar(
                name='Current Sales',
                x=top_20_forecast['Full Name'],
                y=top_20_forecast['Actual Sales (Month)'],
                marker_color='#636EFA'
            ))
            
            fig_forecast.add_trace(go.Bar(
                name='Projected Sales',
                x=top_20_forecast['Full Name'],
                y=top_20_forecast['Projected Sales'],
                marker_color='#00CC96'
            ))
            
            fig_forecast.add_trace(go.Scatter(
                name='Target',
                x=top_20_forecast['Full Name'],
                y=top_20_forecast['Monthly Sales Target'],
                mode='lines+markers',
                line=dict(color='red', dash='dash'),
                marker=dict(size=8)
            ))
            
            fig_forecast.update_layout(
                xaxis_tickangle=-45,
                height=450,
                hovermode='x unified',
                barmode='group'
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
        
        with col2:
            st.subheader("Territory Performance Forecast")
            
            territory_forecast = df.groupby('Territory / Region').agg({
                'Actual Sales (Month)': 'sum',
                'Monthly Sales Target': 'sum'
            }).reset_index()
            
            territory_forecast['Projected Sales'] = (territory_forecast['Actual Sales (Month)'] / current_day * days_in_month)
            territory_forecast['Attainment %'] = (territory_forecast['Projected Sales'] / territory_forecast['Monthly Sales Target'] * 100).round(1)
            
            fig_territory_forecast = px.bar(
                territory_forecast,
                x='Territory / Region',
                y=['Actual Sales (Month)', 'Projected Sales', 'Monthly Sales Target'],
                title="Territory Sales: Current, Projected, and Target",
                barmode='group',
                color_discrete_sequence=['#636EFA', '#00CC96', '#EF553B']
            )
            
            st.plotly_chart(fig_territory_forecast, use_container_width=True)
        
        st.divider()
        
        # Trend Analysis
        st.subheader("📊 Performance Trend Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🚀 Momentum Leaders**")
            df['Sales Velocity'] = df['Actual Sales (Month)'] / current_day
            momentum_leaders = df.nlargest(5, 'Sales Velocity')[['Full Name', 'Sales Velocity', '% to Goal']]
            momentum_leaders['Sales Velocity'] = momentum_leaders['Sales Velocity'].apply(lambda x: f"${x:,.0f}/day")
            momentum_leaders['% to Goal'] = momentum_leaders['% to Goal'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(momentum_leaders, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("**⚡ Activity Champions**")
            activity_champions = df.nlargest(5, 'Daily Activity Score')[['Full Name', 'Daily Activity Score', 'Productivity Index']]
            st.dataframe(activity_champions, hide_index=True, use_container_width=True)
        
        with col3:
            st.markdown("**🎯 Conversion Masters**")
            conversion_masters = df.nlargest(5, 'Conversion Rate')[['Full Name', 'Conversion Rate', 'Win Rate (%)']]
            conversion_masters['Conversion Rate'] = conversion_masters['Conversion Rate'].apply(lambda x: f"{x:.1f}%")
            conversion_masters['Win Rate (%)'] = conversion_masters['Win Rate (%)'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(conversion_masters, hide_index=True, use_container_width=True)
        
        st.divider()
        
        # Risk Assessment
        st.subheader("⚠️ Risk Assessment & Critical Alerts")
        
        risk_categories = {
            'High Risk': df[(df['% to Goal'] < 50) & (df['Daily Activity Score'] < 70)],
            'Medium Risk': df[((df['% to Goal'] >= 50) & (df['% to Goal'] < 80)) | 
                             ((df['Daily Activity Score'] >= 70) & (df['Daily Activity Score'] < 85))],
            'On Track': df[(df['% to Goal'] >= 80) & (df['Daily Activity Score'] >= 85)]
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔴 High Risk Reps", len(risk_categories['High Risk']))
            if not risk_categories['High Risk'].empty:
                high_risk_list = risk_categories['High Risk'][['Full Name', '% to Goal', 'Daily Activity Score']].head(5)
                st.dataframe(high_risk_list, hide_index=True, use_container_width=True)
        
        with col2:
            st.metric("🟡 Medium Risk Reps", len(risk_categories['Medium Risk']))
            if not risk_categories['Medium Risk'].empty:
                medium_risk_list = risk_categories['Medium Risk'][['Full Name', '% to Goal', 'Daily Activity Score']].head(5)
                st.dataframe(medium_risk_list, hide_index=True, use_container_width=True)
        
        with col3:
            st.metric("🟢 On Track Reps", len(risk_categories['On Track']))
            st.success(f"{(len(risk_categories['On Track'])/len(df)*100):.1f}% of team on track!")
        
        st.divider()
        
        # What-if scenarios
        st.subheader("🎲 What-If Scenario Planning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            activity_increase = st.slider(
                "Simulate Activity Increase (%)",
                0, 50, 10,
                help="See impact of increasing team activity by X%"
            )
        
        with col2:
            conversion_increase = st.slider(
                "Simulate Conversion Rate Increase (%)",
                0, 30, 5,
                help="See impact of improving conversion rates by X%"
            )
        
        # Calculate scenario impact
        current_projected = (df['Actual Sales (Month)'].sum() / current_day * days_in_month)
        
        # Simplified scenario calculations
        activity_impact = current_projected * (1 + activity_increase/200)  # 50% of activity increase translates to sales
        conversion_impact = current_projected * (1 + conversion_increase/100)
        combined_impact = current_projected * (1 + activity_increase/200 + conversion_increase/100)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Projection", f"${current_projected:,.0f}")
        
        with col2:
            st.metric("With Activity Boost", f"${activity_impact:,.0f}", 
                     f"+${activity_impact - current_projected:,.0f}")
        
        with col3:
            st.metric("With Conversion Boost", f"${conversion_impact:,.0f}",
                     f"+${conversion_impact - current_projected:,.0f}")
        
        with col4:
            st.metric("Combined Impact", f"${combined_impact:,.0f}",
                     f"+${combined_impact - current_projected:,.0f}")
        
        # Scenario visualization
        scenario_data = pd.DataFrame({
            'Scenario': ['Current', 'Activity +' + str(activity_increase) + '%', 
                        'Conversion +' + str(conversion_increase) + '%', 'Combined'],
            'Projected Sales': [current_projected, activity_impact, conversion_impact, combined_impact]
        })
        
        fig_scenario = px.bar(
            scenario_data,
            x='Scenario',
            y='Projected Sales',
            title="Scenario Comparison",
            color='Projected Sales',
            color_continuous_scale='Greens',
            text='Projected Sales'
        )
        fig_scenario.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_scenario.update_layout(height=400)
        
        st.plotly_chart(fig_scenario, use_container_width=True)

else:
    # Welcome Screen (keep existing)
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
        - Forecasting & Scenarios
        """)
    
    with col2:
        st.markdown("""
        ### ✏️ Data Management
        - Live Editing
        - Bulk Operations
        - Google Sheets Sync
        - CSV Import/Export
        - Column Customization
        - Advanced Filtering
        """)
    
    with col3:
        st.markdown("""
        ### 👥 Team Management
        - Add/Edit/Delete Reps
        - Performance Scoring
        - Invoice Tracking
        - Activity Monitoring
        - Custom Reports
        - 360° Rep Analysis
        """)
    
