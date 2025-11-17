import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import gspread
from google.oauth2.service_account import Credentials
from io import StringIO
import os

# Page configuration
st.set_page_config(
    page_title="Sales Reps Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stDataFrame {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
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

# Sidebar - Data Source Configuration
st.sidebar.title("🔧 Configuration")

data_source = st.sidebar.radio(
    "Select Data Source:",
    ["CSV File", "Google Sheets"],
    key="data_source_radio"
)

# CSV Upload
if data_source == "CSV File":
    st.sidebar.subheader("📁 Upload CSV")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.data_source = "csv"
        st.sidebar.success("✅ CSV loaded successfully!")

# Google Sheets Integration
else:
    st.sidebar.subheader("🔐 Google Sheets Login")
    
    service_account_file = st.sidebar.file_uploader(
        "Upload Service Account JSON",
        type=['json'],
        help="Upload your Google Cloud service account JSON file"
    )
    
    sheet_url = st.sidebar.text_input(
        "Google Sheets URL:",
        value="https://docs.google.com/spreadsheets/d/10eTuYCcJ7qavCwqT27HRI-vexCwwwKSMDJTv6B6XOr0/edit?usp=sharing",
        help="Paste your Google Sheets URL here"
    )
    
    if service_account_file and sheet_url:
        if st.sidebar.button("🔗 Connect to Google Sheets"):
            try:
                # Load service account credentials
                service_account_info = json.load(service_account_file)
                
                # Define the scope
                scope = [
                    'https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive'
                ]
                
                # Authenticate
                credentials = Credentials.from_service_account_info(
                    service_account_info,
                    scopes=scope
                )
                
                client = gspread.authorize(credentials)
                
                # Open the spreadsheet
                spreadsheet = client.open_by_url(sheet_url)
                worksheet = spreadsheet.get_worksheet(0)
                
                # Get all values
                data = worksheet.get_all_values()
                
                # Convert to DataFrame
                if len(data) > 1:
                    st.session_state.df = pd.DataFrame(data[1:], columns=data[0])
                    st.session_state.gsheet_client = worksheet
                    st.session_state.sheet_url = sheet_url
                    st.session_state.data_source = "gsheet"
                    st.sidebar.success("✅ Connected to Google Sheets!")
                else:
                    st.sidebar.error("❌ Sheet is empty!")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
    
    # Instructions
    with st.sidebar.expander("ℹ️ Setup Instructions"):
        st.markdown("""
        **To connect Google Sheets:**
        
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a project and enable Google Sheets API
        3. Create a Service Account
        4. Download the JSON key file
        5. Share your Google Sheet with the service account email
        6. Upload the JSON file here
        
        **Service Account Email Format:**
        `your-service-account@your-project.iam.gserviceaccount.com`
        """)

# Main Content
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Header
    st.markdown('<div class="main-header">📊 Sales Reps Management Dashboard</div>', unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Dashboard Overview",
        "👥 Sales Reps",
        "➕ Add New Rep",
        "✏️ Edit Rep",
        "🗑️ Delete Rep",
        "📄 Invoices"
    ])
    
    # TAB 1: Dashboard Overview
    with tab1:
        st.header("Key Metrics")
        
        # Convert numeric columns
        numeric_columns = ['Monthly Sales Target', 'Actual Sales (Month)', '% to Goal', 
                          'Total Deals Closed (Month)', 'Pipeline Value', 'Win Rate (%)',
                          'Commission Earned (Month)', 'Total Compensation to Date']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_reps = len(df)
            active_reps = len(df[df['Employment Status'] == 'Active'])
            st.metric("Total Reps", total_reps, f"{active_reps} Active")
        
        with col2:
            total_sales = df['Actual Sales (Month)'].sum()
            st.metric("Total Sales", f"${total_sales:,.0f}")
        
        with col3:
            avg_goal = df['% to Goal'].mean()
            st.metric("Avg % to Goal", f"{avg_goal:.1f}%")
        
        with col4:
            total_pipeline = df['Pipeline Value'].sum()
            st.metric("Total Pipeline", f"${total_pipeline:,.0f}")
        
        st.divider()
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sales Performance by Rep")
            fig1 = px.bar(
                df.head(12),
                x='Full Name',
                y=['Monthly Sales Target', 'Actual Sales (Month)'],
                barmode='group',
                title="Target vs Actual Sales",
                color_discrete_sequence=['#636EFA', '#EF553B']
            )
            fig1.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Sales by Territory")
            territory_sales = df.groupby('Territory / Region')['Actual Sales (Month)'].sum().reset_index()
            fig2 = px.pie(
                territory_sales,
                values='Actual Sales (Month)',
                names='Territory / Region',
                title="Sales Distribution by Territory"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Win Rate by Role")
            role_win = df.groupby('Role / Position')['Win Rate (%)'].mean().reset_index()
            fig3 = px.bar(
                role_win,
                x='Role / Position',
                y='Win Rate (%)',
                title="Average Win Rate by Position",
                color='Win Rate (%)',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            st.subheader("Performance Score Distribution")
            fig4 = px.histogram(
                df,
                x='Daily Activity Score',
                nbins=20,
                title="Activity Score Distribution",
                color_discrete_sequence=['#00CC96']
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        # Top Performers
        st.subheader("🏆 Top Performers")
        top_performers = df.nlargest(5, 'Actual Sales (Month)')[
            ['Full Name', 'Role / Position', 'Actual Sales (Month)', '% to Goal', 'Win Rate (%)']
        ]
        st.dataframe(top_performers, use_container_width=True, hide_index=True)
    
    # TAB 2: Sales Reps Data
    with tab2:
        st.header("Sales Reps Data")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.multiselect(
                "Employment Status:",
                options=df['Employment Status'].unique(),
                default=df['Employment Status'].unique()
            )
        
        with col2:
            role_filter = st.multiselect(
                "Role / Position:",
                options=df['Role / Position'].unique(),
                default=df['Role / Position'].unique()
            )
        
        with col3:
            territory_filter = st.multiselect(
                "Territory:",
                options=df['Territory / Region'].unique(),
                default=df['Territory / Region'].unique()
            )
        
        # Apply filters
        filtered_df = df[
            (df['Employment Status'].isin(status_filter)) &
            (df['Role / Position'].isin(role_filter)) &
            (df['Territory / Region'].isin(territory_filter))
        ]
        
        st.subheader(f"Showing {len(filtered_df)} of {len(df)} reps")
        
        # Editable dataframe
        edited_df = st.data_editor(
            filtered_df,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=False,
            key="main_data_editor"
        )
        
        # Save changes
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("💾 Save Changes", type="primary"):
                st.session_state.df = edited_df
                
                # Save to Google Sheets if connected
                if st.session_state.data_source == "gsheet" and st.session_state.gsheet_client:
                    try:
                        # Clear existing data
                        st.session_state.gsheet_client.clear()
                        
                        # Update with new data
                        data_to_upload = [edited_df.columns.tolist()] + edited_df.values.tolist()
                        st.session_state.gsheet_client.update('A1', data_to_upload)
                        
                        st.success("✅ Changes saved to Google Sheets!")
                    except Exception as e:
                        st.error(f"❌ Error saving to Google Sheets: {str(e)}")
                else:
                    st.success("✅ Changes saved locally!")
        
        with col2:
            # Download CSV
            csv = edited_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"sales_reps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # TAB 3: Add New Rep
    with tab3:
        st.header("➕ Add New Sales Rep")
        
        with st.form("add_rep_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                rep_id = st.text_input("Rep ID*", value=str(len(df) + 1))
                full_name = st.text_input("Full Name*")
                email = st.text_input("Email*")
                phone = st.text_input("Phone*")
                role = st.selectbox("Role / Position*", 
                    ["Field Rep", "BDR", "SDR", "Account Executive", "Sales Manager"])
                territory = st.selectbox("Territory / Region*",
                    ["National", "South", "North", "West Coast", "Midwest", "East Coast"])
                start_date = st.date_input("Start Date*")
                employment_status = st.selectbox("Employment Status*", ["Active", "Inactive"])
                contract_type = st.selectbox("Contract Type*", 
                    ["Full-Time", "Part-Time", "Contract"])
            
            with col2:
                monthly_target = st.number_input("Monthly Sales Target*", min_value=0, value=50000)
                base_salary = st.number_input("Base Salary*", min_value=0, value=60000)
                commission_rate = st.text_input("Commission Rate*", value="10%")
                manager_name = st.text_input("Manager Name", value="Chris Johnson")
                training_completed = st.selectbox("Training Completed", ["Yes", "No"])
                certifications = st.text_input("Certifications", value="Salesforce, HubSpot")
                customer_sat_score = st.number_input("Customer Satisfaction Score", 
                    min_value=0.0, max_value=5.0, value=4.0, step=0.1)
                hr_notes = st.text_area("HR Notes")
            
            submitted = st.form_submit_button("✅ Add Rep", type="primary")
            
            if submitted:
                if full_name and email and phone:
                    # Create new row with all columns from original dataframe
                    new_row = {col: "" for col in df.columns}
                    
                    # Fill in the provided values
                    new_row.update({
                        'Rep ID': rep_id,
                        'Full Name': full_name,
                        'Email': email,
                        'Phone': phone,
                        'Role / Position': role,
                        'Territory / Region': territory,
                        'Start Date': start_date.strftime('%Y-%m-%d'),
                        'Employment Status': employment_status,
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
                        'Contract Type': contract_type,
                        'Performance Review Date': "",
                        'HR Notes': hr_notes
                    })
                    
                    # Add to dataframe
                    st.session_state.df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    
                    # Save to Google Sheets if connected
                    if st.session_state.data_source == "gsheet" and st.session_state.gsheet_client:
                        try:
                            row_data = [list(new_row.values())]
                            st.session_state.gsheet_client.append_row(row_data[0])
                            st.success("✅ Rep added successfully to Google Sheets!")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                    else:
                        st.success("✅ Rep added successfully!")
                    
                    st.rerun()
                else:
                    st.error("❌ Please fill in all required fields (marked with *)")
    
    # TAB 4: Edit Rep
    with tab4:
        st.header("✏️ Edit Sales Rep")
        
        # Select rep to edit
        rep_names = df['Full Name'].tolist()
        selected_rep = st.selectbox("Select Rep to Edit:", rep_names)
        
        if selected_rep:
            rep_data = df[df['Full Name'] == selected_rep].iloc[0]
            
            with st.form("edit_rep_form"):
                st.subheader(f"Editing: {selected_rep}")
                
                # Create tabs for different sections
                edit_tab1, edit_tab2, edit_tab3, edit_tab4 = st.tabs([
                    "Personal Info", "Sales Metrics", "Activity", "Compensation"
                ])
                
                updated_data = {}
                
                with edit_tab1:
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
                
                with edit_tab2:
                    col1, col2 = st.columns(2)
                    with col1:
                        updated_data['Monthly Sales Target'] = st.number_input("Monthly Target", 
                            value=float(rep_data['Monthly Sales Target']) if pd.notna(rep_data['Monthly Sales Target']) else 0.0)
                        updated_data['Actual Sales (Month)'] = st.number_input("Actual Sales", 
                            value=float(rep_data['Actual Sales (Month)']) if pd.notna(rep_data['Actual Sales (Month)']) else 0.0)
                        updated_data['Total Deals Closed (Month)'] = st.number_input("Deals Closed", 
                            value=int(rep_data['Total Deals Closed (Month)']) if pd.notna(rep_data['Total Deals Closed (Month)']) else 0)
                    with col2:
                        updated_data['Pipeline Value'] = st.number_input("Pipeline Value", 
                            value=float(rep_data['Pipeline Value']) if pd.notna(rep_data['Pipeline Value']) else 0.0)
                        updated_data['Win Rate (%)'] = st.number_input("Win Rate %", 
                            value=float(rep_data['Win Rate (%)']) if pd.notna(rep_data['Win Rate (%)']) else 0.0)
                        updated_data['Average Deal Size'] = st.number_input("Avg Deal Size", 
                            value=float(rep_data['Average Deal Size']) if pd.notna(rep_data['Average Deal Size']) else 0.0)
                
                with edit_tab3:
                    col1, col2 = st.columns(2)
                    with col1:
                        updated_data['Outbound Calls Made'] = st.number_input("Calls Made", 
                            value=int(rep_data['Outbound Calls Made']) if pd.notna(rep_data['Outbound Calls Made']) else 0)
                        updated_data['Outbound Emails Sent'] = st.number_input("Emails Sent", 
                            value=int(rep_data['Outbound Emails Sent']) if pd.notna(rep_data['Outbound Emails Sent']) else 0)
                        updated_data['Meetings Booked'] = st.number_input("Meetings Booked", 
                            value=int(rep_data['Meetings Booked']) if pd.notna(rep_data['Meetings Booked']) else 0)
                    with col2:
                        updated_data['Daily Activity Score'] = st.number_input("Activity Score", 
                            value=int(rep_data['Daily Activity Score']) if pd.notna(rep_data['Daily Activity Score']) else 0)
                        updated_data['Customer Satisfaction Score'] = st.number_input("CSAT Score", 
                            value=float(rep_data['Customer Satisfaction Score']) if pd.notna(rep_data['Customer Satisfaction Score']) else 0.0,
                            min_value=0.0, max_value=5.0, step=0.1)
                
                with edit_tab4:
                    col1, col2 = st.columns(2)
                    with col1:
                        updated_data['Base Salary'] = st.number_input("Base Salary", 
                            value=float(rep_data['Base Salary']) if pd.notna(rep_data['Base Salary']) else 0.0)
                        updated_data['Commission Rate'] = st.text_input("Commission Rate", 
                            value=str(rep_data['Commission Rate']))
                        updated_data['Commission Earned (Month)'] = st.number_input("Commission Earned", 
                            value=float(rep_data['Commission Earned (Month)']) if pd.notna(rep_data['Commission Earned (Month)']) else 0.0)
                    with col2:
                        updated_data['Bonuses Earned'] = st.number_input("Bonuses", 
                            value=float(rep_data['Bonuses Earned']) if pd.notna(rep_data['Bonuses Earned']) else 0.0)
                        updated_data['Total Compensation to Date'] = st.number_input("Total Compensation", 
                            value=float(rep_data['Total Compensation to Date']) if pd.notna(rep_data['Total Compensation to Date']) else 0.0)
                
                if st.form_submit_button("💾 Save Changes", type="primary"):
                    # Update the dataframe
                    for key, value in updated_data.items():
                        df.loc[df['Full Name'] == selected_rep, key] = value
                    
                    st.session_state.df = df
                    
                    # Save to Google Sheets if connected
                    if st.session_state.data_source == "gsheet" and st.session_state.gsheet_client:
                        try:
                            st.session_state.gsheet_client.clear()
                            data_to_upload = [df.columns.tolist()] + df.values.tolist()
                            st.session_state.gsheet_client.update('A1', data_to_upload)
                            st.success("✅ Changes saved to Google Sheets!")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                    else:
                        st.success("✅ Changes saved!")
                    
                    st.rerun()
    
    # TAB 5: Delete Rep
    with tab5:
        st.header("🗑️ Delete Sales Rep")
        
        st.warning("⚠️ Warning: This action cannot be undone!")
        
        rep_to_delete = st.selectbox("Select Rep to Delete:", df['Full Name'].tolist(), key="delete_select")
        
        if rep_to_delete:
            rep_info = df[df['Full Name'] == rep_to_delete].iloc[0]
            
            st.info(f"""
            **Rep Information:**
            - ID: {rep_info['Rep ID']}
            - Email: {rep_info['Email']}
            - Role: {rep_info['Role / Position']}
            - Territory: {rep_info['Territory / Region']}
            - Status: {rep_info['Employment Status']}
            """)
            
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                if st.button("🗑️ Delete Rep", type="primary"):
                    # Delete from dataframe
                    st.session_state.df = df[df['Full Name'] != rep_to_delete].reset_index(drop=True)
                    
                    # Save to Google Sheets if connected
                    if st.session_state.data_source == "gsheet" and st.session_state.gsheet_client:
                        try:
                            st.session_state.gsheet_client.clear()
                            data_to_upload = [st.session_state.df.columns.tolist()] + st.session_state.df.values.tolist()
                            st.session_state.gsheet_client.update('A1', data_to_upload)
                            st.success("✅ Rep deleted from Google Sheets!")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                    else:
                        st.success("✅ Rep deleted!")
                    
                    st.rerun()
    
    # TAB 6: Invoices
    with tab6:
        st.header("📄 Invoice Management")
        
        # Select rep
        rep_for_invoice = st.selectbox("Select Rep:", df['Full Name'].tolist(), key="invoice_select")
        
        if rep_for_invoice:
            rep_data = df[df['Full Name'] == rep_for_invoice].iloc[0]
            
            st.subheader(f"Invoices for {rep_for_invoice}")
            
            # Extract invoice columns
            invoice_cols = [col for col in df.columns if 'Invoice' in col]
            
            # Group invoices
            invoices = []
            for i in range(1, 11):
                inv_num_col = f'Invoice {i} Number'
                inv_date_col = f'Invoice {i} Date'
                inv_amt_col = f'Invoice {i} Amount'
                
                if all(col in df.columns for col in [inv_num_col, inv_date_col, inv_amt_col]):
                    invoices.append({
                        'Invoice #': rep_data[inv_num_col],
                        'Date': rep_data[inv_date_col],
                        'Amount': rep_data[inv_amt_col]
                    })
            
            if invoices:
                invoice_df = pd.DataFrame(invoices)
                st.dataframe(invoice_df, use_container_width=True, hide_index=True)
                
                # Summary
                try:
                    invoice_df['Amount'] = pd.to_numeric(invoice_df['Amount'], errors='coerce')
                    total_amount = invoice_df['Amount'].sum()
                    st.metric("Total Invoice Amount", f"${total_amount:,.2f}")
                except:
                    st.info("Unable to calculate total - check amount format")
                
                # Edit invoices
                with st.expander("✏️ Edit Invoices"):
                    edited_invoices = st.data_editor(
                        invoice_df,
                        use_container_width=True,
                        num_rows="dynamic",
                        key="invoice_editor"
                    )
                    
                    if st.button("💾 Save Invoice Changes"):
                        # Update the main dataframe
                        for idx, row in edited_invoices.iterrows():
                            if idx < 10:
                                df.loc[df['Full Name'] == rep_for_invoice, f'Invoice {idx+1} Number'] = row['Invoice #']
                                df.loc[df['Full Name'] == rep_for_invoice, f'Invoice {idx+1} Date'] = row['Date']
                                df.loc[df['Full Name'] == rep_for_invoice, f'Invoice {idx+1} Amount'] = row['Amount']
                        
                        st.session_state.df = df
                        st.success("✅ Invoice changes saved!")
                        st.rerun()

else:
    # Welcome screen
    st.markdown('<div class="main-header">📊 Sales Reps Management Dashboard</div>', unsafe_allow_html=True)
    
    st.info("👈 Please upload a CSV file or connect to Google Sheets from the sidebar to get started")
    
    st.markdown("""
    ### Features:
    
    - 📊 **Interactive Dashboard**: Visualize sales performance, territories, and KPIs
    - 👥 **Manage Sales Reps**: View, filter, and edit all sales rep data
    - ➕ **Add New Reps**: Easy form to add new team members
    - ✏️ **Edit Reps**: Update any field for existing reps
    - 🗑️ **Delete Reps**: Remove reps from the system
    - 📄 **Invoice Management**: Track and edit invoices for each rep
    - 🔄 **Live Sync**: Changes sync to Google Sheets in real-time
    - 💾 **Export Data**: Download updated CSV anytime
    
    ### Getting Started:
    
    1. Upload your CSV file OR
    2. Connect to Google Sheets with service account JSON
    3. Start managing your sales team!
    """)
    
    # Sample service account JSON structure
    with st.expander("📝 Sample Service Account JSON Structure"):
        st.code("""
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\\nYOUR_PRIVATE_KEY\\n-----END PRIVATE KEY-----\\n",
  "client_email": "your-service-account@your-project.iam.gserviceaccount.com",
  "client_id": "your-client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "your-cert-url"
}
        """, language="json")
