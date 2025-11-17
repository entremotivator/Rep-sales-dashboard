# Advanced Sales Management Dashboard

A comprehensive Streamlit-based dashboard for managing sales representatives with real-time data editing, Google Sheets integration, and advanced analytics.

## Features

### Core Functionality
- **Live Data Editing**: Edit any column in real-time with instant updates
- **Google Sheets Integration**: Two-way sync with Google Sheets using service account authentication
- **CSV Import/Export**: Upload CSV files and export data anytime
- **Invoice Management**: Track and manage up to 10 invoices per sales rep

### Analytics & Insights
- **Executive Dashboard**: High-level KPIs, charts, and performance metrics
- **Advanced Analytics**: Correlation analysis, trend analysis, comparative analysis
- **Predictive Insights**: AI-powered recommendations and projections
- **Performance Tracking**: Detailed performance scoring and radar charts
- **Team Overview**: Role, territory, and status distributions

### Data Management
- **Add/Edit/Delete Reps**: Full CRUD operations for sales representatives
- **Bulk Operations**: Update multiple reps simultaneously
- **Advanced Filtering**: Filter by status, role, territory, and search by name/email
- **Column Selection**: Choose which columns to display and edit
- **Auto-sync**: Optional automatic synchronization with Google Sheets

### Reports & Export
- **Custom Reports**: Generate reports with selected columns
- **Multiple Formats**: Export as CSV, Excel (XLSX), or JSON
- **Pre-built Reports**: Performance, Sales, Activity, and Compensation reports

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Step 2: Run the Dashboard

\`\`\`bash
streamlit run scripts/advanced_sales_dashboard.py
\`\`\`

The dashboard will open automatically in your default web browser at `http://localhost:8501`

## Usage

### Option 1: CSV Upload

1. Click on "CSV File" in the sidebar
2. Upload your CSV file using the file uploader
3. Your data will load automatically
4. Start managing your sales team!

### Option 2: Google Sheets Integration

#### Setup Google Cloud Service Account

1. **Create Google Cloud Project**
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Note your project ID

2. **Enable Google Sheets API**
   - In the Google Cloud Console, go to "APIs & Services" → "Library"
   - Search for "Google Sheets API"
   - Click "Enable"

3. **Create Service Account**
   - Go to "IAM & Admin" → "Service Accounts"
   - Click "Create Service Account"
   - Enter a name (e.g., "sales-dashboard-service")
   - Grant it the "Editor" role
   - Click "Done"

4. **Generate JSON Key**
   - Click on your newly created service account
   - Go to the "Keys" tab
   - Click "Add Key" → "Create new key"
   - Select "JSON" as the key type
   - Click "Create" and download the JSON file
   - **Keep this file secure!**

5. **Share Your Google Sheet**
   - Open your Google Sheet
   - Click the "Share" button
   - Copy the service account email from the JSON file (format: `your-service-account@your-project.iam.gserviceaccount.com`)
   - Paste it in the "Add people and groups" field
   - Set permission to "Editor"
   - Uncheck "Notify people"
   - Click "Share"

#### Connect to Dashboard

1. In the sidebar, select "Google Sheets"
2. Upload your service account JSON file
3. Paste your Google Sheets URL (use the provided default or your own)
4. Click "Connect to Google Sheets"
5. Your data will load automatically with live sync enabled!

## Dashboard Sections

### 1. Executive Dashboard
- Key performance metrics and KPIs
- Sales performance charts (Target vs Actual)
- Territory distribution pie chart
- Win rate by role analysis
- Activity score distribution
- Top performers and attention areas

### 2. Analytics & Insights
- **Correlation Analysis**: Matrix showing relationships between metrics
- **Trend Analysis**: Activity vs Performance scatter plots, Pipeline analysis
- **Comparative Analysis**: Group metrics by territory, role, or other dimensions
- **Predictive Insights**: AI-powered recommendations based on current performance

### 3. Team Overview
- Team composition statistics (Active/Inactive, Full-Time/Part-Time)
- Distribution charts by role, territory, and status
- Detailed roster with performance levels
- Training and certification tracking

### 4. Performance Tracking
- Comprehensive filters for drilling down
- Sales, activity, lead, and opportunity metrics
- Individual performance deep dive with radar charts
- Sales funnel visualization
- Performance comparison vs team average

### 5. Data Management
- Advanced filtering and search capabilities
- Column selection for focused editing
- Live data editor with validation
- Bulk update operations
- Export filtered data

### 6. Add New Rep
- Comprehensive form organized by sections
- Auto-generates Rep ID
- All standard fields plus custom notes
- Instant sync to Google Sheets (if connected)

### 7. Edit Rep
- **Quick Edit Mode**: Fast updates of key fields
- **Full Edit Mode**: All fields organized in tabs (Personal, Sales, Activity, Lead & Opportunities, Compensation, Other)
- Real-time validation and auto-calculations
- Immediate sync

### 8. Invoice Management
- View all 10 invoices per rep
- Invoice summary with totals and averages
- Interactive charts
- Inline editing
- Export individual rep invoices

### 9. Reports & Export
- Pre-built report templates (Full Team, Performance, Sales, Activity, Compensation)
- Custom report builder with column selection
- Export formats: CSV, Excel, JSON
- Report statistics and preview

## Data Structure

The dashboard expects CSV/Google Sheets data with the following columns:

### Required Columns
- Rep ID
- Full Name
- Email
- Phone
- Role / Position
- Territory / Region
- Employment Status
- Monthly Sales Target
- Actual Sales (Month)

### Optional Columns (Auto-calculated if missing)
- % to Goal
- Total Deals Closed (Month)
- Average Deal Size
- Pipeline Value
- Win Rate (%)
- Outbound Calls Made, Outbound Emails Sent
- Meetings Booked, Meetings Completed
- Lead metrics, Opportunity metrics
- Compensation details
- Activity Score, CSAT Score
- Invoice columns (Invoice 1-10 Number, Date, Amount)

## Performance Scoring

The dashboard automatically calculates a Performance Score based on:
- **Sales Goal Achievement** (30%): % to Goal performance
- **Win Rate** (25%): Conversion success rate
- **Activity Score** (25%): Daily activity and engagement
- **Customer Satisfaction** (20%): CSAT score

Performance Levels:
- **Excellent** (90-100): 🌟
- **Good** (75-89): ✅
- **Average** (60-74): ⚠️
- **Needs Improvement** (<60): ❌

## Tips & Best Practices

### Data Management
- Enable "Auto-sync" for real-time updates to Google Sheets
- Use bulk operations for updating multiple reps at once
- Filter data before editing to focus on specific segments
- Download backups regularly using the export feature

### Performance Tracking
- Review the Executive Dashboard daily for team overview
- Use the Analytics tab weekly for deeper insights
- Track individual performance monthly using Performance Tracking
- Act on AI-powered recommendations promptly

### Google Sheets Integration
- Keep your service account JSON file secure
- Don't share the JSON file publicly
- Regularly review Google Sheet access permissions
- Test connection with a small dataset first

### Invoice Management
- Update invoice data regularly
- Use consistent date formats (YYYY-MM-DD)
- Verify invoice amounts before saving
- Export invoice reports for accounting

## Generate Sample Data

To generate sample data for testing:

\`\`\`bash
python scripts/sample_data_generator.py 25
\`\`\`

This creates a CSV file with 25 sample sales reps with realistic data.

## Troubleshooting

### Google Sheets Connection Issues

**Problem**: "Unable to connect to Google Sheets"
- **Solution**: Verify that the Google Sheets API is enabled in your Google Cloud project
- **Solution**: Ensure the service account email has Editor access to the sheet
- **Solution**: Check that the JSON file is valid and not corrupted

**Problem**: "Permission denied"
- **Solution**: Share the Google Sheet with the service account email
- **Solution**: Grant "Editor" permissions, not just "Viewer"

**Problem**: "Sheet not found"
- **Solution**: Verify the Google Sheets URL is correct
- **Solution**: Ensure you're using the full URL, not a shortened link

### Data Issues

**Problem**: "Data not loading"
- **Solution**: Check CSV format and ensure headers match expected columns
- **Solution**: Verify there are no empty rows at the top of the sheet
- **Solution**: Ensure numeric columns contain valid numbers

**Problem**: "Changes not syncing"
- **Solution**: Enable "Auto-sync" in the dashboard
- **Solution**: Click "Save Changes" manually if auto-sync is disabled
- **Solution**: Check internet connection for Google Sheets sync

### Performance Issues

**Problem**: "Dashboard running slowly"
- **Solution**: Filter data to reduce the number of visible rows
- **Solution**: Disable auto-sync if not needed
- **Solution**: Close other browser tabs to free up memory

## Security Considerations

- **Service Account JSON**: Never commit this file to version control (added to .gitignore)
- **Sensitive Data**: Consider encrypting sensitive compensation data
- **Access Control**: Limit Google Sheet access to authorized users only
- **Backups**: Regularly export and backup your data

## File Structure

\`\`\`
.
├── scripts/
│   ├── advanced_sales_dashboard.py    # Main dashboard application
│   └── sample_data_generator.py       # Sample data generator
├── .streamlit/
│   └── config.toml                    # Streamlit configuration
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
\`\`\`

## Credits

Built with:
- [Streamlit](https://streamlit.io/) - Web application framework
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [gspread](https://gspread.readthedocs.io/) - Google Sheets integration
- [NumPy](https://numpy.org/) - Numerical computing

---

**Version**: 2.0  
**Last Updated**: 2024  
**Created for**: Advanced Sales Team Management
