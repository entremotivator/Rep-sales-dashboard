"""
Sample Data Generator for Sales Dashboard
Generates realistic sample data for testing the dashboard
"""

import pandas as pd
import random
from datetime import datetime, timedelta
import sys

def generate_sample_data(num_reps=20):
    """Generate sample sales rep data"""
    
    # Sample names
    first_names = ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Riley', 'Casey', 'Jamie', 
                   'Avery', 'Drew', 'Sam', 'Blake', 'Cameron', 'Skyler', 'Dakota', 
                   'Reese', 'Quinn', 'Hayden', 'Rowan', 'Finley', 'Sage']
    
    last_names = ['Carter', 'Smith', 'White', 'Lee', 'Johnson', 'Parker', 'Brooks', 
                  'Green', 'Michaels', 'Torres', 'Rivers', 'Hayes', 'Collins', 
                  'Foster', 'Reed', 'Bailey', 'Cooper', 'Morgan', 'Murphy', 'Wright']
    
    roles = ['Field Rep', 'BDR', 'SDR', 'Account Executive', 'Sales Manager']
    territories = ['National', 'South', 'North', 'West Coast', 'Midwest', 'East Coast']
    statuses = ['Active', 'Inactive']
    contract_types = ['Full-Time', 'Part-Time', 'Contract']
    
    data = []
    
    for i in range(num_reps):
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        full_name = f"{first_name} {last_name}"
        
        # Generate realistic sales data
        monthly_target = random.randint(30000, 90000)
        actual_sales = int(monthly_target * random.uniform(0.3, 1.2))
        percent_to_goal = (actual_sales / monthly_target * 100) if monthly_target > 0 else 0
        
        deals_closed = random.randint(5, 30)
        avg_deal_size = int(actual_sales / deals_closed) if deals_closed > 0 else 0
        
        opportunities_created = random.randint(5, 30)
        opportunities_won = random.randint(1, min(10, opportunities_created))
        win_rate = (opportunities_won / opportunities_created * 100) if opportunities_created > 0 else 0
        
        # Generate start date (1-4 years ago)
        start_date = datetime.now() - timedelta(days=random.randint(365, 1460))
        
        rep_data = {
            'Rep ID': i + 1,
            'Full Name': full_name,
            'Email': f"{first_name.lower()}.{last_name.lower()}@company.com",
            'Phone': f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            'Role / Position': random.choice(roles),
            'Territory / Region': random.choice(territories),
            'Start Date': start_date.strftime('%Y-%m-%d'),
            'Employment Status': random.choice(statuses) if random.random() > 0.2 else 'Active',
            'Monthly Sales Target': monthly_target,
            'Actual Sales (Month)': actual_sales,
            '% to Goal': round(percent_to_goal, 1),
            'Total Deals Closed (Month)': deals_closed,
            'Average Deal Size': avg_deal_size,
            'Pipeline Value': random.randint(50000, 200000),
            'Win Rate (%)': round(win_rate, 1),
            'Outbound Calls Made': random.randint(50, 200),
            'Outbound Emails Sent': random.randint(100, 350),
            'Meetings Booked': random.randint(5, 25),
            'Meetings Completed': random.randint(3, 20),
            'New Leads Assigned': random.randint(20, 70),
            'Leads Contacted': random.randint(15, 60),
            'Follow-Ups Due': random.randint(5, 20),
            'Opportunities Created': opportunities_created,
            'Opportunities in Progress': random.randint(3, 15),
            'Opportunities Lost': random.randint(2, 10),
            'Opportunities Won': opportunities_won,
            'Base Salary': random.randint(55000, 90000),
            'Commission Rate': '10%',
            'Commission Earned (Month)': round(actual_sales * 0.10, 2),
            'Bonuses Earned': random.choice([0, 250, 500, 1000]),
            'Total Compensation to Date': random.randint(60000, 95000),
            'Daily Activity Score': random.randint(70, 100),
            'Last Login (CRM)': (datetime.now() - timedelta(days=random.randint(0, 7))).strftime('%Y-%m-%d'),
            'Notes Logged (Count)': random.randint(10, 45),
            'Tasks Completed': random.randint(15, 45),
            'Tasks Overdue': random.randint(0, 8),
            'Lead Response Time (Avg)': f"{random.randint(2, 6)} hours",
            'Customer Satisfaction Score': round(random.uniform(3.0, 5.0), 2),
            'Manager Name': 'Chris Johnson',
            'Training Completed': random.choice(['Yes', 'No']),
            'Certifications': 'Salesforce, HubSpot',
            'Contract Type': random.choice(contract_types),
            'Performance Review Date': (datetime.now() - timedelta(days=random.randint(30, 90))).strftime('%Y-%m-%d'),
            'HR Notes': random.choice(['Good performance', 'Needs improvement', 'Excellent team player', ''])
        }
        
        # Add invoices
        base_amount = 150.75
        for inv_num in range(1, 11):
            rep_data[f'Invoice {inv_num} Number'] = f'INV-{i+1}{inv_num:03d}'
            rep_data[f'Invoice {inv_num} Date'] = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
            rep_data[f'Invoice {inv_num} Amount'] = round(base_amount + (inv_num - 1) * 30.75, 2)
        
        data.append(rep_data)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate sample data
    num_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    
    print(f"Generating sample data for {num_reps} sales reps...")
    df = generate_sample_data(num_reps)
    
    # Save to CSV
    filename = f'sample_sales_data_{num_reps}_reps.csv'
    df.to_csv(filename, index=False)
    
    print(f"✅ Sample data saved to: {filename}")
    print(f"Total reps: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nFirst few rows:")
    print(df[['Rep ID', 'Full Name', 'Role / Position', 'Actual Sales (Month)']].head())
