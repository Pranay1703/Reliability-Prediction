import pandas as pd

def Preprocessing_Hazard(raw_data):
    # Formatting Dates
    raw_data['Sales Month'] = pd.to_datetime(raw_data['Sales Month'], format="%m.%Y")
    raw_data['Production Month'] = pd.to_datetime(raw_data['Production Month'], format="%m.%Y")
    raw_data['Comp Report Dt'] = pd.to_datetime(raw_data['Comp Report Dt'], format="%d.%m.%Y")

    # Concatenate DCC, KMs and Chassis for duplicate Removal
    raw_data['Concatenated'] = raw_data['Chassis no.'].astype(str) + raw_data['Kilometers'].astype(str) + raw_data[
        'Dummy complaint code'].astype(str)

    # Creating a table for concatenated text wise sum of total expenses.
    Chassis_wise_Total_expense = raw_data.groupby('Concatenated')['Total Expenses'].sum()
    raw_data = raw_data.merge(Chassis_wise_Total_expense, left_on='Concatenated', right_on='Concatenated', how='left')
    raw_data.drop('Total Expenses_x', axis=1, inplace=True)

    # Removing Duplicates
    raw_data = raw_data.drop_duplicates(subset='Concatenated')
    raw_data = raw_data.rename(columns={'Total Expenses_y': 'Total Expenses'})

    # Calculating new EPV basis sum of total expenses and overriding Old EPV
    raw_data['EPV'] = raw_data['Total Expenses'] / raw_data['Sales Qty']

    # Identifying Dealer Details and Region through Dealer Details.'
    # raw_data = raw_data.merge(dealer_details[['Dealer Code','Dealer_State','Dealer_Region']], left_on = 'Dealer Code', right_on ='Dealer Code', how = 'left')

    # Calculating Idle Months
    raw_data['Idle days'] = raw_data['Sales Month'] - raw_data['Production Month']
    raw_data['Idle days'] = raw_data['Idle days'].map(lambda x: np.nan if pd.isnull(x) else x.days)
    # raw_data['Idle days']
    raw_data['Idle Months'] = round(raw_data['Idle days'] / 30)
    raw_data['Idle Months']

    Months_list = raw_data['Production Month'].unique()

    for model in raw_data['Model'].unique():
        for sub_model in raw_data['Sub Model'][raw_data['Model'] == model].unique(): \
                for
        complaint_code in raw_data['Dummy complaint code'][
            raw_data['Model'] == model & & raw_data['Sub Model'] == sub_model].unique()
        for month in Months_list:
            if month not in raw_data['Production Month'][
                raw_data['Dummy complaint code'] == complaint_code & & raw_data['Model'] == model & & raw_data[
                    'Sub Model'] == sub_model].unique()
                raw_data['Complaints'][raw_data['Model'] == model & & raw_data['Sub Model'] == sub_model & & raw_data[
                    'Dummy complaint code'] == complaint_code & & raw_data['Production Month'] == month] = 0


# raw_data.to_excel("Duplicate_Removed.xlsx")
return raw_data