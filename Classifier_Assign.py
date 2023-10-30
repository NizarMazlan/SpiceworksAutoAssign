import pandas as pd

def update_prediction_history(predictions):
    try:
        # Load the existing prediction history
        prediction_history_file = r'\\172.16.1.9\public\IT\Spiceworks\Auto Assign\Prediction_History.xlsx'
        prediction_history = pd.read_excel(r'\\172.16.1.9\public\IT\Spiceworks\Auto Assign\Prediction_History.xlsx')

        # Load the actual modules from ticket_export.csv
        actual_modules = pd.read_csv(r'\\172.16.1.9\public\IT\Spiceworks\ticket_export.csv')
        actual_modules = actual_modules[['Ticket Number', 'Sub Category']]  # Adjust columns as needed

        # Create a dictionary from 'actual_modules' for faster lookups
        actual_modules_dict = dict(zip(actual_modules['Ticket Number'], actual_modules['Sub Category']))

        # Update 'Sub Category' in 'Prediction_History' based on 'Ticket Number'
        prediction_history['Sub Category'] = prediction_history['Ticket Number'].map(actual_modules_dict)

        # Compare predicted and actual modules
        prediction_history['Result'] = prediction_history.apply(lambda row:
            'Blank' if pd.isna(row['Sub Category'])
            else 'Correct' if row['Sub Category'] == row['predicted_modules']
            else 'Incorrect', axis=1)

        # Remove rows from 'predictions' that already exist in 'prediction_history'
        predictions_to_add = predictions[~predictions['Ticket Number'].isin(prediction_history['Ticket Number'])]

        # Append the new predictions
        prediction_history = pd.concat([prediction_history, predictions_to_add], ignore_index=True)

        # Save the updated prediction history back to the Excel file
        prediction_history.to_excel(prediction_history_file, index=False)
        print('Prediction history updated successfully.')

    except Exception as e:
        print('Error updating prediction history:', str(e))


def classify():
    try:
        Tickets_to_Assign = pd.read_csv(r'\\172.16.1.9\public\IT\Spiceworks\ticket_export.csv')
        ListName = pd.read_excel(r'\\172.16.1.9\public\IT\Spiceworks\Auto Assign\IT Functional.xlsx')
        Tickets_to_Assign.drop([0,1], axis=0, inplace=True)
        columns_to_be_dropped_model = ['Category','Closed On','Created By','Due On','Priority','Organization Name','Time Spent','Time To Resolve','Organization Host','Link to Ticket','SAP Module','Module Remark','Cr Status','Cr No','Cr','Transport No','Ams','Site','Sub Category']
        Tickets_to_Assign.drop(columns_to_be_dropped_model, axis=1, inplace=True)

        # Filtering to get Open and Unassigned Tickets
        openTickets = Tickets_to_Assign.loc[Tickets_to_Assign['Status'].str.contains('open$')]
        openUnassignedTicket = openTickets[openTickets['Assigned To'].isnull()]

        # Load model
        import pickle
        filename = r'\\172.16.1.9\public\IT\Spiceworks\Auto Assign\best_model_SVM_20ONWARDS.sav'
        loaded_model = pickle.load(open(filename,'rb'))

        # List of words to remove to prevent interuppting current RPA 
        words_to_remove = ['SAP Password','Reset Password','Password Reset', 'Passwort zurücksetzen', '重设密码', 'Passwort Reset']

        # Filter the DataFrame to exclude rows with the specified words in the 'summary' column
        openUnassignedTicket = openUnassignedTicket[~openUnassignedTicket['Summary'].str.contains('|'.join(words_to_remove), case=False, na=False)]

        # Use summary
        smry = openUnassignedTicket["Summary"]


        # Predict the labels
        predicted_labels = loaded_model.predict(smry)
        con = loaded_model.predict_proba(smry)
        confidence_score = con.max(axis=1)
 
        # Create a new column for the predicted labels
        openUnassignedTicket['predicted_modules'] = predicted_labels
        openUnassignedTicket['confidence'] = confidence_score
        
        # Function to update the counter for a given personnel email
        def update_counter(email, increment):
            ListName.loc[ListName['email'] == email, 'counter'] += increment
            ListName.to_excel(r'\\172.16.1.9\public\IT\Spiceworks\Auto Assign\IT Functional.xlsx', index=False)  # Save the updated dataframe to the XLSX file

        # Get IT Personnel that is inside the Module Group
        for index, row in openUnassignedTicket.iterrows():
            modules = row['predicted_modules']
            filterByModule = ListName[ListName['module'].str.contains(modules)]

            # Sort the personnel by counter in ascending order (lowest counter first)
            filterByModule = filterByModule.sort_values(by='counter', ascending=True)

            # Choose the personnel with the lowest counter
            rand = filterByModule.iloc[0]
            email = rand["email"]
            name = rand["name"]

            openUnassignedTicket.at[index, 'assign_to'] = email
            openUnassignedTicket.at[index, 'assignee_name'] = name

            # Update the counter for the selected personnel
            increment_value = 1  # Increment by 1 (you can change this as needed)
            update_counter(email, increment_value)

            # Add a new column combining ticket number and summary
            ticket_number = row['Ticket Number']
            summary = row['Summary']
            combined_text = f"[Ticket #{ticket_number}] {summary}"
            openUnassignedTicket.at[index, 'ticket_summary_combined'] = combined_text
        
        # write DataFrame to excel
        openUnassignedTicket.to_excel(r'\\172.16.1.9\public\IT\Spiceworks\Auto Assign\Prediction_Result.xlsx', index=False)

        # save the excel
        print('DataFrame is written to Excel File successfully.')

        # Update the prediction history with the new predictions
        update_prediction_history(openUnassignedTicket)

    except Exception as e:
        print("Error or There is no ticket unassigned")
        print(e)
        
        # Create an empty DataFrame with the desired columns
        columns = [
            'Ticket Number', 'Summary', 'Description', 'Assigned To',
            'Created On', 'Status', 'predicted_modules', 'confidence',
            'assign_to', 'assignee_name', 'ticket_summary_combined'
        ]

        empty_df = pd.DataFrame(columns=columns)

        # Save the empty DataFrame to an Excel file
        empty_df.to_excel(r'\\172.16.1.9\public\IT\Spiceworks\Auto Assign\Prediction_Result.xlsx', index=False)

        # Update the prediction history with the new predictions
        update_prediction_history(empty_df)

classify()