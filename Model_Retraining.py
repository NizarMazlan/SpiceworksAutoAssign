def build_model():
    import pandas as pd
    # Read data from Excel files
    Tickets = pd.read_excel('C:\\Users\\nizarmustaqeem.m\\OneDrive - KUALA LUMPUR KEPONG BERHAD\\Documents\\Internship\\DS PROJECT\\SPICEWORKS PROJECT\\Finalized\\ticket_export.xlsx','Tickets')
    Ticket_category = pd.read_excel('C:\\Users\\nizarmustaqeem.m\\OneDrive - KUALA LUMPUR KEPONG BERHAD\\Documents\\Internship\\DS PROJECT\\SPICEWORKS PROJECT\\Finalized\\ticket_export.xlsx','Ticket Categories')
    Users = pd.read_excel('C:\\Users\\nizarmustaqeem.m\\OneDrive - KUALA LUMPUR KEPONG BERHAD\\Documents\\Internship\\DS PROJECT\\SPICEWORKS PROJECT\\Finalized\\ticket_export.xlsx','Users')
    Ticket_custom_attributes = pd.read_excel('C:\\Users\\nizarmustaqeem.m\\OneDrive - KUALA LUMPUR KEPONG BERHAD\\Documents\\Internship\\DS PROJECT\\SPICEWORKS PROJECT\\Finalized\\ticket_export.xlsx','Tickets to Custom Attributes')
    End_users = pd.read_excel('C:\\Users\\nizarmustaqeem.m\\OneDrive - KUALA LUMPUR KEPONG BERHAD\\Documents\\Internship\\DS PROJECT\\SPICEWORKS PROJECT\\Finalized\\ticket_export.xlsx','End Users')

    # Dropping first 2 rows (Welcoming Ticket)
    Tickets.drop([0,1], axis=0, inplace=True)

    # pivot the DataFrame
    Ticket_custom_attributes = Ticket_custom_attributes.pivot_table(index='ticket_id', columns='name', values='value', aggfunc='first').reset_index()

    # fill NaN values with "nothing"
    Ticket_custom_attributes = Ticket_custom_attributes.fillna('No')

    # reorder the columns
    Ticket_custom_attributes = Ticket_custom_attributes[['ticket_id' ,'cr_status' ,'ams', 'cr', 'site', 'sub_category']]

    # Ticket Category
    # Create a dictionary mapping ticket_category_id to ticket name
    ticket_id_to_name = dict(zip(Ticket_category['ticket_category_id'], Ticket_category['name']))

    # Replace ticket_category_id with ticket name in df_cleaned
    Tickets['ticket_category_id'] = Tickets['ticket_category_id'].map(ticket_id_to_name)


    # Assignee ID
    # Create new column with full name 
    Users['Name'] = Users['first_name'].str.cat(Users['last_name'],sep=" ")

    # Create a dictionary mapping ticket_category_id to ticket name
    assignee_id_to_name = dict(zip(Users['user_id'], Users['first_name']))

    # Replace assignee_id with user name in df_cleaned
    Tickets['assignee_id'] = Tickets['assignee_id'].map(assignee_id_to_name)

    # Ticket to Custom Attributes
    # Merge the two dataframes based on ticket_id column
    merged_Tickets = pd.merge(Tickets, Ticket_custom_attributes, on='ticket_id', how='left')

    # End Users
    # Create a dictionary mapping ticket_category_id to ticket name
    enduser_id_to_name = dict(zip(End_users['end_user_id'], End_users['email']))
    merged_Tickets['end_user_id'] = merged_Tickets['end_user_id'].map(enduser_id_to_name)

    # Replace ticket_category_id with ticket name in df_cleaned
    ticket_id_to_name = dict(zip(Ticket_category['ticket_category_id'], Ticket_category['name']))
    merged_Tickets['ticket_category_id'] = merged_Tickets['ticket_category_id'].map(enduser_id_to_name)

    # Dropping Unecessary Columns
    columns_to_be_dropped = ['organization_id','priority','close_time_secs','desktop_id','ticket_rules_enabled','type','status','due_at','status_changed_at','import_id','updated_at','first_response_secs','last_user_response_at','email_message_id','master_ticket_number','alerted','muted']
    merged_Tickets.drop(columns_to_be_dropped, axis=1, inplace=True)

    # fill NaN values with "No"
    merged_Tickets["ams"] = merged_Tickets["ams"].fillna('No')
    merged_Tickets["cr"] = merged_Tickets["cr"].fillna('No')
    merged_Tickets["site"] = merged_Tickets["site"].fillna('No')
    merged_Tickets["sub_category"] = merged_Tickets["sub_category"].fillna('No')
    merged_Tickets = merged_Tickets.astype(str)

    # Removing Unecessary Tickets Category
    merged_Tickets = merged_Tickets.drop(merged_Tickets[merged_Tickets.sub_category == 'No'].index)
    merged_Tickets = merged_Tickets.drop(merged_Tickets[merged_Tickets.sub_category == 'Delete'].index)
    merged_Tickets = merged_Tickets.drop(merged_Tickets[merged_Tickets.sub_category == 'BI'].index)
    merged_Tickets = merged_Tickets.drop(merged_Tickets[merged_Tickets.sub_category == 'Test'].index)

    # Filter out only tickets after 2020
    merged_Tickets['created_at'] = pd.to_datetime(merged_Tickets['created_at'])
    merged_Tickets_2020onwards = merged_Tickets[merged_Tickets['created_at'].dt.year >= 2020]

    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.stem import PorterStemmer
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    import re

    # Clean basic characters
    def clean(raw):
        """ Remove hyperlinks and markup """
        result = re.sub("<[a][^>]*>(.+?)</[a]>", 'Link.', raw)
        result = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+",' ',result)
        result = re.sub('&gt;', "", result)
        result = re.sub('&#x27;', "'", result)
        result = re.sub('&quot;', '"', result)
        result = re.sub('&#x2F;', ' ', result)
        result = re.sub('<p>', ' ', result)
        result = re.sub('</i>', '', result)
        result = re.sub('&#62;', '', result)
        result = re.sub('<i>', ' ', result)
        result = re.sub("\n", '', result)
        return result

    # Clean numerics
    def remove_num(texts):
        output = re.sub(r'\d+', '', texts)
        return output

    # Clean emojis
    def deEmojify(x):
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags = re.UNICODE)
        return regrex_pattern.sub(r'', x)

    # Clean and Unify Whitespace
    def unify_whitespaces(x):
        cleaned_string = re.sub(' +', ' ', x)
        return cleaned_string 


    # Clean and Remove Symbols
    def remove_symbols(x):
        cleaned_string = re.sub(r"[^a-zA-Z0-9?!.,]+", ' ', x)
        return cleaned_string

    # Clean and Remove Punctuation
    def remove_punctuation(text):
        final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"',','))
        return final

    # Remove Stopwords
    custom_stop_words = ['http', 'www','gt gt','com','u','T','mailto','f',"com", "selangor", "malaysia", "t", "did", "f", "klk","sent","petaling jaya","damansara petal", "klk","jalan","sdn bhd","mutiara damansara"]
    stop=set(stopwords.words("english") + custom_stop_words)
    stemmer=PorterStemmer()
    lemma=WordNetLemmatizer()

    def remove_stopword(text):
        text=[word.lower() for word in text.split() if word.lower() not in stop]
        return " ".join(text)

    def Lemmatization(text):
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words('english')
        stop_words.extend(['br', 'br'])  # Add additional stop words if needed
        
        word_tokens = nltk.word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in word_tokens if word.lower() not in stop_words]
        lemmatized_text = ' '.join(lemmatized_words)
        
        return lemmatized_text


    def cleaning(df,review):  
        #Remove Duplicates and NaN Values
        #df.drop_duplicates(inplace=True)
        df.dropna(how='any',inplace=True)

        df[review] = df[review].apply(clean)
        df[review] = df[review].apply(deEmojify)
        df[review] = df[review].str.lower() #lowercase
        df[review] = df[review].apply(remove_num)
        df[review] = df[review].apply(remove_symbols)
        df[review] = df[review].apply(remove_punctuation)
        df[review] = df[review].apply(remove_stopword)
        df[review] = df[review].apply(unify_whitespaces)
        df[review] = df[review].apply(Lemmatization)

    cleaning(merged_Tickets_2020onwards,'summary')

    # Convert 'created_at' column to datetime format
    merged_Tickets_2020onwards['created_at'] = pd.to_datetime(merged_Tickets_2020onwards['created_at'])

    # create a dictionary to map assignee names to labels
    assignee_modules = {'WM': 'WM','CDT': 'CDT','SD': 'SD','MM': 'MM','FI': 'FI','CO': 'CO','QM': 'QM','PP': 'PP','PM': 'PM','Basis': 'Basis','SAP-GUI': 'Basis','Authorizations': 'Basis','Network': 'Basis','Interfaces': 'Basis'}

    # use apply() with a lambda function to create a new column based on the assignee name
    merged_Tickets_2020onwards['module'] = merged_Tickets_2020onwards['sub_category'].apply(lambda x: assignee_modules.get(x))

    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    # Split the data into training and testing sets
    X = merged_Tickets_2020onwards['summary']
    y = merged_Tickets_2020onwards['module']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    corpus = X_train

    # Initizalize the vectorizer with max nr words and ngrams (1: single words, 2: two words in a row)
    vectorizer_tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2))
    # Fit the vectorizer to the training data
    vectorizer_tfidf.fit(corpus)
    TfidfVectorizer(max_features=15000, ngram_range=(1, 2))

    # Define the pipeline
    pipeline = Pipeline([
        ("vectorizer", vectorizer_tfidf),
        ("classifier", SVC(kernel='linear', probability=True))
    ])

    # Define the hyperparameter grid for GridSearchCV
    param_grid = {
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': [0.1, 1, 10]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model_SVM = grid_search.best_estimator_

    # Make predictions on the testing set
    y_pred = best_model_SVM.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    classification_report = classification_report(y_test, y_pred)

    
    print('Best Parameters:', grid_search.best_params_)
    print('Accuracy:', accuracy)
    print('Classification Report:')
    print(classification_report)

    import pickle
    pickle.dump(best_model_SVM,open(r"\\172.16.1.9\public\IT\Spiceworks\Auto Assign\best_model_SVM_20ONWARDS.sav", "wb"))

build_model()