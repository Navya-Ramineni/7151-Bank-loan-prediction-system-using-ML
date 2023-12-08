import streamlit as st
import mysql.connector
import hashlib
import numpy as np
import pickle
import pandas as pd
df_train = pd.read_csv('C:/Users/krish/Loandata_train.csv')
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.pipeline import Pipeline
encoder = LabelEncoder()
scaler = MinMaxScaler(feature_range=(0,1))
scaler1 = scaler.fit(np.array(df_train.ApplicantIncome).reshape(-1,1))
scaler2 = scaler.fit(np.array(df_train.CoapplicantIncome).reshape(-1,1))


# Database initialization
conn = mysql.connector.connect(
    host='Localhost',
    user='root',
    #password='Admin@1234',
    database='new_schema'
)
cursor = conn.cursor()

#Create tables if they don't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        password VARCHAR(255) NOT NULL,
        is_admin BOOLEAN NOT NULL
    )
''')
conn.commit()



        
# Streamlit UI
def main():
    st.title("Bank loan Application")




    # Navigation
    menu = ["Sign Up", "Sign In"]
    choice = st.sidebar.selectbox("Menu", menu)

    if  choice == "Sign Up":
        #image_path = "Downloads/bank.jpg"
        #st.image(image_path, caption='MY Bank App', use_column_width=True)
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        is_admin = st.checkbox("Admin Account")
        signup_button = st.button("Sign Up")

        if signup_button:
            if not user_exists(new_username):
                hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
                register_user(new_username, hashed_password, is_admin)
                st.success("Registration successful! You can now sign in.")
            else:
                st.error("Username already exists. Please choose a different one.")

    elif choice == "Sign In":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        login_button = st.sidebar.checkbox("Sign In")

        if login_button:
            if login(username, password):
                st.subheader(f"Hello,{username}, You need to fill all necessary informations    to get a reply for your loan request !")
                loaded_model = pickle.load(open('C:/Users/krish/loan_model.pkl', 'rb'))
                
                def loan_prediction(input_data):
                    
                    

                    # changing the input_data to numpy array
                    input_data_as_numpy_array = np.asarray(input_data)

                    # reshape the array as we are predicting for one instance
                    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

                    prediction = loaded_model.predict(input_data_reshaped)
                    
                    print(prediction)                    

                    if (prediction == 0):
                      return 'Sorry,According to our Calculations, you will not eligible for bank loan          \n  It may be rejected due to Your Loan amount & mortgage due more than your income'
                     
                    else:
                      return 'Congratulations!! you are eligible for Bank Loan'
                # def calculate_debt_to_income_ratio(LoanAmount, MortgageDue, ApplicantIncome, Co_applicantIncome):
                #     total_debit = LoanAmount + MortgageDue
                #     total_income = ApplicantIncome + Co_applicantIncome
                #     if total_income != 0:
                #         debitIR = total_debit / total_income
                #         return debitIR
                #     else:
                #         return None
                  
                col1, col2, col3 = st.columns(3)
                with col1:
                        option1 = st.selectbox('Select Gender type', ('male', 'female', 'Other'))                        
                with col2:
                        option2 = st.selectbox('Select Education', ('Graduate', 'Not Graduate'))
            
                with col3:
                        option3 = st.selectbox('Select Marital status', ('single', 'married'))
                with col1:
                        option4 = st.selectbox('Select Job type', ('ProfExe', 'Office', 'Manager','Sales', 'Self','Other'))
                with col2:
                        option5 = st.selectbox('Select employment status', ('employed', 'self-employed'))
                with col3:
                        YearsInJob = st.text_input('No of Years in Job')
                with col1:
                      ApplicantIncome = st.text_input('ApplicantIncome') 
                with col2:
                      Co_applicantIncome = st.text_input('Co-ApplicantIncome') 
                with col3:
                      debitIR = st.text_input('Debit to Income ratio') 
                     
                         #st.text_input('Debit to Income ratio', value=str(debitIR), key='Debit to Income ratio', disabled=True)
                with col1:
                    LoanAmount = st.text_input('Loan Amount')
                with col2:
                    MortgageDue = st.text_input('Mortgage Due')
                JOB =1
                if (option1=='ProfExe'):
                    JOB = 3
                elif(option1=='Office'):
                    JOB = 1
                elif (option1=='Manager'):
                    JOB = 0
                elif(option1=='Sales'):
                    JOB = 4
                elif(option1=='Self'):
                    JOB = 5
                elif(option1=='Other'):
                    JOB = 2
                with col3:
                    LoanTerm = st.text_input('Loan Term in months')
                with col1:
                    PropertyValue = st.text_input('Property Value')
                with col2:
                    Credit_cards = st.text_input('No of Credit enquires')
                with col3:
                    credit_lines = st.text_input('Credit score')
                prediction = ''
                if st.button('Loan Test Result'):
                    sql= "insert into loan_application(Username,ApplicantIncome,MortgageDue,LoanAmount) values(%s,%s,%s,%s)"
                    val= (username,ApplicantIncome,MortgageDue,LoanAmount)
                    cursor.execute(sql,val)
                    conn.commit()                    
                    ApplicantIncome = float(ApplicantIncome)
                    ApplicantIncome = scaler1.transform(np.array(ApplicantIncome).reshape(-1,1))
                    prediction = loan_prediction([MortgageDue, PropertyValue,
                                                     LoanAmount, ApplicantIncome, JOB, YearsInJob,
                                                     debitIR,Credit_cards, credit_lines])
                    
                st.success(prediction)
           
        #admin page starts here            
        
        if is_admin_user(username):  
                    st.sidebar.checkbox("Admin")
                    st.info("You are logged in as admin.")
                    #Admin update the loan status
                    def update_approval_status(connection, application_id, approved):
                        cursor = connection.cursor()
                        query = "UPDATE loan_application SET approved = %s WHERE id = %s"
                        values = (approved, application_id)
                        cursor.execute(query, values)
                        connection.commit()
                    def get_loan_applications(connection):
                        cursor = connection.cursor()
                        cursor.execute("SELECT * FROM loan_application")
                        column_names = [i[0] for i in cursor.description]
                        data = cursor.fetchall()
                        return column_names,data
                    def get_loan_status(approved):
                        return "Approved" if approved else "Rejected" if approved is False else "Pending"
                    
                    st.subheader('User Loan Application details')
                    connection = mysql.connector.connect(
                        host='Localhost',
                        user='root',
                        #password='Admin@1234',
                        database='new_schema'
                    )                    
                    if connection is not None:
                        column_names,loan_applications = get_loan_applications(conn)                        
                        st.table([column_names] + list(loan_applications))
                        
                    #admin can allow to update the loan status    
                    st.subheader('Update Loan Status')    
                    selected_application_id = st.number_input('Enter Application ID:', step=1, value=1)
                    approval_status = st.radio('Select Approval Status:', ['Pending', 'Approved', 'Rejected'])                    
                    if st.button('Update Approval Status'):
                        #approved = True if approval_status == 'Approved' else False
                        update_approval_status(connection, selected_application_id, approval_status)
                        st.success(f'Approval status for Application ID {selected_application_id} updated successfully!')
                    
        else:
                        user_panel(username)

# Function to check if the user exists
def user_exists(username):
    cursor.execute('SELECT * FROM users WHERE username=%s', (username,))
    return cursor.fetchone() is not None

# Function to register a new user
def register_user(username, password, is_admin):
    cursor.execute('INSERT INTO users (username, password, is_admin) VALUES (%s, %s, %s)',
                   (username, password, is_admin))
    conn.commit()

# Function to check if the entered credentials are valid
def login(username, password):
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute('SELECT * FROM users WHERE username=%s AND password=%s', (username, hashed_password))
    return cursor.fetchone() is not None
    st.experimental_rerun()

# Function to check if the user is an admin
def is_admin_user(username):
    cursor.execute('SELECT is_admin FROM users WHERE username=%s', (username,))
    result = cursor.fetchone()
    return result[0] if result else False

# Admin panel
def admin_panel():
    st.subheader("Admin Panel")
    st.write("This is the admin panel.")

# User panel
def user_panel(username):
     st.subheader("")


    
# def main_page():
#     st.title("Main Page")
#     st.write("Welcome to the main page!")
#     if st.session_state.authenticated:
#     main_page()

if __name__ == '__main__':
    main()
