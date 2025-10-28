import streamlit as st
import pandas as pd
import numpy as np
import datetime
import sqlite3
from sklearn.ensemble import IsolationForest

# --- Database setup ---
conn = sqlite3.connect("bank.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS accounts
             (acc_number TEXT PRIMARY KEY, name TEXT, balance REAL)''')
c.execute('''CREATE TABLE IF NOT EXISTS transactions
             (id INTEGER PRIMARY KEY AUTOINCREMENT, acc_number TEXT, type TEXT, amount REAL, date TEXT)''')
conn.commit()

# --- Fraud model ---
if "fraud_model" not in st.session_state:
    st.session_state.fraud_model = IsolationForest(contamination=0.05, random_state=42)
    sample_data = np.random.normal(loc=0, scale=1, size=(100,1))
    st.session_state.fraud_model.fit(sample_data)

st.title("🏦 Full Bank Management App (with Transfers & Fraud Detection)")

# --- Sidebar Navigation ---
menu = st.sidebar.radio("📋 Sections", [
    "🏠 Home",
    "🧾 Create Account",
    "💰 Account Operations",
    "📜 All Accounts",
    "🔍 Search Account",
    "📆 Transaction Report",
    "⚙️ Manage Accounts"
])

# --- Home ---
if menu == "🏠 Home":
    st.subheader("Welcome to the Bank App!")
    st.write("Use the sidebar to navigate between features.")
    st.image("https://cdn-icons-png.flaticon.com/512/2331/2331970.png", width=150)

# --- Create New Account ---
elif menu == "🧾 Create Account":
    st.header("Create New Account")
    name = st.text_input("Customer Name")
    acc_number = st.text_input("Account Number")
    if st.button("Create Account"):
        c.execute("SELECT * FROM accounts WHERE acc_number=?", (acc_number,))
        if c.fetchone():
            st.error("⚠️ Account already exists!")
        elif not acc_number or not name:
            st.error("Please enter valid name and account number!")
        else:
            c.execute("INSERT INTO accounts VALUES (?, ?, ?)", (acc_number, name, 0.0))
            conn.commit()
            st.success(f"✅ Account {acc_number} for {name} created successfully!")

# --- Account Operations ---
elif menu == "💰 Account Operations":
    st.header("Account Operations")
    c.execute("SELECT acc_number FROM accounts")
    all_accounts = [row[0] for row in c.fetchall()]
    selected_acc = st.selectbox("Select Account", options=all_accounts if all_accounts else ["No accounts yet"])

    if selected_acc != "No accounts yet":
        # Get balance
        c.execute("SELECT balance FROM accounts WHERE acc_number=?", (selected_acc,))
        result = c.fetchone()
        if not result:
            st.error("Account not found.")
            st.stop()
        balance = result[0]
        st.subheader(f"Balance: ${balance:.2f}")
        if balance < 100:
            st.warning("⚠️ Low balance warning!")

        # Transaction input
        transaction_type = st.selectbox("Transaction Type", ["Deposit", "Withdraw", "Transfer"])
        amount = st.number_input("Amount", min_value=0.0, step=0.01)

        # Transfer target
        transfer_to = None
        if transaction_type == "Transfer":
            transfer_to = st.selectbox("Transfer To Account", options=[a for a in all_accounts if a != selected_acc])

        if st.button("Submit Transaction"):
            pred_input = np.array([[amount]])
            prediction = st.session_state.fraud_model.predict(pred_input)
            if prediction[0] == -1:
                st.warning("⚠️ Transaction flagged as potentially FRAUDULENT!")
            else:
                if transaction_type == "Deposit":
                    balance += amount
                    c.execute("UPDATE accounts SET balance=? WHERE acc_number=?", (balance, selected_acc))
                    c.execute("INSERT INTO transactions (acc_number, type, amount, date) VALUES (?, ?, ?, ?)",
                              (selected_acc, "Deposit", amount, datetime.datetime.now()))

                elif transaction_type == "Withdraw":
                    if amount > balance:
                        st.error("❌ Insufficient balance!")
                        st.stop()
                    balance -= amount
                    c.execute("UPDATE accounts SET balance=? WHERE acc_number=?", (balance, selected_acc))
                    c.execute("INSERT INTO transactions (acc_number, type, amount, date) VALUES (?, ?, ?, ?)",
                              (selected_acc, "Withdraw", amount, datetime.datetime.now()))

                elif transaction_type == "Transfer":
                    c.execute("SELECT balance FROM accounts WHERE acc_number=?", (transfer_to,))
                    target_result = c.fetchone()
                    if target_result is None:
                        st.error("Target account not found!")
                        st.stop()
                    target_balance = target_result[0]
                    if amount > balance:
                        st.error("❌ Insufficient balance!")
                        st.stop()
                    # Update balances
                    balance -= amount
                    target_balance += amount
                    c.execute("UPDATE accounts SET balance=? WHERE acc_number=?", (balance, selected_acc))
                    c.execute("UPDATE accounts SET balance=? WHERE acc_number=?", (target_balance, transfer_to))
                    now = datetime.datetime.now()
                    c.execute("INSERT INTO transactions (acc_number, type, amount, date) VALUES (?, ?, ?, ?)",
                              (selected_acc, f"Transfer to {transfer_to}", amount, now))
                    c.execute("INSERT INTO transactions (acc_number, type, amount, date) VALUES (?, ?, ?, ?)",
                              (transfer_to, f"Transfer from {selected_acc}", amount, now))

                conn.commit()
                st.success(f"✅ Transaction successful. New balance: ${balance:.2f}")

        # Transaction History
        st.subheader("Transaction History")
        c.execute("SELECT type, amount, date FROM transactions WHERE acc_number=? ORDER BY id DESC", (selected_acc,))
        rows = c.fetchall()
        if rows:
            df = pd.DataFrame(rows, columns=["Type", "Amount", "Date"])
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Transactions CSV", csv, "transactions.csv", "text/csv")
        else:
            st.write("No transactions yet.")

# --- All Accounts ---
elif menu == "📜 All Accounts":
    st.header("All Bank Accounts")
    c.execute("SELECT * FROM accounts")
    data = c.fetchall()
    if data:
        df = pd.DataFrame(data, columns=["Account Number", "Name", "Balance"])
        st.dataframe(df)
    else:
        st.info("No accounts available yet.")

# --- Search Account ---
elif menu == "🔍 Search Account":
    st.header("Search for an Account")
    keyword = st.text_input("Enter account number or name:")
    if st.button("Search"):
        c.execute("SELECT * FROM accounts WHERE acc_number LIKE ? OR name LIKE ?", (f"%{keyword}%", f"%{keyword}%"))
        results = c.fetchall()
        if results:
            df = pd.DataFrame(results, columns=["Account Number", "Name", "Balance"])
            st.dataframe(df)
        else:
            st.warning("No matching accounts found.")

# --- Transaction Report ---
elif menu == "📆 Transaction Report":
    st.header("Transaction Report by Date Range")
    start_date = st.date_input("From", datetime.date(2024, 1, 1))
    end_date = st.date_input("To", datetime.date.today())
    c.execute("SELECT * FROM transactions WHERE date BETWEEN ? AND ?", (start_date, end_date))
    data = c.fetchall()
    if data:
        df = pd.DataFrame(data, columns=["ID", "Account Number", "Type", "Amount", "Date"])
        st.dataframe(df)
    else:
        st.info("No transactions found in this range.")

# --- Manage Accounts ---
elif menu == "⚙️ Manage Accounts":
    st.header("Manage Accounts")
    c.execute("SELECT acc_number, name FROM accounts")
    accs = c.fetchall()
    if accs:
        selected_acc = st.selectbox("Select Account", [f"{a[0]} - {a[1]}" for a in accs])
        acc_number = selected_acc.split(" - ")[0]

        action = st.radio("Select Action", ["Edit Name", "Delete Account"])
        if action == "Edit Name":
            new_name = st.text_input("New Name")
            if st.button("Update Name"):
                c.execute("UPDATE accounts SET name=? WHERE acc_number=?", (new_name, acc_number))
                conn.commit()
                st.success("✅ Account name updated successfully!")

        elif action == "Delete Account":
            if st.button("❌ Delete Account"):
                c.execute("DELETE FROM accounts WHERE acc_number=?", (acc_number,))
                c.execute("DELETE FROM transactions WHERE acc_number=?", (acc_number,))
                conn.commit()
                st.warning("Account and related transactions deleted successfully!")
    else:
        st.info("No accounts found.")
