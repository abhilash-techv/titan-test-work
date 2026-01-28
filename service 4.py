import io
import os
import time
import glob 
from datetime import datetime
from fastapi import HTTPException
from utils.db import get_db_connection,get_db_engine,cohort_sftp_csv_exports_insert
import psycopg2
import json
import pickle
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sqlalchemy import text, inspect
import pandas as pd
from utils.file_helpers import get_file_path, file_read, file_upload_pickles, file_upload
from conf.conf import MODELS_DIR, BUSINESS_FEATURES


# conn = get_db_connection()
engine = get_db_engine()

# ---------------- Date Creation ----------------
def get_date_range():

    member_query = """
        SELECT enrollmentdate 
        FROM memberdata
        WHERE LOWER(enrollmentchannelcode) IN (
            'tanishq', 'encircle', 'encirclewebsite', 'ecommtanishq'
        )
    """

    txn_query = """
        SELECT invoice_date
        FROM transactionsdata
        WHERE LOWER(channel) = 'tanishq'
    """

    member_df = pd.read_sql(member_query, engine)
    txn_df = pd.read_sql(txn_query, engine)

    member_df["enrollmentdate"] = pd.to_datetime(member_df["enrollmentdate"], errors="coerce")
    txn_df["invoice_date"] = pd.to_datetime(txn_df["invoice_date"], errors="coerce")

    member_df = member_df.dropna(subset=["enrollmentdate"])
    txn_df = txn_df.dropna(subset=["invoice_date"])

    # Get max date from DB
    db_max_date = max(
        member_df["enrollmentdate"].max(),
        txn_df["invoice_date"].max()
    )

    # Fixed start date
    fixed_min_date = pd.to_datetime("2021-04-01")

    min_date = fixed_min_date
    max_date = db_max_date

    # ---- LOGGING ----
    print(f"📅 Date Range Generated → Start Date: {min_date.date()}  |  End Date: {max_date.date()}")

    # Generate date range
    date_range = pd.date_range(start=min_date, end=max_date).sort_values(ascending=False)

    return date_range


# ---------------- New Customer Count ----------------
def get_new_customer_count(engine):
    """
    Returns daily count of new customers using a single, efficient SQL query.
    This is preferred for better performance and scalability.
    """
    sql = """
    WITH FirstPurchases AS (
    SELECT
        cardno,
        MIN(invoice_date)::date AS first_purchase_date
    FROM transactionsdata
    WHERE LOWER(channel) = 'tanishq'
    GROUP BY cardno
    )
    SELECT
        first_purchase_date AS date,
        COUNT(cardno) AS NewCustomerCount
    FROM FirstPurchases
    GROUP BY first_purchase_date
    ORDER BY first_purchase_date;
    """
    df = pd.read_sql(sql, engine)
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_new_customer_count_full(engine, date_range):
    # --- Print start & end dates ---
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_new_customer_count_full → Start Date: {start_date}, End Date: {end_date}")

    df = get_new_customer_count(engine)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")

    # Optional rename if needed
    df = df.rename(columns={"newcustomercount": "NewCustomerCount"})

    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})

    merged = date_df.merge(df, on="date", how="left")
    merged["NewCustomerCount"] = merged["NewCustomerCount"].fillna(0).astype(int)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")

    return merged


# ---------------- Repeat Customer Count ----------------
def get_repeat_customer_count(engine):
    sql = """
        SELECT cardno, invoice_date
        FROM transactionsdata
        WHERE LOWER(channel) = 'tanishq'
    """
    
    txn_df = pd.read_sql(sql, engine)
    txn_df["invoice_date"] = pd.to_datetime(txn_df["invoice_date"], errors="coerce")
    txn_df = txn_df.dropna(subset=["invoice_date", "cardno"])
    first_purchase = txn_df.groupby("cardno")["invoice_date"].min().reset_index()
    first_purchase.rename(columns={"invoice_date": "first_purchase_date"}, inplace=True)
    txn_df = txn_df.merge(first_purchase, on="cardno", how="left")
    repeat_txns = txn_df[txn_df["invoice_date"] > txn_df["first_purchase_date"]]
    repeat_customer_count = (
        repeat_txns.groupby(repeat_txns["invoice_date"].dt.date)["cardno"]
        .nunique()
        .reset_index()
        .rename(columns={"invoice_date": "date", "cardno": "repeat_customers"})
    )
    repeat_customer_count["date"] = pd.to_datetime(repeat_customer_count["date"])
    return repeat_customer_count

def get_repeat_customer_count_full(engine, date_range):
    # --- Print start & end dates ---
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_repeat_customer_count_full → Start Date: {start_date}, End Date: {end_date}")

    repeat_customer_count = get_repeat_customer_count(engine)
    repeat_customer_count['date'] = pd.to_datetime(repeat_customer_count['date'])

    date_df = pd.DataFrame({"date": pd.to_datetime(date_range)})

    merged = date_df.merge(repeat_customer_count, on="date", how="left").fillna(0)
    merged["repeat_customers"] = merged["repeat_customers"].astype(int)
    
    return merged

# ---------------- New Customer Sales ----------------
def get_new_customer_sales(engine):
    sql = """
        SELECT cardno, invoice_date, eligible_amount
        FROM transactionsdata
        WHERE LOWER(channel) = 'tanishq'
    """
    
    txn_df = pd.read_sql(sql, engine)    
    txn_df["invoice_date"] = pd.to_datetime(txn_df["invoice_date"], errors="coerce")
    txn_df = txn_df.dropna(subset=["invoice_date", "cardno", "eligible_amount"])
    first_purchase = txn_df.groupby("cardno")["invoice_date"].min().reset_index()
    first_purchase.rename(columns={"invoice_date": "first_purchase_date"}, inplace=True)
    txn_df = txn_df.merge(first_purchase, on="cardno", how="left")
    new_txns = txn_df[txn_df["invoice_date"] == txn_df["first_purchase_date"]]
    new_customer_sales = (
        new_txns.groupby(new_txns["invoice_date"].dt.date)["eligible_amount"]
        .sum()
        .reset_index()
        .rename(columns={"invoice_date": "date", "eligible_amount": "new_customer_sales"})
    )
    new_customer_sales["date"] = pd.to_datetime(new_customer_sales["date"])
    print(new_customer_sales)
    return new_customer_sales

def get_new_customer_sales_full(engine, date_range):
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_new_customer_sales_full → Start Date: {start_date}, End Date: {end_date}")
    new_sales = get_new_customer_sales(engine)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(new_sales, on="date", how="left").fillna(0)
    merged["new_customer_sales"] = merged["new_customer_sales"].astype(float)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")
    return merged

# ---------------- Repeat Customer Sales ----------------
def get_repeat_customer_sales(engine):
    sql = """
        SELECT cardno, invoice_date, eligible_amount
        FROM transactionsdata
        WHERE LOWER(channel) = 'tanishq'
    """
    
    txn_df = pd.read_sql(sql, engine)    
    txn_df["invoice_date"] = pd.to_datetime(txn_df["invoice_date"], errors="coerce")
    txn_df = txn_df.dropna(subset=["invoice_date", "cardno", "eligible_amount"])
    first_purchase = txn_df.groupby("cardno")["invoice_date"].min().reset_index()
    first_purchase.rename(columns={"invoice_date": "first_purchase_date"}, inplace=True)
    txn_df = txn_df.merge(first_purchase, on="cardno", how="left")
    repeat_txns = txn_df[txn_df["invoice_date"] > txn_df["first_purchase_date"]]
    repeat_customer_sales = (
        repeat_txns.groupby(repeat_txns["invoice_date"].dt.date)["eligible_amount"]
        .sum()
        .reset_index()
        .rename(columns={"invoice_date": "date", "eligible_amount": "repeat_customer_sales"})
    )

    repeat_customer_sales["date"] = pd.to_datetime(repeat_customer_sales["date"])
    return repeat_customer_sales

def get_repeat_customer_sales_full(engine, date_range):
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_repeat_customer_sales_full → Start Date: {start_date}, End Date: {end_date}")
    repeat_sales = get_repeat_customer_sales(engine)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(repeat_sales, on="date", how="left").fillna(0)
    merged["repeat_customer_sales"] = merged["repeat_customer_sales"].astype(float)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")
    return merged

# ---------------- Birthday Customer Count ----------------
def get_birthday_customer_count(engine, window=15):
    member_sql = """
        SELECT card_no, dob
        FROM memberdata
        WHERE LOWER(enrollmentchannelcode) IN (
            'tanishq', 'encircle', 'encirclewebsite', 'ecommtanishq'
        )
    """

    txn_sql = """
        SELECT cardno, invoice_date
        FROM transactionsdata
        WHERE LOWER(channel) = 'tanishq'
    """

    df_members = pd.read_sql(member_sql, engine)
    df_txns = pd.read_sql(txn_sql, engine)

    df_members['card_no'] = df_members['card_no'].astype(str).str.split('.').str[0].str.strip()
    df_txns['cardno'] = df_txns['cardno'].astype(str).str.split('.').str[0].str.strip()

    df = pd.merge(df_txns, df_members, left_on='cardno', right_on='card_no', how='inner')
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df = df.dropna(subset=['invoice_date', 'dob'])

    # Calculate birthday in the transaction year
    def safe_birthday(row):
        try:
            return pd.Timestamp(
                year=row['invoice_date'].year,
                month=row['dob'].month,
                day=row['dob'].day
            )
        except ValueError:
            # Handle Feb 29 → Feb 28
            return pd.Timestamp(
                year=row['invoice_date'].year,
                month=row['dob'].month,
                day=28
            )

    df['birthday_this_year'] = df.apply(safe_birthday, axis=1)

    # Day difference
    df['days_diff'] = (df['invoice_date'] - df['birthday_this_year']).dt.days

    # Filter ±15 days
    df = df[(df['days_diff'] >= -15) & (df['days_diff'] <= 15)]

    # 🔥 GROUP BY PURCHASE DAY (NOT MONTH)
    df['purchase_day'] = df['invoice_date'].dt.normalize()   # Day-wise 00:00:00

    # Final day-wise count
    counts = (
        df.groupby('purchase_day')['cardno']
        .nunique()
        .reset_index()
        .rename(columns={'purchase_day': 'date', 'cardno': 'birthday_customers'})
    )

    return counts

def get_birthday_customer_count_full(engine, date_range):
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_birthday_customer_count_full → Start Date: {start_date}, End Date: {end_date}")
  
    birthday_count = get_birthday_customer_count(engine)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    
    merged = date_df.merge(birthday_count, on="date", how="left")
    merged["birthday_customers"] = merged["birthday_customers"].fillna(0).astype(int)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")
    return merged


# ---------------- Anniversary Customer Count ----------------
def get_anniversary_customer_count(engine, window=15):

    member_sql = """
        SELECT card_no, anniversary
        FROM memberdata
        WHERE LOWER(enrollmentchannelcode) IN (
            'tanishq', 'encircle', 'encirclewebsite', 'ecommtanishq'
        )
    """

    txn_sql = """
        SELECT cardno, invoice_date
        FROM transactionsdata
        WHERE LOWER(channel) = 'tanishq'
    """

    df_members = pd.read_sql(member_sql, engine)
    df_txns = pd.read_sql(txn_sql, engine)

    df_members['card_no'] = (
        df_members['card_no'].astype(str).str.split('.').str[0].str.strip()
    )
    df_txns['cardno'] = (
        df_txns['cardno'].astype(str).str.split('.').str[0].str.strip()
    )

    df = pd.merge(df_txns, df_members, left_on='cardno', right_on='card_no', how='inner')

    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    df['anniversary'] = pd.to_datetime(df['anniversary'], errors='coerce')

    df = df.dropna(subset=['invoice_date', 'anniversary'])

    # Safe anniversary handling
    def safe_anniversary(row):
        try:
            return pd.Timestamp(
                year=row['invoice_date'].year,
                month=row['anniversary'].month,
                day=row['anniversary'].day
            )
        except ValueError:
            # Handle Feb 29 → Feb 28
            return pd.Timestamp(
                year=row['invoice_date'].year,
                month=row['anniversary'].month,
                day=28
            )

    df['anniversary_this_year'] = df.apply(safe_anniversary, axis=1)

    df['days_diff'] = (df['invoice_date'] - df['anniversary_this_year']).dt.days

    # Select dates within ±15-day window
    df = df[(df['days_diff'] >= -15) & (df['days_diff'] <= 15)]

    # *************** IMPORTANT CHANGE ***************
    # Group by purchase DAY (invoice_date), not month
    df['purchase_day'] = df['invoice_date'].dt.normalize()  # yyyy-mm-dd 00:00:00

    counts = (
        df.groupby('purchase_day')['cardno']
        .nunique()
        .reset_index()
        .rename(columns={'purchase_day': 'date', 'cardno': 'anniversary_customers'})
    )

    return counts

def get_anniversary_customer_count_full(engine, date_range):
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_anniversary_customer_count_full → Start Date: {start_date}, End Date: {end_date}")
  
    annv_count = get_anniversary_customer_count(engine)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(annv_count, on="date", how="left")
    merged["anniversary_customers"] = merged["anniversary_customers"].fillna(0).astype(int)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")
    return merged

# ---------------- Birthday Customer Sales ----------------
def get_birthday_customer_sales(engine, window=15):
    member_sql = """
        SELECT card_no, dob
        FROM memberdata
        WHERE LOWER(enrollmentchannelcode) IN (
            'tanishq', 'encircle', 'encirclewebsite', 'ecommtanishq'
        )
    """

    txn_sql = """
        SELECT cardno, invoice_date, eligible_amount
        FROM transactionsdata
        WHERE LOWER(channel) = 'tanishq'
    """

    df_members = pd.read_sql(member_sql, engine)
    df_txns = pd.read_sql(txn_sql, engine)

    # Clean card numbers
    df_members['card_no'] = (
        df_members['card_no'].astype(str).str.split('.').str[0].str.strip()
    )
    df_txns['cardno'] = (
        df_txns['cardno'].astype(str).str.split('.').str[0].str.strip()
    )

    # Merge data
    df = pd.merge(df_txns, df_members, left_on='cardno', right_on='card_no', how='inner')

    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    df = df.dropna(subset=['invoice_date', 'dob'])

    # Construct birthday for the invoice year
    def safe_birthday(row):
        try:
            return pd.Timestamp(
                year=row['invoice_date'].year,
                month=row['dob'].month,
                day=row['dob'].day
            )
        except ValueError:
            # Handle Feb 29 case
            return pd.Timestamp(
                year=row['invoice_date'].year,
                month=row['dob'].month,
                day=28
            )

    df['birthday_this_year'] = df.apply(safe_birthday, axis=1)

    # Calculate day difference
    df['days_diff'] = (df['invoice_date'] - df['birthday_this_year']).dt.days

    # Filter within ± window
    df = df[(df['days_diff'] >= -window) & (df['days_diff'] <= window)]

    # ******** KEY CHANGE: group by PURCHASE DATE (DAY-WISE) ********
    df['purchase_day'] = df['invoice_date'].dt.normalize()  # yyyy-mm-dd

    sales = (
        df.groupby('purchase_day')['eligible_amount']
        .sum()
        .reset_index()
        .rename(columns={'purchase_day': 'date', 'eligible_amount': 'BirthdayCustomerSales'})
    )

    return sales

def get_birthday_customer_sales_full(engine, date_range):
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_birthday_customer_sales_full → Start Date: {start_date}, End Date: {end_date}")
  
    birthday_sales = get_birthday_customer_sales(engine)  
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(birthday_sales, on="date", how="left")
    merged["BirthdayCustomerSales"] = merged["BirthdayCustomerSales"].fillna(0)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")
    return merged

# ---------------- Anniversary Customer Sales ----------------
def get_anniversary_customer_sales(engine, window=15):
    member_sql = """
        SELECT card_no, anniversary
        FROM memberdata
        WHERE LOWER(enrollmentchannelcode) IN (
            'tanishq', 'encircle', 'encirclewebsite', 'ecommtanishq'
        )
    """

    txn_sql = """
        SELECT cardno, invoice_date, eligible_amount
        FROM transactionsdata
        WHERE LOWER(channel) = 'tanishq'
    """

    df_members = pd.read_sql(member_sql, engine)
    df_txns = pd.read_sql(txn_sql, engine)

    # Clean card numbers
    df_members['card_no'] = df_members['card_no'].astype(str).str.split('.').str[0].str.strip()
    df_txns['cardno'] = df_txns['cardno'].astype(str).str.split('.').str[0].str.strip()

    # Merge
    df = pd.merge(df_txns, df_members, left_on='cardno', right_on='card_no', how='inner')

    # Convert dates
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    df['anniversary'] = pd.to_datetime(df['anniversary'], errors='coerce')
    df = df.dropna(subset=['invoice_date', 'anniversary'])

    # ---- Construct anniversary date for the invoice year (vectorized) ----
    anniv = df['anniversary']
    inv = df['invoice_date']

    df['anniversary_this_year'] = pd.to_datetime({
        "year": inv.dt.year,
        "month": anniv.dt.month,
        "day": anniv.dt.day.clip(upper=28)
    }, errors='coerce')

    # ---- Day difference ----
    df['days_diff'] = (df['invoice_date'] - df['anniversary_this_year']).dt.days

    # ---- Filter ± window days ----
    df = df[(df['days_diff'] >= -window) & (df['days_diff'] <= window)]

    # ---- GROUP BY DAYWISE instead of MONTHWISE ----
    sales = (
        df.groupby(df['invoice_date'].dt.date)['eligible_amount']
        .sum()
        .reset_index()
        .rename(columns={'invoice_date': 'date', 'eligible_amount': 'AnnvCustomerSales'})
    )

    # Convert date column back to datetime
    sales['date'] = pd.to_datetime(sales['date'])

    return sales

def get_anniversary_customer_sales_full(engine, date_range):
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_anniversary_customer_sales_full → Start Date: {start_date}, End Date: {end_date}")
  
    annv_sales = get_anniversary_customer_sales(engine)  # function we defined earlier
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(annv_sales, on="date", how="left")
    merged["AnnvCustomerSales"] = merged["AnnvCustomerSales"].fillna(0)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")
    return merged

# ---------------- Tital Customer Transaction ----------------
def get_total_customer_transactions(engine, date_range):
    query = """
        SELECT invoice_date, cardno
        FROM transactionsdata
        WHERE LOWER(channel) = 'tanishq'
    """
    df_txns = pd.read_sql(query, engine)
    df_txns["invoice_date"] = pd.to_datetime(df_txns["invoice_date"], errors="coerce")
    df_txns = df_txns.dropna(subset=["invoice_date"])

    txn_count = (
        df_txns.groupby(df_txns["invoice_date"].dt.date)["cardno"]
        .nunique()
        .reset_index()
        .rename(columns={"cardno": "TotalCustomerTransactions", "invoice_date": "date"})
    )
    txn_count["date"] = pd.to_datetime(txn_count["date"], errors="coerce")
    return txn_count

def get_total_customer_transactions_full(engine, date_range):
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_total_customer_transactions_full → Start Date: {start_date}, End Date: {end_date}")
  
    txn_count = get_total_customer_transactions(engine, date_range)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})

    merged = date_df.merge(txn_count, on="date", how="left")
    merged["TotalCustomerTransactions"] = merged["TotalCustomerTransactions"].fillna(0).astype(int)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")
    return merged

# ---------------- Transacted Customer Sales ----------------
def get_transacted_customer_sales(engine):
    """
    Get daily transacted customer sales (raw, not aligned to full date range).
    """
    query = """
        SELECT invoice_date, cardno, eligible_amount
        FROM transactionsdata
        WHERE LOWER(channel) = 'tanishq'
    """
    df_txns = pd.read_sql(query, engine)
    df_txns["invoice_date"] = pd.to_datetime(df_txns["invoice_date"], errors="coerce")
    df_txns = df_txns.dropna(subset=["invoice_date", "eligible_amount"])
    sales_df = (
        df_txns.groupby(df_txns["invoice_date"].dt.date)["eligible_amount"]
        .sum()
        .reset_index()
        .rename(columns={"eligible_amount": "TransactedCustomerSales", "invoice_date": "date"})
    )

    sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
    return sales_df


def get_transacted_customer_sales_full(engine, date_range):
    """
    Align daily transacted customer sales with full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_transacted_customer_sales_full → Start Date: {start_date}, End Date: {end_date}")
  
    sales_df = get_transacted_customer_sales(engine)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(sales_df, on="date", how="left")
    merged["TransactedCustomerSales"] = merged["TransactedCustomerSales"].fillna(0)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")
    return merged

# ---------------- Customer Birthdays Per Day (Enrollment-based) ----------------
def get_customer_birthdays_perday(engine):
    """
    Counts all enrolled customers whose birthday falls on each day.
    """
    member_df = pd.read_sql("""
        SELECT card_no, dob, enrollmentdate
        FROM memberdata
        WHERE enrollmentdate IS NOT NULL
          AND dob IS NOT NULL
          AND LOWER(enrollmentchannelcode) IN (
              'tanishq', 'encircle', 'encirclewebsite', 'ecommtanishq'
          )
    """, engine)
    member_df['dob'] = pd.to_datetime(member_df['dob'], errors='coerce')
    member_df['enrollmentdate'] = pd.to_datetime(member_df['enrollmentdate'], errors='coerce')
    member_df['birth_month'] = member_df['dob'].dt.month
    member_df['birth_day'] = member_df['dob'].dt.day
    member_df['enroll_year_month'] = member_df['enrollmentdate'].dt.to_period('M')
    max_date = member_df['enrollmentdate'].max()
    date_range = pd.date_range(start='2021-04-01', end=max_date)
    results = []
    for current_date in date_range:
        cutoff_period = current_date.to_period('M')
        enrolled_cust = member_df[member_df['enroll_year_month'] <= cutoff_period]
        birthday_cust = enrolled_cust[
            (enrolled_cust['birth_month'] == current_date.month) &
            (enrolled_cust['birth_day'] == current_date.day)
        ]
        cust_count = birthday_cust['card_no'].nunique()
        results.append({'date': current_date, 'CustomerBirthdaysPerday': cust_count})

    df_birthdays = pd.DataFrame(results)
    return df_birthdays

def get_customer_birthdays_perday_full(engine, date_range):
    """
    Aligns CustomerBirthdaysPerday to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_customer_birthdays_perday_full → Start Date: {start_date}, End Date: {end_date}")
  
    birthday_df = get_customer_birthdays_perday(engine)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(birthday_df, on="date", how="left")
    merged["CustomerBirthdaysPerday"] = merged["CustomerBirthdaysPerday"].fillna(0).astype(int)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")

    return merged

# ---------------- Customer Anniversaries Per Day (Enrollment-based) ----------------
def get_customer_anniversaries_perday(engine):
    """
    Counts all enrolled customers whose anniversary falls on each day.
    """
    member_df = pd.read_sql("""
        SELECT card_no, anniversary, enrollmentdate
        FROM memberdata
        WHERE enrollmentdate IS NOT NULL
          AND anniversary IS NOT NULL
          AND LOWER(enrollmentchannelcode) IN (
              'tanishq', 'encircle', 'encirclewebsite', 'ecommtanishq'
          )
    """, engine)
    member_df['anniversary'] = pd.to_datetime(member_df['anniversary'], errors='coerce')
    member_df['enrollmentdate'] = pd.to_datetime(member_df['enrollmentdate'], errors='coerce')
    member_df['anniv_month'] = member_df['anniversary'].dt.month
    member_df['anniv_day'] = member_df['anniversary'].dt.day
    member_df['enroll_year_month'] = member_df['enrollmentdate'].dt.to_period('M')
    max_date = member_df['enrollmentdate'].max()
    date_range = pd.date_range(start='2021-04-01', end=max_date)
    results = []
    for current_date in date_range:
        cutoff_period = current_date.to_period('M')
        enrolled_cust = member_df[member_df['enroll_year_month'] <= cutoff_period]
        anniversary_cust = enrolled_cust[
            (enrolled_cust['anniv_month'] == current_date.month) &
            (enrolled_cust['anniv_day'] == current_date.day)
        ]
        cust_count = anniversary_cust['card_no'].nunique()
        results.append({'date': current_date, 'CustomerAnniversariesPerday': cust_count})
    df_anniversaries = pd.DataFrame(results)
    return df_anniversaries


def get_customer_anniversaries_perday_full(engine, date_range):
    """
    Aligns CustomerAnniversariesPerday to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_customer_anniversaries_perday_full → Start Date: {start_date}, End Date: {end_date}")
  
    anniversary_df = get_customer_anniversaries_perday(engine)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(anniversary_df, on="date", how="left")
    merged["CustomerAnniversariesPerday"] = merged["CustomerAnniversariesPerday"].fillna(0).astype(int)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")

    return merged

# ---------------- Daily Enrolled Customers ----------------
def get_daily_enrolled_customers(engine):
    """
    Counts all customers enrolled on each date.
    """
    df = pd.read_sql("""
        SELECT enrollmentdate, card_no
        FROM public."memberdata"
        WHERE enrollmentdate IS NOT NULL
          AND LOWER(enrollmentchannelcode) IN (
              'tanishq', 'encircle', 'encirclewebsite', 'ecommtanishq'
          )
    """, engine)

    df['enrollmentdate'] = pd.to_datetime(df['enrollmentdate'], errors='coerce')
    
    daily_counts = (
        df.groupby('enrollmentdate')['card_no']
        .nunique()
        .reset_index()
        .rename(columns={'enrollmentdate': 'date', 'card_no': 'EnrolledCustomerCount'})
    )
    return daily_counts

def get_daily_enrolled_customers_full(engine, date_range):
    """
    Aligns EnrolledCustomerCount to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_daily_enrolled_customers_full → Start Date: {start_date}, End Date: {end_date}")
  
    enrolled_df = get_daily_enrolled_customers(engine)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(enrolled_df, on="date", how="left")
    merged["EnrolledCustomerCount"] = merged["EnrolledCustomerCount"].fillna(0).astype(int)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")

    return merged

# ---------------- Members Archived Per Day ----------------
# ---------------- Members Archived Per Day ----------------
def get_members_archived(engine):
    enrollments = pd.read_sql("""
        SELECT card_no, enrollmentdate
        FROM public."memberdata"
        WHERE enrollmentdate IS NOT NULL
          AND enrollmentdate <= '2025-06-04'
          AND LOWER(enrollmentchannelcode) IN (
              'tanishq', 'encircle', 'encirclewebsite', 'ecommtanishq'
          )
    """, engine)
 
    # 🚨 Hard stop if no data
    if enrollments.empty:
        return pd.DataFrame(columns=["date", "MembersArchived"])
 
    last_txn = pd.read_sql("""
        SELECT cardno, MAX(invoice_date) AS last_transaction_date
        FROM public."transactionsdata"
        WHERE LOWER(channel) = 'tanishq'
        GROUP BY cardno
    """, engine)
 
    enrollments.rename(columns={'card_no': 'cardno'}, inplace=True)
 
    df = enrollments.merge(last_txn, on='cardno', how='left')
 
    df['enrollmentdate'] = pd.to_datetime(df['enrollmentdate'], errors='coerce')
    df['last_transaction_date'] = pd.to_datetime(df['last_transaction_date'], errors='coerce')
 
    # 🚨 Drop NaT enrollment dates
    df = df.dropna(subset=['enrollmentdate'])
 
    if df.empty:
        return pd.DataFrame(columns=["date", "MembersArchived"])
 
    start_date = df['enrollmentdate'].min()
    end_date = pd.to_datetime('2025-06-04')
 
    date_range = pd.date_range(start=start_date, end=end_date)
 
    results = []
    for current_date in date_range:
        enrolled_customers = df[df['enrollmentdate'] <= current_date]
        threshold_date = current_date - pd.DateOffset(months=36)
 
        archived_customers = enrolled_customers[
            (enrolled_customers['last_transaction_date'].isna()) |
            (enrolled_customers['last_transaction_date'] < threshold_date)
        ]
 
        results.append({
            'date': current_date,
            'MembersArchived': archived_customers['cardno'].nunique()
        })
 
    return pd.DataFrame(results)

def get_members_archived_full(engine, date_range):
    """
    Aligns MembersArchived to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_members_archived_full → Start Date: {start_date}, End Date: {end_date}")
  
    archived_df = get_members_archived(engine)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(archived_df, on="date", how="left")
    merged["MembersArchived"] = merged["MembersArchived"].fillna(0).astype(int)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")

    return merged

# ---------------- Cross Channel Buyers Per Day ----------------
def get_cross_channel_buyers(engine):
    """
    Returns daily count of cross-channel buyers based on first Tanishq transaction
    and prior other-channel transactions or enrollment.
    """
    sql = """
    WITH calendar AS (
        SELECT generate_series(
            DATE '2021-04-01',
            (SELECT MAX(invoice_date::date) FROM transactionsdata),
            INTERVAL '1 day'
        )::date AS txn_date
    ),
    ta_first_txn AS (
        SELECT
            cardno,
            MIN(invoice_date::date) AS first_ta_date
        FROM transactionsdata
        WHERE LOWER(channel) = 'tanishq'
        GROUP BY cardno
    ),
    prior_other_txn AS (
        SELECT DISTINCT t.cardno
        FROM transactionsdata t
        JOIN ta_first_txn ta ON t.cardno = ta.cardno
        WHERE LOWER(t.channel) <> 'tanishq'
          AND t.invoice_date::date < ta.first_ta_date
    ),
    enrolled_before_ta AS (
        SELECT DISTINCT m.card_no
        FROM memberdata m
        JOIN ta_first_txn ta ON m.card_no = ta.cardno
        WHERE m.enrollmentdate::date < ta.first_ta_date
    ),
    cross_channel_buyers AS (
        SELECT cardno, first_ta_date
        FROM ta_first_txn
        WHERE cardno IN (
            SELECT cardno FROM prior_other_txn
            UNION
            SELECT card_no FROM enrolled_before_ta
        )
    )
    SELECT
        c.txn_date,
        COUNT(cb.cardno) AS cross_channel_buyer_count
    FROM
        calendar c
    LEFT JOIN cross_channel_buyers cb
        ON cb.first_ta_date = c.txn_date
    GROUP BY
        c.txn_date
    ORDER BY
        c.txn_date;
    """

    df = pd.read_sql(sql, engine)
    df.rename(columns={'txn_date': 'date', 'cross_channel_buyer_count': 'CrossChannelBuyers'}, inplace=True)
    return df

def get_cross_channel_buyers_full(engine, date_range):
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_cross_channel_buyers_full → Start Date: {start_date}, End Date: {end_date}")
  
    df_ccb = get_cross_channel_buyers(engine) 
    df_ccb['date'] = pd.to_datetime(df_ccb['date'], errors='coerce')  

    date_df = pd.DataFrame({"date": pd.to_datetime(date_range)})
    merged = date_df.merge(df_ccb, on="date", how="left")
    merged["CrossChannelBuyers"] = merged["CrossChannelBuyers"].fillna(0).astype(int)
    return merged

# ---------------- Diamond Enthusiasts ----------------

def get_diamond_enthusiasts(engine, dates):
    start_date = pd.to_datetime(dates[0]).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(dates[1]).strftime("%Y-%m-%d")
    sql = """
        SELECT 
            t.invoice_date::date AS txn_date,
            COUNT(DISTINCT t.cardno) AS diamond_enthusiast_count
        FROM transactionsdata t
        JOIN memberdata m ON t.cardno = m.card_no
        WHERE t.category ILIKE %s
          AND LOWER(t.channel) = 'tanishq'
          AND LOWER(m.enrollmentchannelcode) IN ('tanishq','encircle','encirclewebsite','ecommtanishq')
          AND t.invoice_date::date BETWEEN %s AND %s
        GROUP BY t.invoice_date::date
        ORDER BY t.invoice_date::date;
    """

    pattern = '%diamond%'
    params = (pattern, start_date, end_date)   
    df = pd.read_sql(sql, engine, params=params)
    df.rename(columns={'txn_date': 'date', 'diamond_enthusiast_count': 'DiamondEnthusiasts'}, inplace=True)
    return df

def get_diamond_enthusiasts_full(engine, date_range):
    """
    Align Diamond Enthusiasts to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_diamond_enthusiasts_full → Start Date: {start_date}, End Date: {end_date}")
  
    df_de = get_diamond_enthusiasts(engine, date_range)
    df_de['date'] = pd.to_datetime(df_de['date'], errors='coerce')
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range)})
    merged = date_df.merge(df_de, on="date", how="left")
    merged['DiamondEnthusiasts'] = merged['DiamondEnthusiasts'].fillna(0).astype(int)
    
    return merged

# ---------------- Dormant Customers Per Day ----------------
def get_dormant_customers(engine):
    """
    Returns daily count of dormant customers (eligible members with last txn > 6 months ago),
    filtered by transaction channel and enrollment channels.
    """
    sql = """
    WITH calendar AS (
        SELECT generate_series(
            DATE '2021-04-01',
            (SELECT MAX(invoice_date::date) FROM transactionsdata),
            INTERVAL '1 day'
        )::date AS cal_date
    ),
    last_txn AS (
        SELECT
            cardno,
            MAX(invoice_date::date) AS last_txn_date
        FROM transactionsdata
        WHERE eligible_amount <> 0
          AND LOWER(channel) = 'tanishq'
        GROUP BY cardno
    ),
    eligible_members AS (
        SELECT
            m.card_no,
            m.pointbalance,
            l.last_txn_date
        FROM memberdata m
        LEFT JOIN last_txn l ON m.card_no = l.cardno
        WHERE m.pointbalance > 50
          AND LOWER(m.enrollmentchannelcode) IN ('tanishq','encircle','encirclewebsite','ecommtanishq')
    ),
    daily_dormant AS (
        SELECT
            cal.cal_date,
            COUNT(*) AS dormant_count
        FROM calendar cal
        JOIN eligible_members em ON
            em.last_txn_date IS NOT NULL
            AND em.last_txn_date < (cal.cal_date - INTERVAL '6 months')
        GROUP BY cal.cal_date
    )
    SELECT
        cal.cal_date,
        COALESCE(dd.dormant_count, 0) AS dormant_customer_count
    FROM calendar cal
    LEFT JOIN daily_dormant dd ON cal.cal_date = dd.cal_date
    ORDER BY cal.cal_date;
    """


    df = pd.read_sql(sql, engine)
    df.rename(columns={'cal_date': 'date', 'dormant_customer_count': 'DormantCustomers'}, inplace=True)
    return df

def get_dormant_customers_full(engine, date_range):
    """
    Align DormantCustomers to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_dormant_customers_full → Start Date: {start_date}, End Date: {end_date}")
  
    df_dc = get_dormant_customers(engine)
    df_dc['date'] = pd.to_datetime(df_dc['date'], errors='coerce')
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range)})
    merged = date_df.merge(df_dc, on="date", how="left")
    merged["DormantCustomers"] = merged["DormantCustomers"].fillna(0).astype(int)
    return merged

# ---------------- Low Point Balance Customers ----------------
def get_low_point_balance_customers(engine):
    """
    Returns daily count of customers with point balance < 50,
    filtered by enrollment channel.
    """
    sql = """
    WITH calendar AS (
        SELECT generate_series(
            DATE '2021-04-01',
            (SELECT MAX(invoice_date::date) FROM transactionsdata),
            INTERVAL '1 day'
        )::date AS cal_date
    )
    SELECT
        cal.cal_date,
        COUNT(*) FILTER (
            WHERE m.pointbalance < 50
              AND m.enrollmentdate::date <= cal.cal_date
              AND LOWER(m.enrollmentchannelcode) IN ('tanishq','encircle','encirclewebsite','ecommtanishq')
        ) AS low_point_balance_customers
    FROM
        calendar cal
    CROSS JOIN
        memberdata m
    GROUP BY
        cal.cal_date
    ORDER BY
        cal.cal_date;
    """
    df = pd.read_sql(sql, engine)
    df.rename(columns={'cal_date': 'date', 'low_point_balance_customers': 'LowPointBalanceCustomers'}, inplace=True)
    return df

def get_low_point_balance_customers_full(engine, date_range):
    """
    Align LowPointBalanceCustomers to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_low_point_balance_customers_full → Start Date: {start_date}, End Date: {end_date}")
  
    df_lp = get_low_point_balance_customers(engine)
    df_lp['date'] = pd.to_datetime(df_lp['date'], errors='coerce')
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(df_lp, on="date", how="left")
    merged["LowPointBalanceCustomers"] = merged["LowPointBalanceCustomers"].fillna(0).astype(int)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")
    
    return merged

# ---------------- Active Customers ----------------
def get_active_customers(engine):
    """
    Returns daily count of active customers.
    """
    sql = """
    WITH calendar AS (
        SELECT generate_series(
            DATE '2021-04-01',
            (SELECT MAX(invoice_date::date) FROM transactionsdata),
            INTERVAL '1 day'
        )::date AS cal_date
    ),
    active_customers AS (
        SELECT DISTINCT m.card_no, t.invoice_date::date AS txn_date
        FROM memberdata m
        JOIN transactionsdata t
            ON m.card_no = t.cardno
        WHERE
            t.eligible_amount <> 0
            AND t.invoice_date::date >= m.enrollmentdate::date
            AND LOWER(t.channel) IN ('tanishq','encircle','encirclewebsite','ecommtanishq')
            AND LOWER(m.enrollmentchannelcode) IN ('tanishq','encircle','encirclewebsite','ecommtanishq')
    )
    SELECT
        c.cal_date,
        COUNT(DISTINCT a.card_no) FILTER (
            WHERE a.txn_date = c.cal_date
        ) AS active_customer_count
    FROM calendar c
    LEFT JOIN active_customers a
        ON a.txn_date = c.cal_date
    GROUP BY c.cal_date
    ORDER BY c.cal_date;
    """
    df = pd.read_sql(sql, engine)
    df.rename(columns={'cal_date': 'date', 'active_customer_count': 'ActiveCustomers'}, inplace=True)
    return df

def get_active_customers_full(engine, date_range):
    """
    Align ActiveCustomers to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_active_customers_full → Start Date: {start_date}, End Date: {end_date}")
  
    df_active = get_active_customers(engine)
    df_active['date'] = pd.to_datetime(df_active['date'], errors='coerce')
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range)})
    merged = date_df.merge(df_active, on="date", how="left")
    merged["ActiveCustomers"] = merged["ActiveCustomers"].fillna(0).astype(int)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")
    
    return merged

# ---------------- Campaigns Deployed ----------------
def get_campaigns_deployed(engine):
    """
    Returns daily count of campaigns deployed.
    """
    sql = """
    WITH normalized AS (
    SELECT
        id,
        CASE
            -- If already in ISO format (YYYY-MM-DD or with time part)
            WHEN "Deployment_date" ~ '^\d{4}-\d{2}-\d{2}'
                THEN "Deployment_date"::date
            -- If in DD-MM-YYYY
            WHEN "Deployment_date" ~ '^\d{2}-\d{2}-\d{4}'
                THEN to_date("Deployment_date", 'DD-MM-YYYY')
            ELSE NULL  -- invalid/unexpected format
        END AS deploy_date
    FROM campaigndata
    ),
    calendar AS (
        SELECT generate_series(
            DATE '2021-04-01',
            (SELECT MAX(deploy_date) FROM normalized),
            INTERVAL '1 day'
        )::date AS cal_date
    )
    SELECT
        c.cal_date,
        COUNT(n.id) FILTER (WHERE n.deploy_date = c.cal_date) AS campaigns_deployed
    FROM calendar c
    LEFT JOIN normalized n
        ON n.deploy_date = c.cal_date
    GROUP BY c.cal_date
    ORDER BY c.cal_date;

    """
    df = pd.read_sql(sql, engine)
    df.rename(columns={'cal_date': 'date', 'campaigns_deployed': 'CampaignsDeployed'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    
    return df


def get_campaigns_deployed_full(engine, date_range):
    """
    Align campaigns deployed to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_campaigns_deployed_full → Start Date: {start_date}, End Date: {end_date}")
  
    df_campaigns = get_campaigns_deployed(engine)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(df_campaigns, on="date", how="left")
    merged['CampaignsDeployed'] = merged['CampaignsDeployed'].fillna(0).astype(int)
    merged['date'] = merged['date'].dt.strftime("%d/%m/%Y")
    
    return merged


# ---------------- Targeted Count ----------------
# This function's sole job is to get the raw data from the database
def get_targeted_count(engine):
    sql = """
    WITH calendar AS (
      SELECT generate_series(
          DATE '2021-04-01',
          (SELECT MAX(
                  CASE
                      WHEN "Deployment_date" SIMILAR TO '__-__-____%' THEN to_date("Deployment_date", 'DD-MM-YYYY')
                      WHEN "Deployment_date" SIMILAR TO '____-__-____%' THEN to_date(SUBSTRING("Deployment_date" FOR 10), 'YYYY-MM-DD')
                      ELSE NULL
                  END
              ) FROM campaigndata),
          INTERVAL '1 day'
      )::date AS cal_date
      )
      SELECT
          c.cal_date,
          COALESCE(SUM(
              CASE
                  WHEN TRIM(cd."Target_Count") ~ '^[0-9]+$' THEN TRIM(cd."Target_Count")::integer
                  ELSE 0
              END
          ) FILTER (
                  WHERE
                      CASE
                          WHEN cd."Deployment_date" SIMILAR TO '__-__-____%' THEN to_date(cd."Deployment_date", 'DD-MM-YYYY')
                          WHEN cd."Deployment_date" SIMILAR TO '____-__-____%' THEN to_date(SUBSTRING(cd."Deployment_date" FOR 10), 'YYYY-MM-DD')
                          ELSE NULL
                      END = c.cal_date
              ), 0) AS targetedcount
      FROM calendar c
      LEFT JOIN campaigndata cd
          ON
               CASE
                   WHEN cd."Deployment_date" SIMILAR TO '__-__-____%' THEN to_date(cd."Deployment_date", 'DD-MM-YYYY')
                   WHEN cd."Deployment_date" SIMILAR TO '____-__-____%' THEN to_date(SUBSTRING(cd."Deployment_date" FOR 10), 'YYYY-MM-DD')
                   ELSE NULL
               END = c.cal_date
      GROUP BY c.cal_date
      ORDER BY c.cal_date;
    """
    df = pd.read_sql(text(sql), engine)
    df.rename(columns={'cal_date': 'date'}, inplace=True)
    return df

# This function handles the date range alignment and final formatting
def get_targeted_count_full(engine, date_range):
    """
    Align TargetedCount to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_targeted_count_full → Start Date: {start_date}, End Date: {end_date}")
  
    # 1. Get the data from the database
    df_targeted = get_targeted_count(engine)
    
    # 2. Convert date column to datetime type (it should already be, but good practice)
    df_targeted['date'] = pd.to_datetime(df_targeted['date'])

    # 3. Create the full date range DataFrame
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})

    # 4. Merge the two DataFrames on the 'date' column
    merged = date_df.merge(df_targeted, on="date", how="left")

    # 5. Fill NaN values and convert type
    merged["targetedcount"] = merged["targetedcount"].fillna(0).astype(int)

    # 6. Convert the final date column to the desired string format
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")

    return merged

# ---------------- Buyers ----------------
def get_buyers(engine):
    """
    Returns daily count of buyers from campaigndata.
    """
    sql = """
    WITH calendar AS (
      SELECT generate_series(
          DATE '2021-04-01',
          (SELECT MAX(
                  CASE
                      WHEN "Deployment_date" SIMILAR TO '__-__-____%' THEN to_date("Deployment_date", 'DD-MM-YYYY')
                      WHEN "Deployment_date" SIMILAR TO '____-__-____%' THEN to_date(SUBSTRING("Deployment_date" FOR 10), 'YYYY-MM-DD')
                      ELSE NULL
                  END
              ) FROM campaigndata),
          INTERVAL '1 day'
      )::date AS cal_date
      )
      SELECT
          c.cal_date,
          COALESCE(SUM(
              -- Safely cast Buyers to integer
              CASE
                  WHEN TRIM(cd."Buyers") ~ '^[0-9]+$' THEN TRIM(cd."Buyers")::integer
                  ELSE 0
              END
          ) FILTER (
                  WHERE
                      CASE
                          WHEN cd."Deployment_date" SIMILAR TO '__-__-____%' THEN to_date(cd."Deployment_date", 'DD-MM-YYYY')
                          WHEN cd."Deployment_date" SIMILAR TO '____-__-____%' THEN to_date(SUBSTRING(cd."Deployment_date" FOR 10), 'YYYY-MM-DD')
                          ELSE NULL
                      END = c.cal_date
              ), 0) AS buyers
      FROM calendar c
      LEFT JOIN campaigndata cd
          ON
               CASE
                   WHEN cd."Deployment_date" SIMILAR TO '__-__-____%' THEN to_date(cd."Deployment_date", 'DD-MM-YYYY')
                   WHEN cd."Deployment_date" SIMILAR TO '____-__-____%' THEN to_date(SUBSTRING(cd."Deployment_date" FOR 10), 'YYYY-MM-DD')
                   ELSE NULL
               END = c.cal_date
      GROUP BY c.cal_date
      ORDER BY c.cal_date;
    """
    
    df = pd.read_sql(text(sql), engine)
    df.rename(columns={'cal_date': 'date', 'buyers': 'Buyers'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def get_buyers_full(engine, date_range):
    """
    Align Buyers to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_buyers_full → Start Date: {start_date}, End Date: {end_date}")
  
    df_buyers = get_buyers(engine)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(df_buyers, on="date", how="left")
    merged["Buyers"] = merged["Buyers"].fillna(0).astype(int)
    merged.rename(columns={"Buyers": "campaignbuyers"}, inplace=True)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")

    return merged

# ---------------- Incremental Revenue ----------------
def get_incremental_rev(engine):
    """
    Returns daily incremental revenue from campaigndata.
    """
    sql = """
    WITH calendar AS (
      SELECT generate_series(
          DATE '2021-04-01',
          (SELECT MAX(
                  CASE
                      WHEN "Deployment_date" SIMILAR TO '__-__-____%' THEN to_date("Deployment_date", 'DD-MM-YYYY')
                      WHEN "Deployment_date" SIMILAR TO '____-__-____%' THEN to_date(SUBSTRING("Deployment_date" FOR 10), 'YYYY-MM-DD')
                      ELSE NULL
                  END
              ) FROM campaigndata),
          INTERVAL '1 day'
      )::date AS cal_date
      )
      SELECT
          c.cal_date,
          COALESCE(SUM(
              -- Safely cast Incremental_rev to numeric
              CASE
                  WHEN TRIM(cd."Incremental_rev") ~ '^-?[0-9]+(\.[0-9]+)?$' THEN TRIM(cd."Incremental_rev")::numeric
                  ELSE 0
              END
          ) FILTER (
                  WHERE
                      CASE
                          WHEN cd."Deployment_date" SIMILAR TO '__-__-____%' THEN to_date(cd."Deployment_date", 'DD-MM-YYYY')
                          WHEN cd."Deployment_date" SIMILAR TO '____-__-____%' THEN to_date(SUBSTRING(cd."Deployment_date" FOR 10), 'YYYY-MM-DD')
                          ELSE NULL
                      END = c.cal_date
              ), 0) AS increment_rev
      FROM calendar c
      LEFT JOIN campaigndata cd
          ON
               CASE
                   WHEN cd."Deployment_date" SIMILAR TO '__-__-____%' THEN to_date(cd."Deployment_date", 'DD-MM-YYYY')
                   WHEN cd."Deployment_date" SIMILAR TO '____-__-____%' THEN to_date(SUBSTRING(cd."Deployment_date" FOR 10), 'YYYY-MM-DD')
                   ELSE NULL
               END = c.cal_date
      GROUP BY c.cal_date
      ORDER BY c.cal_date;
    """
    
    df = pd.read_sql(text(sql), engine)
    df.rename(columns={'cal_date': 'date', 'increment_rev': 'IncrementalRev'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def get_incremental_rev_full(engine, date_range):
    """
    Align IncrementalRev to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_incremental_rev_full → Start Date: {start_date}, End Date: {end_date}")
  
    df_rev = get_incremental_rev(engine)
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(df_rev, on="date", how="left")
    merged["IncrementalRev"] = merged["IncrementalRev"].fillna(0).astype(float)
    merged.rename(columns={"IncrementalRev": "campaignincrementalrev"}, inplace=True)
    merged["date"] = merged["date"].dt.strftime("%d/%m/%Y")

    return merged

def get_campaign_details(engine):
    """
    Returns daily aggregated campaign details including:
    - Campaign_Name
    - Deployment_date (CampaignStartDate)
    - Deployment_date + 5 days (CampaignEndDate)
    """
    sql = """
    WITH formatted_data AS (
        SELECT
            CASE
                WHEN "Deployment_date" SIMILAR TO '__-__-____%' THEN to_date("Deployment_date", 'DD-MM-YYYY')
                WHEN "Deployment_date" SIMILAR TO '____-__-____%' THEN to_date(SUBSTRING("Deployment_date" FOR 10), 'YYYY-MM-DD')
                ELSE NULL
            END AS deploy_date,
            "Campaign_Name",
            "Channel",
            "Brand",
            "Region",
            CASE 
                WHEN TRIM("Sale_Value1") ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN "Sale_Value1"::numeric
                ELSE 0
            END AS sale_value1,
            CASE 
                WHEN TRIM("Conversion1") ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN "Conversion1"::numeric
                ELSE 0
            END AS conversion1
        FROM campaigndata
        WHERE "Deployment_date" IS NOT NULL
    )
    SELECT 
        deploy_date AS date,
        STRING_AGG(DISTINCT "Campaign_Name", ', ') AS CampaignName,
        STRING_AGG(DISTINCT "Channel", ', ') AS Channel,
        STRING_AGG(DISTINCT "Brand", ', ') AS Brand,
        STRING_AGG(DISTINCT "Region", ', ') AS Region,
        SUM(sale_value1) AS SaleValue1,
        AVG(conversion1) AS Conversion1,
        deploy_date AS CampaignStartDate,
        (deploy_date + INTERVAL '5 days') AS CampaignEndDate
    FROM formatted_data
    GROUP BY deploy_date
    ORDER BY deploy_date;
    """

    df = pd.read_sql(text(sql), engine)
    df["date"] = pd.to_datetime(df["date"])
    df["campaignstartdate"] = pd.to_datetime(df["campaignstartdate"])
    df["campaignenddate"] = pd.to_datetime(df["campaignenddate"])

    return df

def get_campaign_details_full(engine, date_range):
    """
    Align campaign-related fields to full date range.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_campaign_details_full → Start Date: {start_date}, End Date: {end_date}")
  
    df_details = get_campaign_details(engine)

    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    merged = date_df.merge(df_details, on="date", how="left")

    merged.columns = [col.strip().lower() for col in merged.columns]

    defaults = {
        "campaignname": "N/A",
        "channel": "N/A",
        "brand": "N/A",
        "region": "N/A",
        "salevalue1": 0.0,
        "conversion1": 0.0,
        "campaignstartdate": pd.NaT,
        "campaignenddate": pd.NaT
    }

    for col, default in defaults.items():
        if col not in merged.columns:
            merged[col] = default
        else:
            merged[col] = merged[col].fillna(default)

    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%d/%m/%Y")

    merged.rename(columns={
        "campaignname": "CampaignName",
        "channel": "CampaignChannel",
        "brand": "CampaignBrand",
        "region": "CampaignRegion",
        "salevalue1": "CampaignSaleValue1",
        "conversion1": "CampaignConversion1",
        "campaignstartdate": "CampaignStartDate",
        "campaignenddate": "CampaignEndDate"
    }, inplace=True)

    return merged


# ---------------- Previous Month Base ----------------
def get_previous_month_base_full(engine, date_range):
    """
    Adds PreviousMonthBase column with fixed value on June 1, 2025.
    """
    start_date = date_range.min()
    end_date = date_range.max()
    print(f"📌 get_previous_month_base_full → Start Date: {start_date}, End Date: {end_date}")
  
    date_df = pd.DataFrame({"date": pd.to_datetime(date_range, format="%d/%m/%Y")})
    date_df["PreviousMonthBase"] = 0
    target_date = pd.to_datetime("2025-06-01")
    date_df.loc[date_df["date"] == target_date, "PreviousMonthBase"] = 32980

    return date_df

def save_to_consolidated_table(df, engine):
    """
    Smart consolidated uploader:
      - Normalizes column names
      - Renames incoming fields to match DB schema
      - Checks DB schema
      - Detects max existing `date`
      - If empty → take rows >= 01-04-2021
      - Else → append only rows > max(date)
      - Fast insert using to_sql
    """

    START_DATE = datetime(2021, 4, 1)

    # --------------------------------------------------------------------
    # 1️⃣ Normalize incoming column names
    # --------------------------------------------------------------------
    df.columns = [
        str(col).strip().lower().replace(" ", "_") for col in df.columns
    ]
    # 1️⃣ Normalize incoming column names
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    # ✅ Print all columns
    print("Columns after normalization:", df.columns.tolist())

    # ✅ Print min/max date if 'date' column exists
    if "date" in df.columns:
        print("Min date in DataFrame:", df["date"].min())
        print("Max date in DataFrame:", df["date"].max())

    # --------------------------------------------------------------------
    # 2️⃣ Apply your rename mapping
    # --------------------------------------------------------------------
    rename_mapping = {
        "newcustomercount": "newcustomercount",
        "repeat_customers": "repeatcustomercount",
        "new_customer_sales": "newcustomersales",
        "repeat_customer_sales": "repeatcustomersales",
        "birthdaycustomersales": "birthdaycustomersales",
        "annvcustomersales": "annvcustomersales",
        "transactedcustomersales": "transactedcustomersales",
        "birthday_customers": "birthdaycustomercount",
        "anniversary_customers": "anncustomercount",
        "totalcustomertransactions": "totalcustomertransactions",
        "customerbirthdaysperday": "customerbirthdaysperday",
        "customeranniversariesperday": "customeranniversariesperday",
        "enrolledcustomercount": "enrolledcustomercount",
        "membersarchived": "membersarchived",
        "crosschannelbuyers": "crosschannelbuyers",
        "diamondenthusiasts": "diamondenthusiasts",
        "dormantcustomers": "dormantcustomers",
        "lowpointbalancecustomers": "lowpointbalancecustomers",
        "activecustomers": "activecustomers",
        "campaignsdeployed": "campaignsdeployed",
        "buyers": "campaignbuyers",
        "incrementalrev": "campaignincrementalrev",
        "channel": "campaignchannel",
        "brand": "campaignbrand",
        "region": "campaignregion",
        "salevalue1": "campaignsalevalue1",
        "conversion1": "campaignconversion1",
        "targetedcount": "campaigntargetedcount",
        "previousmonthbase": "previousmonthbase",

        # New fields
        "campaignname": "campaignname",
        "campaignstartdate": "campaignstartdate",
        "campaignenddate": "campaignenddate",
    }

    df = df.rename(columns=rename_mapping)

    # --------------------------------------------------------------------
    # 3️⃣ Fetch database schema columns
    # --------------------------------------------------------------------
    inspector = inspect(engine)
    db_columns = [col['name'] for col in inspector.get_columns('consolidateddata', schema='public')]
    required_columns = set(db_columns) - {"id"}  # Drop serial PK

    # --------------------------------------------------------------------
    # 4️⃣ Convert & standardize date columns
    # --------------------------------------------------------------------
    if "date" not in df.columns:
        raise ValueError("Missing required column: 'date'")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "campaignstartdate" in df.columns:
        df["campaignstartdate"] = pd.to_datetime(df["campaignstartdate"], errors="coerce")

    if "campaignenddate" in df.columns:
        df["campaignenddate"] = pd.to_datetime(df["campaignenddate"], errors="coerce")

    # --------------------------------------------------------------------
    # 5️⃣ Get max date from DB
    # --------------------------------------------------------------------
    with engine.begin() as conn:
        max_date_result = conn.execute(text("SELECT MAX(date) FROM public.consolidateddata")).fetchone()

    max_date = max_date_result[0] if max_date_result and max_date_result[0] else None
    print(f"max_date:{max_date}")
    # --------------------------------------------------------------------
    # 6️⃣ Filter rows based on date rules
    # --------------------------------------------------------------------
    if max_date is None:
        df = df[df["date"] >= START_DATE]
    else:
        df = df[df["date"] > pd.to_datetime(max_date)]

    # No rows left? Exit gracefully.
    if df.empty:
        return {"message": "No new consolidated rows to insert.", "rows_inserted": 0}
    start_insert_date = df["date"].min()
    end_insert_date = df["date"].max()
    print(f"📌 Saving consolidated rows → From: {start_insert_date} To: {end_insert_date}")

    # --------------------------------------------------------------------
    # 7️⃣ Column validation: missing & extra columns
    # --------------------------------------------------------------------
    df_cols_set = set(df.columns)

    missing_cols = required_columns - df_cols_set
    extra_cols = df_cols_set - required_columns

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if extra_cols:
        print(f"Warning: Extra columns ignored: {extra_cols}")

    df = df[list(required_columns)]  # Exact DB order except 'id'

    # --------------------------------------------------------------------
    # 8️⃣ Insert into DB
    # --------------------------------------------------------------------
    df.to_sql(
        "consolidateddata",
        engine,
        schema="public",
        if_exists="append",
        index=False,
        method="multi"
    )

    return {
        "message": f"{len(df)} records inserted into consolidateddata.",
        "rows_inserted": len(df)
    }

# ---------------- Helper ----------------
def ensure_datetime(df, col="Date", fmt="%d/%m/%Y"):
    """Ensure a column is in datetime64[ns] format."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
    return df

def consolidated_data_prepare():
    engine = get_db_engine()

    dates = get_date_range()  
    print(f"📌 ETL Date Range → Start: {dates[0]} | End: {dates[-1]}")

    # 1️⃣ Create full date range as datetime
    df_dates = pd.DataFrame({"Date": pd.to_datetime(dates, format="%d/%m/%Y")})
    START_DATE = df_dates["Date"].min()  # Ensure first date is correct
    END_DATE = df_dates["Date"].max()

    metrics_funcs = [
        get_new_customer_count_full,
        get_repeat_customer_count_full,
        get_new_customer_sales_full,
        get_repeat_customer_sales_full,
        get_birthday_customer_count_full,
        get_birthday_customer_sales_full,
        get_anniversary_customer_count_full,
        get_anniversary_customer_sales_full,
        get_total_customer_transactions_full,
        get_transacted_customer_sales_full,
        get_customer_birthdays_perday_full,
        get_customer_anniversaries_perday_full,
        get_daily_enrolled_customers_full,
        get_members_archived_full,
        get_cross_channel_buyers_full,
        get_diamond_enthusiasts_full,
        get_dormant_customers_full,
        get_low_point_balance_customers_full,
        get_active_customers_full,
        get_campaigns_deployed_full,
        get_targeted_count_full,
        get_buyers_full,
        get_incremental_rev_full,
        get_campaign_details_full,  
        get_previous_month_base_full
    ]

    metric_dfs = []
    for func in metrics_funcs:
        start_time = time.time()
        df = func(engine, dates)
        end_time = time.time()
        duration = end_time - start_time
        print(f"⏱️ {func.__name__} executed in {duration:.4f} seconds")
        
        # Ensure datetime column
        df.rename(columns={"date": "Date"}, inplace=True)
        df = ensure_datetime(df, "Date", "%d/%m/%Y")
        metric_dfs.append(df)

    # 2️⃣ Merge metrics onto full date range
    df = df_dates.copy()
    for mdf in metric_dfs:
        df = df.merge(mdf, on="Date", how="left")

    # 3️⃣ Fill missing rows with zeros for numeric columns
    for col in df.select_dtypes(include=["float", "int"]).columns:
        df[col] = df[col].fillna(0)

    # Optional: Fill missing object columns with empty string
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("")

    # 4️⃣ Force Date column to datetime before saving
    df["Date"] = pd.to_datetime(df["Date"])

    # 5️⃣ Print final min/max date to verify
    print(f"📌 Consolidated DF → Start: {df['Date'].min()} | End: {df['Date'].max()}")

    # 6️⃣ Save to consolidated table
    save_to_consolidated_table(df, engine)

    # 7️⃣ Optional: Excel output
    output_folder = "output_excel"
    os.makedirs(output_folder, exist_ok=True)
    for f in glob.glob(os.path.join(output_folder, "*.xlsx")):
        os.remove(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_folder, f"consolidated_data_{timestamp}.xlsx")
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="ConsolidatedData")
        df_dates.to_excel(writer, index=False, sheet_name="DateRange")
    
    return output_file, f"consolidated_data_{timestamp}.xlsx"
