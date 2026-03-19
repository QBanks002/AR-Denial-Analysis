
import pandas as pd
import numpy as np
import os

def load_and_clean(filepath):
    df = pd.read_csv(filepath)

    # Strip whitespace from all string columns
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

    # Parse dates
    df['DOS']            = pd.to_datetime(df['DOS'])
    df['Submitted Date'] = pd.to_datetime(df['Submitted Date'])
    df['Worked Date']    = pd.to_datetime(df['Worked Date'])
    df['Follow up date'] = pd.to_datetime(df['Follow up date'])

    # Tag denied claims
    df['is_denied'] = df['Status'] == 'Denied'

    # Categorize denial root causes into fixable vs. non-fixable
    fixable_codes = [
        'Incorrect Submission', 'Missing Modifiers', 'Invalid Procedure Code',
        'Provider Info Missing', 'Dx inconsistent with CPT', 'Duplicate Claim',
        'Claim not on file', 'Claim Error'
    ]
    df['is_fixable'] = df['Status Code'].isin(fixable_codes) & df['is_denied']

    # Save cleaned version
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/ar_denials_cleaned.csv', index=False)
    print(f"Cleaned data saved — {df.shape[0]} rows, {df.shape[1]} columns")

    return df

if __name__ == '__main__':
    df = load_and_clean('Synthetic_AR_Medical_Dataset_with_Realistic_Denial.csv')
