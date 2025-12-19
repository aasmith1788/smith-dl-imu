"""
Quick summary statistics for demographic data
Generates descriptive statistics for age, sex, height, and weight
"""

import pandas as pd
import numpy as np
from os.path import join

# File path
DEMOGRAPHICS_PATH = r'R:\KumarLab3\PROJECTS\wesens\Data\Analysis\smith_dl\IMU Deep Learning\IMUforKnee\preperation\demographics.xlsx'

def demographic_summary():
    """Generate and print summary statistics for demographics."""

    # Load demographics
    print("="*80)
    print("DEMOGRAPHIC SUMMARY STATISTICS")
    print("="*80)
    print(f"Loading data from: {DEMOGRAPHICS_PATH}\n")

    df = pd.read_excel(DEMOGRAPHICS_PATH)

    print(f"Total participants: {len(df)}")
    print(f"Columns available: {list(df.columns)}\n")

    # Identify demographic columns
    age_col = 'age' if 'age' in df.columns else None
    height_col = 'height' if 'height' in df.columns else None
    weight_col = 'weight_bl' if 'weight_bl' in df.columns else ('weight' if 'weight' in df.columns else None)
    sex_col = 'sex' if 'sex' in df.columns else ('gender' if 'gender' in df.columns else None)

    # Summary for continuous variables
    print("="*80)
    print("CONTINUOUS VARIABLES")
    print("="*80)

    continuous_vars = []
    if age_col:
        continuous_vars.append(age_col)
    if height_col:
        continuous_vars.append(height_col)
    if weight_col:
        continuous_vars.append(weight_col)

    if continuous_vars:
        summary = df[continuous_vars].describe()
        print(summary)
        print()

        # Additional statistics
        print("-"*80)
        print("ADDITIONAL STATISTICS")
        print("-"*80)
        for col in continuous_vars:
            data = df[col].dropna()
            print(f"\n{col.upper()}:")
            print(f"  N (valid):       {len(data)}")
            print(f"  Missing:         {df[col].isna().sum()}")
            print(f"  Mean ± SD:       {data.mean():.2f} ± {data.std():.2f}")
            print(f"  Median [IQR]:    {data.median():.2f} [{data.quantile(0.25):.2f} - {data.quantile(0.75):.2f}]")
            print(f"  Range:           {data.min():.2f} - {data.max():.2f}")

    # Summary for categorical variables (sex/gender)
    if sex_col:
        print("\n" + "="*80)
        print("CATEGORICAL VARIABLES")
        print("="*80)
        print(f"\n{sex_col.upper()} DISTRIBUTION:")
        print("-"*80)

        sex_counts = df[sex_col].value_counts()
        sex_percentages = df[sex_col].value_counts(normalize=True) * 100

        for category in sex_counts.index:
            count = sex_counts[category]
            pct = sex_percentages[category]
            print(f"  {category}: {count} ({pct:.1f}%)")

        missing = df[sex_col].isna().sum()
        if missing > 0:
            print(f"  Missing: {missing}")

    # BMI calculation if height and weight available
    if height_col and weight_col:
        print("\n" + "="*80)
        print("BODY MASS INDEX (BMI)")
        print("="*80)

        # Assuming height in cm, weight in kg
        # BMI = weight (kg) / (height (m))^2
        df_calc = df[[height_col, weight_col]].dropna()
        bmi = df_calc[weight_col] / (df_calc[height_col] / 100) ** 2

        print(f"\nN (valid pairs):  {len(bmi)}")
        print(f"Mean ± SD:        {bmi.mean():.2f} ± {bmi.std():.2f}")
        print(f"Median [IQR]:     {bmi.median():.2f} [{bmi.quantile(0.25):.2f} - {bmi.quantile(0.75):.2f}]")
        print(f"Range:            {bmi.min():.2f} - {bmi.max():.2f}")

        # BMI categories
        print("\nBMI Categories:")
        print("-"*80)
        underweight = (bmi < 18.5).sum()
        normal = ((bmi >= 18.5) & (bmi < 25)).sum()
        overweight = ((bmi >= 25) & (bmi < 30)).sum()
        obese = (bmi >= 30).sum()

        total = len(bmi)
        print(f"  Underweight (<18.5):     {underweight} ({underweight/total*100:.1f}%)")
        print(f"  Normal (18.5-24.9):      {normal} ({normal/total*100:.1f}%)")
        print(f"  Overweight (25-29.9):    {overweight} ({overweight/total*100:.1f}%)")
        print(f"  Obese (≥30):             {obese} ({obese/total*100:.1f}%)")

    # Save summary to file
    output_file = DEMOGRAPHICS_PATH.replace('.xlsx', '_summary.txt')
    with open(output_file, 'w') as f:
        f.write("DEMOGRAPHIC SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total participants: {len(df)}\n\n")

        if continuous_vars:
            f.write("CONTINUOUS VARIABLES:\n")
            f.write("-"*80 + "\n")
            f.write(df[continuous_vars].describe().to_string())
            f.write("\n\n")

        if sex_col:
            f.write("SEX/GENDER DISTRIBUTION:\n")
            f.write("-"*80 + "\n")
            f.write(df[sex_col].value_counts().to_string())
            f.write("\n\n")

        if height_col and weight_col:
            f.write("BMI STATISTICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"Mean ± SD: {bmi.mean():.2f} ± {bmi.std():.2f}\n")
            f.write(f"Median: {bmi.median():.2f}\n")
            f.write(f"Range: {bmi.min():.2f} - {bmi.max():.2f}\n")

    print("\n" + "="*80)
    print(f"Summary saved to: {output_file}")
    print("="*80 + "\n")

if __name__ == "__main__":
    demographic_summary()
