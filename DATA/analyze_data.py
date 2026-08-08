import pandas as pd
import glob
import os

data_dir = '/Users/ntgr/Desktop/This.Mac/MYPROJECT/BENI_PROJECT/DATA'
all_files = glob.glob(os.path.join(data_dir, "National exam_dataset_S3_*.csv"))

df_list = []
for file in all_files:
    try:
        df = pd.read_csv(file)
        df_list.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not df_list:
    print("No data found.")
    exit()

combined_df = pd.concat(df_list, ignore_index=True)

# Standardize column names
combined_df.columns = combined_df.columns.str.strip().str.lower()

print("## Dataset Overview")
print(f"Total Records: {len(combined_df)}")
print(f"Columns: {', '.join(combined_df.columns)}")
print(f"Years covered: {sorted(combined_df['school year'].unique()) if 'school year' in combined_df.columns else 'N/A'}")

print("\n## Missing Values")
print(combined_df.isnull().sum().to_string())

print("\n## Summary Statistics (Numeric columns)")
print(combined_df.describe().to_string())

subjects = [col for col in combined_df.columns if 'grade' in col]

if subjects and 'school year' in combined_df.columns:
    print("\n## Average Grades by Year")
    yearly_avg = combined_df.groupby('school year')[subjects].mean()
    print(yearly_avg.to_string())

if 'option' in combined_df.columns:
    print("\n## Top Options by Student Count")
    print(combined_df['option'].value_counts().head(10).to_string())

    if subjects:
        print("\n## Average Grades by Option (Top 5 options)")
        top_options = combined_df['option'].value_counts().head(5).index
        option_avg = combined_df[combined_df['option'].isin(top_options)].groupby('option')[subjects].mean()
        print(option_avg.to_string())
