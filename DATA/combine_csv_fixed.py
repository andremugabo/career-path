import pandas as pd
from pathlib import Path

# Fix 1: The CSV files are located in the current directory, not in a 'national_exam' folder.
folder = Path(".")

output_file = "National_exam_dataset_S3_2017-2025.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    for csv_file in sorted(folder.glob("*.csv")):

        # Read CSV. Added low_memory=False to suppress DtypeWarnings.
        df = pd.read_csv(csv_file, encoding="utf-8", low_memory=False)

        # Fix 2: 'csv_file.stem[:31]' evaluates to 'National exam_dataset_S3_2017-2' 
        # for ALL files, which results in duplicate sheet names and overwrites them.
        # Instead, extract the year from the parenthesis at the end of the filename.
        sheet_name = csv_file.stem.split("(")[1].strip(")")

        # Write to Excel sheet
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

print(f"Created: {output_file}")
