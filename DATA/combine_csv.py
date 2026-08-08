import pandas as pd
from pathlib import Path

folder = Path(".")

output_file = "National_exam_dataset_S3_2017-2025.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    for csv_file in sorted(folder.glob("*.csv")):
        print(f"Processing {csv_file.name}")
        # Read CSV
        df = pd.read_csv(csv_file, encoding="utf-8")

        # Use filename as sheet name
        sheet_name = csv_file.stem[:31]

        # Write to Excel sheet
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

print(f"Created: {output_file}")
