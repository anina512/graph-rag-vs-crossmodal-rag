# download_first_200_rows.py
import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

def main():
    api = KaggleApi()
    api.authenticate()

    dataset_identifier = "pes12017000148/food-ingredients-and-recipe-dataset-with-images"
    download_path = "input/first_200"
    os.makedirs(download_path, exist_ok=True)

    print(f"Downloading dataset {dataset_identifier} …")
    api.dataset_download_files(dataset_identifier, path=download_path, unzip=True, quiet=False)

    # Identify the CSV file. Adjust filename if needed.
    # For example: let's assume the file is named "recipes.csv"
    csv_filename = None
    for fname in os.listdir(download_path):
        if fname.lower().endswith('.csv'):
            csv_filename = os.path.join(download_path, fname)
            break

    if csv_filename is None:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")

    print(f"Found CSV file: {csv_filename}. Reading first 200 rows…")
    df = pd.read_csv(csv_filename, nrows=200)

    # Write out the subset CSV
    output_csv = os.path.join(download_path, "first_200_rows.csv")
    df.to_csv(output_csv, index=False)
    print(f"Wrote first 200 rows to {output_csv}")

    # (Optional) Remove other files if you only want the subset
    for fname in os.listdir(download_path):
        if fname not in ("first_200_rows.csv", os.path.basename(csv_filename)):
            os.remove(os.path.join(download_path, fname))
    print("Cleanup done, only subset kept.")

if __name__ == '__main__':
    main()
