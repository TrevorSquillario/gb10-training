import os
import gzip
import shutil
import argparse
import requests
from pathlib import Path

def download_and_extract(target_dir):
    # Ensure the directory exists
    target_path = Path(target_dir).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    # Directory where extracted CSVs will be placed
    extracted_dir = target_path / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://github.com/DataTalksClub/nyc-tlc-data/releases/download/yellow"
    # The 2021 yellow taxi release contains months 01 through 07
    months = [f"{i:02d}" for i in range(1, 8)]
    
    for month in months:
        file_name = f"yellow_tripdata_2021-{month}.csv.gz"
        url = f"{base_url}/{file_name}"
        file_path = target_path / file_name
        output_csv = extracted_dir / f"yellow_tripdata_2021-{month}.csv"
        
        # 1. Download
        print(f"Downloading {file_name}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            
            # 2. Extract
            print(f"Extracting to {output_csv}...")
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_csv, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Optional: Remove the .gz file after extraction to save space
            # os.remove(file_path)
            
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract NYC TLC data.")
    default_dir = os.path.join(os.path.expanduser("~"), "gb10", "nyc-data")
    
    parser.add_argument(
        "--dir", 
        default=default_dir, 
        help=f"Target directory for data (default: {default_dir})"
    )
    
    args = parser.parse_args()
    download_and_extract(args.dir)
    print("\nDone!")