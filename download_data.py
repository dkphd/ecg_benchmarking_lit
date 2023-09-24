import os
import zipfile
import requests
import tqdm

def download_file(url, filename):
    """
    Download file from a URL to a given filename with a progress bar.
    """
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}. Status code: {response.status_code}")
    
    # Get the total file size
    file_size = int(response.headers.get('content-length', 0))
    
    # Initialize the progress bar
    progress_bar = tqdm.tqdm(total=file_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            progress_bar.update(len(chunk))
            f.write(chunk)
    
    progress_bar.close()

def download_and_unzip_data():
    # Create a new directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Change to the 'data' directory
    os.chdir("data")
    
    # Define URL and filename
    url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    filename = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    
    # Download the file
    download_file(url, filename)
    
    # Unzip the file
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    # Rename the folder
    os.rename("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3", "ptbxl")



if __name__ == "__main__":
    download_and_unzip_data()
