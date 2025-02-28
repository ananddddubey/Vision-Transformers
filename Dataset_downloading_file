import os
import requests

# Ensure the directory exists
os.makedirs("./content", exist_ok=True)

# Define the URLs and corresponding file paths
files = {
    "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat": "./content/Indian_pines_corrected.mat",
    "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat": "./content/Indian_pines_gt.mat",
    "https://github.com/gokriznastic/HybridSN/raw/master/data/Salinas_corrected.mat": "./content/Salinas_corrected.mat",
    "https://github.com/gokriznastic/HybridSN/raw/master/data/Salinas_gt.mat": "./content/Salinas_gt.mat",
    "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat": "./content/PaviaU.mat",
    "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat": "./content/PaviaU_gt.mat"
}

# Download files using Python
for url, path in files.items():
    if not os.path.isfile(path):  # Check if file already exists
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, "wb") as file:
                file.write(response.content)
            print(f"Saved to {path}")
        else:
            print(f"Failed to download {url}, Status Code: {response.status_code}")
    else:
        print(f"File already exists: {path}")
