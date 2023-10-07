import os
import zipfile
from tqdm import tqdm
import gdown

url = "https://drive.google.com/file/d/1vQA62M_K5_djmfDWzJ3TD5jYS8Xi3AVV/view?usp=sharing"
gdown.download(url=url, quiet=False, fuzzy=True)

with zipfile.ZipFile('images2.zip', 'r') as zip_ref:
    file_list = zip_ref.namelist()

    with tqdm(total=len(file_list), desc="Extracting files") as pbar:
        for file in file_list:
            os.makedirs('data', exist_ok=True)
            zip_ref.extract(file, path='data')
            pbar.update(1)
