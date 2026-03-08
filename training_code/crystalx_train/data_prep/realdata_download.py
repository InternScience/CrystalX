import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
import time

max_retries = 500
retry_delay = 2  # seconds
#base_url = 'https://www.crystallography.net/cod/result.php?CODSESSION=4pdpjptu6ta01u2d2fr61esh36&page=3&count=1000&order_by=file&order=asc'
#base_url = 'https://www.crystallography.net/cod/result.php?CODSESSION=4pdpjptu6ta01u2d2fr61esh36&page=13&count=1000&order_by=file&order=asc'
#base_url = 'https://www.crystallography.net/cod/result.php?CODSESSION=4pdpjptu6ta01u2d2fr61esh36&page=16&count=1000&order_by=file&order=asc'
#base_url = 'https://www.crystallography.net/cod/result.php?CODSESSION=4pdpjptu6ta01u2d2fr61esh36&page=32&count=1000&order_by=file&order=asc'
#base_url = 'https://www.crystallography.net/cod/result.php?CODSESSION=4pdpjptu6ta01u2d2fr61esh36&page=35&count=1000&order_by=file&order=asc'
#f'https://www.crystallography.net/cod/result.php?CODSESSION=g9f2q29tmakam2pb7lgu03b0mo&count=1000&page={str(i)}&order_by=file&order=asc'
for i in range(7):
    print(i)
    base_url = f'https://www.crystallography.net/cod/result.php?CODSESSION=ndaepisqb6usttlpo3kljfnbru&page={str(i)}&count=1000&order_by=file&order=asc'
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)

    cif_save_folder = 'Zeitschrift_cif_files'
    os.makedirs(cif_save_folder, exist_ok=True)

    hkl_save_folder = 'Zeitschrift_hkl_files'
    os.makedirs(hkl_save_folder, exist_ok=True)

    for link in links:
        href = link['href']
        if href.endswith('.cif'):
            file_url = urljoin(base_url, href)
            file_name = os.path.join(cif_save_folder, os.path.basename(href))
            # 下载文件
            for attempt in range(max_retries):
                try:
                    with requests.get(file_url, stream=True) as r:
                        r.raise_for_status()
                        with open(file_name, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192): 
                                if chunk:
                                    f.write(chunk)
                    break  # 如果下载成功，跳出循环
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed. Error: {e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Max retries reached. Download failed.")
        elif href.endswith('.hkl'):
            file_url = urljoin(base_url, href)
            file_name = os.path.join(hkl_save_folder, os.path.basename(href))
            
            # 下载文件
            with requests.get(file_url, stream=True) as r:
                r.raise_for_status()
                with open(file_name, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        if chunk:
                            f.write(chunk)

print("下载完成！")
