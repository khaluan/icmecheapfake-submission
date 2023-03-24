RAW_DIR = 'D:\Context_raw'
PROCESS_DIR = 'D:\Context_processed'
from tqdm import tqdm
from crawler import crawl_context_from_file
import json
from os import listdir
from os.path import isfile, join
onlyfiles = [join(RAW_DIR, f) for f in listdir(RAW_DIR) if isfile(join(RAW_DIR, f))]

for name in tqdm(onlyfiles):
    new_name = name.replace(RAW_DIR, PROCESS_DIR)
    with open(name, 'r', encoding='utf8') as ifile:
        content = ifile.read()
    
    data = crawl_context_from_file(content)

    with open(new_name, 'w+', encoding='utf8') as ofile:
        json.dump(data, ofile, ensure_ascii=False)