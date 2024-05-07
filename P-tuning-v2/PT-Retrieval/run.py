import json
import os
with open('{oagqa_all_data_path}', 'r') as fr:
    oagqa_data_files = json.loads(fr.read())

for k,v in oagqa_data_files.items():
    os.system(f"bash eval_scripts/evaluate_oagqav2.sh {k} 20")