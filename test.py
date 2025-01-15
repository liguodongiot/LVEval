
from utils import (
    ensure_dir, 
    seed_everything,
    get_dataset_names,
    post_process,
    load_jsonl,
    load_LVEval_dataset,
    dump_preds_results_once,
)
import re
from tqdm import tqdm


datasets = get_dataset_names(['cmrc_mixup'], ['16k'])

for dataset in tqdm(datasets):
    datas = load_LVEval_dataset(dataset, "/data")
    dataset_name = re.split('_.{1,3}k', dataset)[0]
    print(dataset_name)



    