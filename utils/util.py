import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import torch
import warnings
from datetime import datetime


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
    
def get_parent_path():
    """
    Get the path of the parent path script.
    """
    return Path(__file__).parent.parent.resolve()
    
def submission_check(train_data, submission_data, sample_submission_data=None):
    idx_to_label = train_data.drop_duplicates(subset=['label_index']).set_index('label_index')['label'].to_dict()
    submission_data['label'] = submission_data['label'].map(idx_to_label)
    
    submission_final = sample_submission_data.copy() if sample_submission_data is not None else submission_data.copy()
    submission_final.iloc[:, 1:] = 0
    
    for i, row in submission_data.iterrows():
        class_name = row['label']
        if class_name in submission_final.columns:
            submission_final.loc[submission_final['ID'], class_name] = 1
    return submission_final

def save_csv(data, file_name):
    """
    Save DataFrame to CSV file.
    
    :param data: DataFrame to save.
    :param path: Path to save the CSV file.
    """
    if isinstance(data, pd.DataFrame):
        base_dir = get_parent_path() / 'saved' / 'results'
        make_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = Path(base_dir) / make_datetime / file_name
        
        if not save_path.parent.is_dir():
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
        data.to_csv(save_path, index=False, float_format="%.4f")
        
    else:
        warnings.warn("Provided data is not a DataFrame. Saving as empty CSV.")
        pd.DataFrame().to_csv(save_path, index=False, float_format="%.6f")