import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
import pandas as pd
from utils import get_parent_path, save_csv
import torch.nn.functional as F


def main(config):
    logger = config.get_logger('test')
    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        config['data_loader']['args']['csv_file'],
        batch_size=32,
        shuffle=False,
        validation_split=0.0,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # load sample submission
    sample_submission = pd.read_csv(get_parent_path() / 'data' / 'sample_submission.csv')
    class_names = sample_submission.columns[1:]
    
    train_mapped = pd.read_csv(get_parent_path() / 'data' / 'train_csv' / 'train_mapped.csv')
    label_map_df = train_mapped[['label_index', 'label']].drop_duplicates().sort_values('label_index')
    class_names = label_map_df['label'].tolist()
    
    results = []

    with torch.no_grad():
        for data, ids in tqdm(data_loader):
            data = data.to(device)
            outputs = model(data)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            
            for prob in probs:
                result = {
                    class_names[i]: prob[i].item()
                    for i in range(len(class_names))
                }
                results.append(result)
                

    pred = pd.DataFrame(results)
    sample_submission_path = get_parent_path() / 'data' / 'sample_submission.csv'
    submission = pd.read_csv(sample_submission_path, encoding='utf-8-sig')

    class_columns = submission.columns[1:]
    pred = pred[class_columns]

    submission[class_columns] = pred.values
    save_csv(submission, 'submission.csv')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
