import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
import pandas as pd
from utils import get_parent_path


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

    predictions = []
    image_ids = []

    with torch.no_grad():
        for data, file_name in tqdm(data_loader, desc='Testing'):
            data = data.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy().tolist())
            image_ids.extend(file_name)
            
    # Save predictions to a CSV file
    submission = pd.DataFrame({
        'image_id': image_ids,
        'label': predictions
    })
    output_file = get_parent_path() / config['data_loader']['args']['data_dir'] / 'submission.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_file, index=False)
    
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
