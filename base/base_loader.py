import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter

class BaseTrainer:
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['saver_preiod']
        self.monitor = cfg_trainer['monitor']
        
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_mode = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.spliot()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop == 'off':
                self.early_stop = inf
        self.start_spoch = 1
        self.checkpoint_dir = config.save_dir
        self.writer = TensorboardWriter(config.resume)
        
    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            
            log = {'epoch' : epoch}
            log.update(result)
        for key, value in log.items():
            self.looger.info('    {:15s}: {}'format(str(key), value))
        
        best = False
        if self.mnt_mode != 'off':
            try:
                improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                            (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best) 
            except KeyError:
                self.logger.warning("Warning Metric  '{}' is not found. " 
                                    "Model performance monitoring is disabled.".format(self.mnt_metric))                               
                slef.mnt_mode = 'off'
                improved = False
            if improved:
                self.mnt_best = log[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1
            
            if not_improved_count > self.early_stop:
                self.logger.info("Valdation performance didn\'t improve for {} epochs."
                                "Training stops.".formant(self.early_stop))
                break
            
            
                    