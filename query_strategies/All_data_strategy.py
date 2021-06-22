from .Strategy import Strategy
from activelearningdataset import ActiveLearningDataset
import sys
from datetime import datetime

class All_Data_Strategy(Strategy):
    def __init__(self, ALD, net, args, logger, log_file, n_epochs, **kwargs):
        super().__init__(ALD, net, args, logger)
        tic = datetime.now()
        self.args['n_epoch'] = n_epochs
        self.train()
        P = self.predict(self.ALD.X_test, self.ALD.Y_test)
        Acc = round(1.0 * (self.ALD.Y_test==P).sum().item() / len(self.ALD.Y_test), 4)
        
        ##### LOGGING #####
        self.logger.debug(f"Accuracy: {Acc}")
        self.logger.info(f'*** Result from training on all data ***  Run time: {datetime.now() - tic}')
        self.logger.info(f'Accuracy: {Acc}\n')
        

        sys.exit('Finished running on all data, terminating...')

  