import numpy as np



# Split into training and test set
# Split into labeled and unlabeled data
# Move data from unlabeled to labeled

class ActiveLearningDataset:

    def __init__(self, X_train, Y_train, X_test, Y_test, X_valid, Y_valid, init_labeled, num_classes=10):
        
        self.X = X_train
        self.Y = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.X_test = X_test
        self.Y_test = Y_test

        self.index = {'labeled': np.arange(0), 'unlabeled': np.arange(len(self.X))} 
        self.index['labeled'], self.index['unlabeled'] = self.split_into_labeled_unlabeled(init_labeled)
        self.class_count = np.zeros(num_classes) 

    def __repr__(self):
        return f"Dataset \nNumber of datapoints: {str(len(self.X))} \nNum indexes labeled: {len(self.index['labeled'])}"

    def __str__(self):
        return f"Dataset \nNumber of datapoints: {str(len(self.X))} \nNum indexes labeled: {len(self.index['labeled'])}"

    def split_data(self, ratio):
        pass

    def split_into_labeled_unlabeled(self, init_num_labeled):
        '''
        Shuffles all training data before splitting into labeled and non-labeled sets
        :param: Number of intially labeled samples 
        :return: Indices for initially labeled and non-labeled data sets
        '''
        shuffled_indices = np.random.permutation(self.index['unlabeled'])
        return shuffled_indices[:init_num_labeled], shuffled_indices[init_num_labeled:]
    
    def count_class_query(self, queried_idxs):
        '''
        Function to count how many times each class have been queried 
        '''
        try:
            for idx in queried_idxs:
                self.class_count[self.Y[idx]] += 1
        except Exception as e:
            print(str(e))
        class_count_prct = [round(elem/sum(self.class_count),2) for elem in self.class_count]

        return self.class_count, class_count_prct

    def move_from_unlabeled_to_labeled(self, idxs_unlabeled, strategy):
        '''
        Maps local index of unlabeled data point to global index w.r.t X, and moves index from unlabeled to train
        :param: idxs to be moved
        :return: Data point(s) and label(s) of the data corresponding to the moved index/indices.
        '''
    
        if not isinstance(idxs_unlabeled, list):
            idxs_unlabeled = list(idxs_unlabeled)
        
        if (type(strategy).__name__ in ['DFAL_Strategy', 'Random_Strategy', 'Uncertainty_Strategy', 'Max_Entropy_Strategy', 'ActiveLearningByLearning_Strategy']):
        

            self.index['unlabeled'] = self.index['unlabeled'][ ~np.isin(self.index['unlabeled'], idxs_unlabeled)]
            self.index['labeled'] = np.append(self.index['labeled'], idxs_unlabeled, axis=0)
        
        else:

            idx = self.index['unlabeled'][idxs_unlabeled]
            
            if not isinstance(idx, list):
                idx = list(idx)
        
            self.index['unlabeled'] = np.delete(self.index['unlabeled'], idxs_unlabeled, axis=0)
            self.index['labeled'] = np.append(self.index['labeled'], idxs_unlabeled, axis=0)

    

if __name__ == "__main__":
    pass