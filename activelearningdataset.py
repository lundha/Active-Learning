import numpy as np



# Split into training and test set
# Split into labeled and unlabeled data
# Move data from unlabeled to labeled
class Dataset():
    pass



class ActiveLearningDataset:

    def __init__(self, X,Y, init_labeled):
        self.X = X
        self.Y = Y
        self.index = {'labeled': np.arange(0), 'unlabeled': np.arange(len(X))} 
        self.index['labeled'], self.index['unlabeled'] = self.split_into_labeled_unlabeled(init_labeled)
        self.class_count = np.zeros(10) # 10 number of classes

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
        for idx in queried_idxs:
            self.class_count[self.Y[idx]] += 1

        return self.class_count

    def move_from_unlabeled_to_labeled(self, idxs_unlabeled, strategy):
        '''
        Maps local index of unlabeled data point to global index w.r.t X, and moves index from unlabeled to train
        :param: idxs to be moved
        :return: Data point(s) and label(s) of the data corresponding to the moved index/indices.
        '''
    
        if not isinstance(idxs_unlabeled, list):
            idxs_unlabeled = list(idxs_unlabeled)
        
        if (type(strategy).__name__ in ['DFAL', 'coreset', 'Random_Strategy', 'Uncertainty_Strategy', 'Max_Entropy_Strategy', 'ActiveLearningByLearning']):
        

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