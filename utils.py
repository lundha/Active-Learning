
from dataloader import PlanktonDataSet, Resize, Normalize, ToTensor, Convert2RGB
from torchvision import transforms, utils

def load_data_pool(data_dir, header_file, filename, file_ending, num_classes):
    
    composed = transforms.Compose([Convert2RGB(), Resize(224), Normalize(), ToTensor()])
        
    try:
        dataset = PlanktonDataSet(data_dir=data_dir, header_file=header_file, csv_file=filename, file_ending=file_ending,
                                    transform=composed, num_classes=num_classes)
    except Exception as e:
        print(f"Could not load dataset")

    return dataset

def calculate_distance_matrix():
    print('calculate distance matrix')
    t_start = datetime.now()
    dist_mat = np.matmul(embedding, embedding.transpose())
    sq = np.array(dist_mat.diagonal()).reshape(len(self.X), 1)
    dist_mat *= -2
    dist_mat += sq
    dist_mat += sq.transpose()
    dist_mat = np.sqrt(dist_mat)
    print(datetime.now() - t_start)



if __name__ == "__main__":
    print("Hello")