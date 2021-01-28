
import dataloader
import click
from utils import load_data_pool

# Load data

data_dir = "/Users/martin.lund.haug/Documents/Masteroppgave/caltech-10-classes/train/"
header_file = data_dir + "header.tfl.txt"
filename = data_dir + "image_set.data"
file_ending = ".jpg"
num_classes = 10


dataset = load_data_pool(data_dir, header_file, filename, file_ending, num_classes)

print(dataset.dataset)

# Transform data to desired form


# Create embedding

def create_embedding():

    lb_flag = self.idxs_lb.copy()
    embedding = self.get_embedding(self.X, self.Y)
    embedding = embedding.numpy()


# Calculate distance matrix

# Find greedy solution 
