
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

