from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset): # Creamos un dataset simple donde almacenamos las imagenes
    def __init__(self, ds):
        temp = DataLoader(ds, len(ds))
        images, labels = next(iter(temp))
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]