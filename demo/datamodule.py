import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from pytorch_lightning import LightningDataModule
import openml


class StandardScaler:
    def __init__(self, mean=None, scale=None, epsilon=1e-7):
        self.mean_ = mean
        self.scale_ = scale
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean_ = torch.mean(values, dim=dims)
        self.scale_ = torch.std(values, dim=dims)
        return self

    def transform(self, values):
        return (values - self.mean_) / (self.scale_ + self.epsilon)

    def inverse_transform(self, values):
        return values * self.scale_ + self.mean_


class ScaledDataset(Dataset):
    def __init__(self, dataset, scaler_x, scaler_y):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def scale(self, v, scaler):
        shape = v.shape
        scaled = scaler.transform(v)
        return scaled.reshape(shape)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.scale(x, self.scaler_x)
        y = self.scale(y, self.scaler_y)
        return x, y


class CustomDataModule(LightningDataModule):
    def __init__(self, ds_id):
        super().__init__()
        
        # Dataset loading with OpenML
        ds = openml.datasets.get_dataset(ds_id)
        x, y, _, _ = ds.get_data(dataset_format='dataframe', target=ds.default_target_attribute)

        # Preprocessing
        x = x.select_dtypes(['number'])
        x = x.dropna(axis='columns', how='any')
        x = torch.from_numpy(x.to_numpy('float32'))
        y = torch.from_numpy(y.to_numpy('float32'))
        
        # Splitting
        data = TensorDataset(x, y)
        data_train, data_val, data_calib, data_test = random_split(
            dataset=data,
            lengths=[0.5, 0.1, 0.15, 0.25],
            generator=torch.Generator().manual_seed(0),
        )

        # Scaling
        x, y = data_train[:]
        x_scaler = StandardScaler().fit(x)
        y_scaler = StandardScaler().fit(y)
        self.x_size = x.shape[1]
        self.data_train = ScaledDataset(data_train, x_scaler, y_scaler)
        self.data_val = ScaledDataset(data_val, x_scaler, y_scaler)
        self.data_calib = ScaledDataset(data_calib, x_scaler, y_scaler)
        self.data_test = ScaledDataset(data_test, x_scaler, y_scaler)
    
    def get_dataloader(self, dataset, shuffle=False):
        return DataLoader(dataset=dataset, num_workers=1, batch_size=512, shuffle=shuffle)

    def train_dataloader(self):
        return self.get_dataloader(self.data_train, shuffle=True)
    
    def val_dataloader(self):
        return self.get_dataloader(self.data_val)
    
    def calib_dataloader(self):
        return self.get_dataloader(self.data_calib)
    
    def test_dataloader(self):
        return self.get_dataloader(self.data_test)
