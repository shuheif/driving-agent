from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from dataset import CarlaDrivingDataset

TRANSFORM_CARLA_IMAGES = [
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
]


class DrivingDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            data_dir: str,
            test_ratio: float=.1,
            val_ratio: float=.2,
            num_workers: int=4,
            seed: int=1234,
        ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.seed = seed

    def setup(self):
        full_dataset = CarlaDrivingDataset(data_dir=self.data_dir, transform=TRANSFORM_CARLA_IMAGES)
        total_size = len(full_dataset)
        test_size = int(total_size * self.test_ratio)
        train_size = total_size - test_size
        val_size = int(train_size * self.val_ratio)
        self.train_dataset, self.test_dataset = random_split(full_dataset, [train_size, test_size])
        self.val_dataset, _ = random_split(self.train_dataset, [val_size, train_size - val_size])

    def _get_dataloader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers, pin_memory=True)

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)
    
    def get_data_loaders(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()
