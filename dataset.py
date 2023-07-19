import glob
import os.path as osp
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CarlaDrivingDataset(Dataset):
    def __init__(self, data_dir: str, transform=[]):
        self.image_list = glob.glob(osp.join(data_dir, 'images/*.jpg'))
        self.log_file = pd.read_csv(glob.glob(osp.join(data_dir, 'log.csv'))[0])
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path)
        image = self.transform(image)
        frame_id = Path(image_path).stem
        drive_command = self._get_one_hot_command(frame_id)
        return image, drive_command
    
    def _get_one_hot_command(self, frame_id):
        log_at_frame = self.log_file.loc[self.log_file['frame'] == int(frame_id)]
        steer = log_at_frame['steer'].values[0] if len(log_at_frame) > 0 else 0
        throttle = log_at_frame['throttle'].values[0] if len(log_at_frame) > 0 else 0
        brake = log_at_frame['brake'].values[0] if len(log_at_frame) > 0 else 0
        longitudinal = throttle + brake * -1
        if steer > .1:
            return torch.tensor([1., 0., 0., 0.]) # right
        elif steer < -.1:
            return torch.tensor([0., 1., 0., 0.]) # left
        elif longitudinal > 0:
            return torch.tensor([0., 0., 1., 0.]) # forward
        else:
            return torch.tensor([0., 0., 0., 1.]) # stop


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse
    parser = argparse.ArgumentParser(description="dataset.py")
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    dataset = CarlaDrivingDataset(args.data_dir, transform=[transforms.ToTensor()])
    dataloader = DataLoader(dataset, batch_size=1)
    iterator = iter(dataloader)
    X, y = next(iterator)
    print('X: ', X.shape, ', y: ', y.shape)
    plt.imshow(X[0].permute(1, 2, 0))
    plt.show()
