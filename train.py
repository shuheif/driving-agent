from pytorch_lightning import LightningModule, Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim, nn
from torchvision.models import resnet18, ResNet18_Weights
import argparse

from utils import cross_entropy_for_onehot, accuracy

NUM_CLASSES = 4 # left, right, forward, stop


class DriveCommandClassifier(LightningModule):
    def __init__(self, weights=None):
        super().__init__()
        self.resnet = resnet18(weights=weights)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, NUM_CLASSES)
        self.loss_module = cross_entropy_for_onehot

    def forward(self, images):
        return self.resnet(images)
    
    def training_step(self, batch, batch_idx):
        return self._evaluate(batch, "train")
    
    def test_step(self, batch, batch_idx) -> None:
        self._evaluate(batch, "test")
    
    def validation_step(self, batch, batch_idx) -> None:
        self._evaluate(batch, "val")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
    def _evaluate(self, batch, stage: str):
        images, labels = batch
        preds = self.resnet(images)
        loss = self.loss_module(preds, labels)
        acc = accuracy(preds, labels)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_accuracy", acc[0], prog_bar=True)
        return loss


if __name__ == "__main__":
    from data_module import DrivingDataModule
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()
    data_module = DrivingDataModule(data_dir=args.data_dir, batch_size=args.batch_size)
    data_module.setup()
    train_loader, val_loader, test_loader = data_module.get_data_loaders()
    model = DriveCommandClassifier(weights=ResNet18_Weights.DEFAULT)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=5,
        monitor="val_loss",
        filename="resnet18-ep{epoch:02d}-vloss{val_loss:.2f}",
    )
    tb_logger = loggers.TensorBoardLogger("tb_logs", name="resnet18")
    trainer = Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.fast_dev_run,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=tb_logger,
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model=model, dataloaders=test_loader)
