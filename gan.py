import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

import matplotlib.pyplot as plt

import pytorch_lightning as pl

MAX_EPOCHS = 100
BATCH_SIZE = 128
NUM_WORKERS = int(os.cpu_count() / 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
])

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", 
                 batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)

class Discriminator(nn.Module):
    def __init__(self, input_dim=1, img_size=28):
        super(Discriminator, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.ffwd = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.cnn(img)
        x = x.view(-1, 320)
        x = self.ffwd(x)
        return x

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.fc = nn.Linear(latent_dim, 7*7*64)

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=7)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 7, 7)
        x = self.conv_blocks(x)
        return x

class GAN(pl.LightningModule):
    def __init__(self, latent_dim=100, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(latent_dim=self.hparams.latent_dim)
        self.discriminator = Discriminator()

        self.loss_fn = nn.BCELoss()

        self.val_z = torch.randn(6, self.hparams.latent_dim)

        self.automatic_optimization = False  # Disable automatic optimization

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        optimizer_g, optimizer_d = self.optimizers()

        z = torch.randn(imgs.size(0), self.hparams.latent_dim)
        z = z.type_as(imgs)

        # Train generator
        optimizer_g.zero_grad()
        fake_imgs = self(z)
        disc_pred = self.discriminator(fake_imgs)
        true = torch.ones(imgs.size(0), 1).type_as(imgs)
        loss_g = self.loss_fn(disc_pred, true)
        self.manual_backward(loss_g)
        optimizer_g.step()

        self.log("loss_g", loss_g, prog_bar=True)

        # Train discriminator
        optimizer_d.zero_grad()
        pred_real = self.discriminator(imgs)
        true = torch.ones(imgs.size(0), 1).type_as(imgs)
        real_loss = self.loss_fn(pred_real, true)

        pred_fake = self.discriminator(fake_imgs.detach())
        fake = torch.zeros(imgs.size(0), 1).type_as(imgs)
        fake_loss = self.loss_fn(pred_fake, fake)

        loss_d = (real_loss + fake_loss) / 2
        self.manual_backward(loss_d)
        optimizer_d.step()

        self.log("loss_d", loss_d, prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [optimizer_g, optimizer_d], []

    def plot_imgs(self):
        z = self.val_z.type_as(self.generator.fc.weight)
        sample_imgs = self(z).cpu()

        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.axis('off')
        plt.savefig(f'epoch_{self.current_epoch}.png')
        plt.close()

    #def on_epoch_end(self):
    def on_train_epoch_end(self, unused=None):
        print(f'Saving images for epoch {self.current_epoch}')  # Debugging line
        self.plot_imgs()

if __name__ == "__main__":
    datamodule = MNISTDataModule()
    model = GAN()

    trainer = pl.Trainer(max_epochs=MAX_EPOCHS)
    trainer.fit(model, datamodule)
