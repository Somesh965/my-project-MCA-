import os
import itertools
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from glob import glob

#  CONFIG
class1 = 'no_dr'         # Domain A
class2 = 'severe_dr'     # Domain B
image_size = 128
batch_size = 4
num_epochs = 3  # Changed from 50 to 3
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Transforms
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#  Dataset
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_A, root_B, transform):
        self.A_paths = sorted(glob(os.path.join(root_A, '*')))
        self.B_paths = sorted(glob(os.path.join(root_B, '*')))
        self.transform = transform

    def __getitem__(self, index):
        A_img = Image.open(self.A_paths[index % len(self.A_paths)]).convert('RGB')
        B_img = Image.open(self.B_paths[index % len(self.B_paths)]).convert('RGB')
        return {
            'A': self.transform(A_img),
            'B': self.transform(B_img)
        }

    def __len__(self):
        return max(len(self.A_paths), len(self.B_paths))

dataset = ImageDataset(
    root_A=r"S:\Final Year Project\diabetic-retinopathy-ai\data\diabetic_retinopathy\NO_DR",
    root_B=r"S:\Final Year Project\diabetic-retinopathy-ai\data\diabetic_retinopathy\severe_DR",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#  Generator
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_residual_blocks=9):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResnetBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

#  Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)

#  CycleGAN Models
G_AB = GeneratorResNet(3, 3).to(device)
G_BA = GeneratorResNet(3, 3).to(device)
D_A = Discriminator(3).to(device)
D_B = Discriminator(3).to(device)

#  Losses
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(beta1, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

#  Training Loop
print(" Starting CycleGAN Training...")
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        # Get discriminator output shape
        pred_shape = D_A(real_A).shape

        valid = torch.ones(pred_shape, device=device)
        fake = torch.zeros(pred_shape, device=device)

        # ----------------------
        #  Train Generators
        # ----------------------
        optimizer_G.zero_grad()

        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_G = loss_GAN_AB + loss_GAN_BA + 10.0 * (loss_cycle_A + loss_cycle_B)
        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        optimizer_D_A.zero_grad()
        loss_real = criterion_GAN(D_A(real_A), valid)
        loss_fake = criterion_GAN(D_A(fake_A.detach()), fake)
        loss_D_A = (loss_real + loss_fake) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        optimizer_D_B.zero_grad()
        loss_real = criterion_GAN(D_B(real_B), valid)
        loss_fake = criterion_GAN(D_B(fake_B.detach()), fake)
        loss_D_B = (loss_real + loss_fake) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()

        if i % 50 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i}/{len(dataloader)}] "
                  f"[D_A loss: {loss_D_A.item():.4f}] [D_B loss: {loss_D_B.item():.4f}] "
                  f"[G loss: {loss_G.item():.4f}]")

    # Save sample output
    os.makedirs(f"gan_outputs/cyclegan_{class1}_to_{class2}", exist_ok=True)
    save_image(fake_B.data[:4], f"gan_outputs/cyclegan_{class1}_to_{class2}/epoch_{epoch+1}.png", normalize=True)

# Save models
torch.save(G_AB.state_dict(), f"gan_outputs/cyclegan_{class1}_to_{class2}/G_AB.pth")
torch.save(G_BA.state_dict(), f"gan_outputs/cyclegan_{class1}_to_{class2}/G_BA.pth")
print(" CycleGAN training complete and models saved!")
