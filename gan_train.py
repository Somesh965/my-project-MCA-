import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

# --- DCGAN Generator and Discriminator ---
class DCGAN_G(nn.Module):
    def __init__(self, nz=100, ngf=32, nc=3):
        super(DCGAN_G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class DCGAN_D(nn.Module):
    def __init__(self, nc=3, ndf=32):
        super(DCGAN_D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

# --- Placeholder for StyleGAN2 (to be replaced with real implementation) ---
class StyleGAN2_Generator(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

class StyleGAN2_Discriminator(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x

# --- Dataset and transforms ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(root, f))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# --- DCGAN Training ---
def train_dcgan(dataloader, device, epochs=3):
    print("Starting DCGAN training...")
    os.makedirs("gan_outputs/dcgan", exist_ok=True)
    G = DCGAN_G().to(device)
    D = DCGAN_D().to(device)
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)

    for epoch in range(epochs):
        total_batches = len(dataloader)
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_images = G(noise)

            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            D.zero_grad()
            output_real = D(real_images)
            lossD_real = criterion(output_real, real_labels)
            output_fake = D(fake_images.detach())
            lossD_fake = criterion(output_fake, fake_labels)
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            G.zero_grad()
            output_fake = D(fake_images)
            lossG = criterion(output_fake, real_labels)
            lossG.backward()
            optimizerG.step()

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{total_batches}] [D loss: {lossD.item():.4f}] [G loss: {lossG.item():.4f}]")

        save_image(fake_images.data, f"gan_outputs/dcgan/fake_epoch_{epoch}.png", normalize=True)

    torch.save(G.state_dict(), "gan_outputs/dcgan/generator.pth")
    torch.save(D.state_dict(), "gan_outputs/dcgan/discriminator.pth")
    print("DCGAN training finished.")

# --- StyleGAN2 Training (placeholder) ---
def train_stylegan2(dataloader, device, epochs=3):
    print("Starting StyleGAN2 training...")
    os.makedirs("gan_outputs/stylegan2", exist_ok=True)
    G = StyleGAN2_Generator().to(device)
    D = StyleGAN2_Discriminator().to(device)
    for epoch in range(epochs):
        for i, real_images in enumerate(dataloader):
            real_images = real_images.to(device)
            pass

    torch.save(G.state_dict(), "gan_outputs/stylegan2/generator.pth")
    torch.save(D.state_dict(), "gan_outputs/stylegan2/discriminator.pth")
    print("StyleGAN2 training finished.")

# --- Main function ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "S:/Final Year Project/diabetic-retinopathy-ai/data/diabetic_retinopathy"
    batch_size = 16

    dataloader = DataLoader(CustomImageFolder(dataset_path, transform), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    train_models = ["dcgan", "stylegan2"]

    if "dcgan" in train_models:
        train_dcgan(dataloader, device, epochs=3)

    if "stylegan2" in train_models:
        train_stylegan2(dataloader, device, epochs=3)

if __name__ == "__main__":
    main()
