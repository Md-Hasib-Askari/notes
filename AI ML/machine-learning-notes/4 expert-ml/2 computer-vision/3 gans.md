# GANs (Generative Adversarial Networks)

## Overview
GANs are a class of neural networks that consist of two competing models: a Generator that creates fake data and a Discriminator that tries to distinguish real from fake data. This adversarial training process leads to the generation of highly realistic synthetic data.

## Core Concepts
- **Generator (G)**: Creates fake data from random noise
- **Discriminator (D)**: Classifies data as real or fake
- **Adversarial Training**: Minimax game between G and D
- **Nash Equilibrium**: Optimal state where both networks are balanced

## Mathematical Foundation
### Objective Function
```
min_G max_D V(D,G) = E_x~p_data(x)[log D(x)] + E_z~p_z(z)[log(1 - D(G(z)))]
```
- **Discriminator Goal**: Maximize ability to distinguish real vs fake
- **Generator Goal**: Minimize discriminator's ability to detect fakes

## Basic GAN Architecture

### Generator
```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: nz x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: ngf x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: nc x 64 x 64
        )
    
    def forward(self, input):
        return self.main(input)
```

### Discriminator
```python
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: nc x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)
```

## Training Process
```python
import torch.optim as optim

# Initialize models
netG = Generator()
netD = Discriminator()

# Loss function and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training loop
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        # Update Discriminator
        netD.zero_grad()
        
        # Train with real data
        real_data = data[0]
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), 1.0)  # Real labels
        
        output = netD(real_data)
        errD_real = criterion(output, label)
        errD_real.backward()
        
        # Train with fake data
        noise = torch.randn(batch_size, nz, 1, 1)
        fake_data = netG(noise)
        label.fill_(0.0)  # Fake labels
        
        output = netD(fake_data.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        
        optimizerD.step()
        
        # Update Generator
        netG.zero_grad()
        label.fill_(1.0)  # Generator wants discriminator to believe fakes are real
        
        output = netD(fake_data)
        errG = criterion(output, label)
        errG.backward()
        
        optimizerG.step()
        
        if i % 100 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                  f'Loss_D: {errD_real.item() + errD_fake.item():.4f} '
                  f'Loss_G: {errG.item():.4f}')
```

## GAN Variants

### DCGAN (Deep Convolutional GAN)
- **Architecture**: Uses convolutional layers instead of fully connected
- **Guidelines**: Specific architectural guidelines for stable training
- **Key Features**: BatchNorm, LeakyReLU, Adam optimizer

### WGAN (Wasserstein GAN)
```python
# WGAN loss (Wasserstein distance)
def wasserstein_loss_d(real_output, fake_output):
    return -torch.mean(real_output) + torch.mean(fake_output)

def wasserstein_loss_g(fake_output):
    return -torch.mean(fake_output)

# Weight clipping for Lipschitz constraint
def clip_weights(model, clip_value=0.01):
    for p in model.parameters():
        p.data.clamp_(-clip_value, clip_value)
```

### WGAN-GP (WGAN with Gradient Penalty)
```python
def gradient_penalty(discriminator, real_data, fake_data, device):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated = interpolated.requires_grad_(True)
    
    prob_interpolated = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty
```

### Conditional GAN (cGAN)
```python
class ConditionalGenerator(nn.Module):
    def __init__(self, nz=100, num_classes=10, embed_size=50):
        super(ConditionalGenerator, self).__init__()
        self.embed = nn.Embedding(num_classes, embed_size)
        self.main = nn.Sequential(
            nn.Linear(nz + embed_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        embedded_labels = self.embed(labels)
        input_combined = torch.cat([noise, embedded_labels], 1)
        return self.main(input_combined)
```

### CycleGAN (Image-to-Image Translation)
```python
class CycleGAN:
    def __init__(self):
        self.G_AB = Generator()  # Domain A to B
        self.G_BA = Generator()  # Domain B to A
        self.D_A = Discriminator()  # Discriminator for domain A
        self.D_B = Discriminator()  # Discriminator for domain B
    
    def cycle_consistency_loss(self, real_A, real_B, lambda_cycle=10.0):
        # A -> B -> A
        fake_B = self.G_AB(real_A)
        reconstructed_A = self.G_BA(fake_B)
        cycle_loss_A = nn.L1Loss()(reconstructed_A, real_A)
        
        # B -> A -> B
        fake_A = self.G_BA(real_B)
        reconstructed_B = self.G_AB(fake_A)
        cycle_loss_B = nn.L1Loss()(reconstructed_B, real_B)
        
        return lambda_cycle * (cycle_loss_A + cycle_loss_B)
```

### StyleGAN
- **Progressive Growing**: Gradually increases resolution during training
- **Style Transfer**: Uses adaptive instance normalization
- **Mapping Network**: Maps latent code to intermediate latent space

## Advanced Techniques

### Spectral Normalization
```python
from torch.nn.utils import spectral_norm

# Apply spectral normalization to stabilize training
class SpectralNormDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(3, 64, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 4, 2, 1))
        # ... more layers
```

### Self-Attention
```python
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        query = self.query_conv(x).view(batch_size, -1, height * width)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(attention)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x
```

## Evaluation Metrics

### Inception Score (IS)
```python
import torch.nn.functional as F
from torchvision.models import inception_v3

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    N = len(imgs)
    assert batch_size > 0
    assert N > batch_size
    
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    if cuda:
        inception_model.cuda()
    
    def get_pred(x):
        if resize:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    
    # Get predictions
    preds = np.zeros((N, 1000))
    for i in range(0, N, batch_size):
        batch = imgs[i:i+batch_size]
        if cuda:
            batch = batch.cuda()
        preds[i:i+batch_size] = get_pred(batch)
    
    # Calculate IS
    scores = []
    for i in range(splits):
        part = preds[i * (N // splits): (i + 1) * (N // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    
    return np.mean(scores), np.std(scores)
```

### Fréchet Inception Distance (FID)
```python
def calculate_fid(real_features, fake_features):
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fid
```

## Common Challenges

### Training Instability
- **Mode Collapse**: Generator produces limited variety
- **Vanishing Gradients**: Discriminator becomes too strong
- **Training Imbalance**: One network dominates the other

### Solutions
```python
# Learning rate scheduling
scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99)
scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)

# Different update frequencies
if epoch % 2 == 0:  # Update discriminator less frequently
    optimizerD.step()
optimizerG.step()

# Label smoothing
real_labels = torch.full((batch_size,), 0.9)  # Instead of 1.0
fake_labels = torch.full((batch_size,), 0.1)  # Instead of 0.0
```

## Applications

### Image Generation
- **Face Generation**: Creating realistic human faces
- **Art Creation**: Generating artistic images and styles
- **Data Augmentation**: Creating synthetic training data

### Image-to-Image Translation
- **Style Transfer**: Converting images between different artistic styles
- **Domain Adaptation**: Converting between different image domains
- **Super Resolution**: Enhancing image resolution

### Video Generation
- **Video Synthesis**: Creating realistic video sequences
- **Frame Interpolation**: Generating intermediate frames
- **Video Enhancement**: Improving video quality

### Other Applications
- **Text-to-Image**: Generating images from text descriptions
- **Music Generation**: Creating musical compositions
- **Drug Discovery**: Generating molecular structures

## Best Practices
1. **Start with DCGAN guidelines** for stable training
2. **Use appropriate learning rates** (typically 0.0002)
3. **Monitor training carefully** and adjust hyperparameters
4. **Use feature matching** to stabilize training
5. **Implement gradient penalties** for better convergence
6. **Use spectral normalization** in discriminator
7. **Apply progressive growing** for high-resolution images

## Tools and Frameworks
- **PyTorch**: Most popular framework for GAN research
- **TensorFlow/Keras**: Alternative implementation option
- **StyleGAN2-PyTorch**: High-quality face generation
- **BigGAN-PyTorch**: Large-scale image generation
- **CycleGAN**: Image-to-image translation

## Future Directions
- **Diffusion Models**: Alternative generative approach
- **Neural Radiance Fields (NeRF)**: 3D-aware generation
- **Few-shot Generation**: Learning from limited data
- **Controllable Generation**: Fine-grained control over outputs
- **Efficiency Improvements**: Faster training and inference

## Resources
- **Original Paper**: "Generative Adversarial Nets" by Goodfellow et al.
- **The GAN Zoo**: Comprehensive list of GAN variants
- **Papers**: DCGAN, WGAN, StyleGAN, BigGAN, CycleGAN
- **Implementations**: PyTorch-GAN, TensorFlow-GAN
- **Datasets**: CelebA, CIFAR-10, ImageNet, LSUN
