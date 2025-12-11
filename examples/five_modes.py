import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import deque
from lib.particle_prior import ParticlePrior
from lib.vicreg_loss import VICRegLikeLoss

# ==========================================
# 1. Setup & Data
# ==========================================
WORDS = ['apple', 'grape', 'lemon', 'melon', 'berry'] 
# We set num_particles = len(WORDS) to encourage 1-to-1 mapping for this toy problem
NUM_PARTICLES = len(WORDS) 

CHARS = "abcdefghijklmnopqrstuvwxyz_ "
CHAR_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_CHAR = {i: c for i, c in enumerate(CHARS)}
SEQ_LEN = 6

WORD_COLORS = ['#FF5555', '#50FA7B', '#F1FA8C', '#BD93F9', '#8BE9FD']

def str_to_tensor(text_list):
    batch_indices = []
    for text in text_list:
        text = text.ljust(SEQ_LEN, '_')[:SEQ_LEN]
        indices = [CHAR_IDX.get(c, 26) for c in text]
        batch_indices.append(indices)
    tensor = torch.tensor(batch_indices, dtype=torch.long)
    return F.one_hot(tensor, num_classes=len(CHARS)).permute(0, 2, 1).float()

def tensor_to_str(tensor_logits):
    indices = torch.argmax(tensor_logits, dim=1).cpu().numpy()
    results = []
    for row in indices:
        s = "".join([IDX_CHAR.get(i, '?') for i in row]).replace('_', '').strip()
        results.append(s)
    return results

# ==========================================
# 2. Components
# ==========================================

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len(CHARS) * SEQ_LEN, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2) 
        )
    def forward(self, x): 
        return self.net(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, len(CHARS) * SEQ_LEN)
        )
    def forward(self, z):
        flat = self.net(z)
        return flat.view(-1, len(CHARS), SEQ_LEN)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear((len(CHARS) * SEQ_LEN) + 2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1) 
        )
    def forward(self, x, z):
        x_flat = x.flatten(1)
        joint = torch.cat([x_flat, z], dim=1)
        return self.net(joint)

def r1_penalty(logits, x, gamma=10.0):
    grad = torch.autograd.grad(
        outputs=logits.sum(), inputs=x, 
        create_graph=True, only_inputs=True
    )[0]
    return grad.pow(2).view(x.shape[0], -1).sum(1).mean() * (gamma / 2)

# ==========================================
# 3. Training & Viz
# ==========================================

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Models
    E = Encoder().to(device)
    G = Generator().to(device)
    D = Discriminator().to(device)
    
    # NEW: The Particle Prior
    prior = ParticlePrior(num_particles=NUM_PARTICLES, z_dim=2).to(device)
    vic_loss_fn = VICRegLikeLoss()

    # Optimizers
    # We add prior parameters to GE optimizer so they can move to optimal locations
    opt_GE = torch.optim.Adam(list(E.parameters()) + list(G.parameters()) + list(prior.parameters()), 
                              lr=1e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    criterion = nn.BCEWithLogitsLoss()

    loss_D_hist = deque(maxlen=200)
    loss_GE_hist = deque(maxlen=200)

    # --- SETUP DASHBOARD ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_latent = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :]) 
    ax_table.axis('off')

    plt.ion()

    for epoch in range(20001):
        batch_size = 64
        
        # --- TRAIN D ---
        opt_D.zero_grad()
        real_words = [random.choice(WORDS) for _ in range(batch_size)]
        x_real = str_to_tensor(real_words).to(device)
        x_real.requires_grad_(True) 
        
        z_enc = E(x_real).detach()
        pred_real = D(x_real, z_enc)
        loss_d_real = criterion(pred_real, torch.ones_like(pred_real))

        # NEW: Sample from Particle Prior instead of randn
        z_prior, _ = prior.sample(batch_size)
        x_gen = G(z_prior).detach()
        x_gen_soft = F.softmax(x_gen, dim=1)
        pred_fake = D(x_gen_soft, z_prior) # We pass the Particle, not random noise
        loss_d_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

        r1 = r1_penalty(pred_real, x_real, gamma=1.0)
        loss_d = loss_d_real + loss_d_fake + r1
        loss_d.backward()
        opt_D.step()

        # --- TRAIN GE (and Prior) ---
        opt_GE.zero_grad()
        x_real.requires_grad_(False)
        
        # Encoder Loss
        z_enc = E(x_real)
        pred_enc = D(x_real, z_enc)
        loss_ge_enc = criterion(pred_enc, torch.zeros_like(pred_enc)) 
        
        # Generator Loss (Sample prior again)
        z_prior, _ = prior.sample(batch_size)
        x_gen = G(z_prior)
        x_gen_soft = F.softmax(x_gen, dim=1)
        pred_gen = D(x_gen_soft, z_prior)
        loss_ge_gen = criterion(pred_gen, torch.ones_like(pred_gen)) 
        
        # NEW: Regularize the Prior
        # This forces the "stars" to stay apart
        loss_vic = vic_loss_fn(prior.z) * 2.0 # Weight for structure
        
        loss_ge = loss_ge_enc + loss_ge_gen + loss_vic
        loss_ge.backward()
        opt_GE.step()

        loss_D_hist.append(loss_d.item())
        loss_GE_hist.append(loss_ge.item())

        # --- VISUALIZATION (Every 50 epochs) ---
        if epoch % 50 == 0:
            ax_loss.clear()
            ax_loss.plot(loss_D_hist, label="D", color='#FF5555', alpha=0.8)
            ax_loss.plot(loss_GE_hist, label="G/E/Prior", color='#8BE9FD', alpha=0.8)
            ax_loss.legend(loc='upper right')
            ax_loss.set_title("Losses")

            # 2. Latent Space Map
            ax_latent.clear()
            with torch.no_grad():
                vocab_x = str_to_tensor(WORDS).to(device)
                vocab_z = E(vocab_x).cpu().numpy()
                typo_z = E(str_to_tensor(["aple"]).to(device)).cpu().numpy()
                # Get Prior Positions
                prior_z = prior.z.detach().cpu().numpy()

            # Plot Prior Particles (The "Stars")
            ax_latent.scatter(prior_z[:,0], prior_z[:,1], color='white', marker='*', s=300, label="Prior Particles", edgecolors='black', zorder=1)

            # Plot Encoded Words
            for i, txt in enumerate(WORDS):
                ax_latent.scatter(vocab_z[i,0], vocab_z[i,1], color=WORD_COLORS[i], s=120, edgecolors='white', alpha=0.9, zorder=2)
                ax_latent.text(vocab_z[i,0]+0.05, vocab_z[i,1]+0.05, txt, color=WORD_COLORS[i], fontsize=10, fontweight='bold')
            
            # Plot Typo
            ax_latent.scatter(typo_z[0,0], typo_z[0,1], color='cyan', marker='X', s=150, label="Typo", zorder=3)
            
            ax_latent.set_title(f"Particle Space (Stars=Prior)")
            ax_latent.grid(True, alpha=0.2)

            # 3. Live Reconstruction Table
            ax_table.clear(); ax_table.axis('off')
            with torch.no_grad():
                recon_vocab = tensor_to_str(G(E(vocab_x)))
                recon_typo = tensor_to_str(G(E(str_to_tensor(["aple"]).to(device))))[0]

            table_txt = f"{'ORIGINAL':<12} | {'RECONSTRUCTED':<15} | {'STATUS'}\n"
            table_txt += "-" * 45 + "\n"
            valid_cnt = 0
            for i, word in enumerate(WORDS):
                rec = recon_vocab[i]
                status = "✅" if word == rec else "❌"
                if word == rec: valid_cnt += 1
                table_txt += f"{word:<12} | {rec:<15} | {status}\n"
            
            table_txt += "-" * 45 + "\n"
            status = "✨ MAGICAL ✨" if recon_typo == "apple" else "..."
            table_txt += f"{'aple':<12} | {recon_typo:<15} | {status}\n"

            header = f"Epoch: {epoch} | Accuracy: {valid_cnt}/{len(WORDS)}"
            ax_table.text(0.5, 0.9, header, ha='center', fontsize=14, color='white', fontweight='bold')
            ax_table.text(0.5, 0.5, table_txt, ha='center', va='center', fontsize=12, fontfamily='monospace', color='#F8F8F2')

            plt.pause(0.01)
            if epoch % 200 == 0:
                os.makedirs("results", exist_ok=True)
                fig.savefig(f"results/frame_{epoch:05d}.png", dpi=100, facecolor='#282a36')

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    train()
