# Particle Interpolation Stretch Toys (No MoG)

<a id="particle-interpolation-stretch"></a>

## Scientific Question

When a generator $G$ and discrete zero-sigma `ParticlePrior` (`kind: "particle_cloud"`, `sigma: 0.0`, `standardize: false`) learn a small finite set of latent particles $z_1, \dots, z_M$ **without MoG kernel smoothing**, does unseen latent interior space between particles produce valid, smoothly progressing manifold outputs rather than chord-cutting dead space, double-exposed pixel ghosts, or Voronoi lookup snapping?

All three tasks execute through `GANTrainer` under protocol `seed = 0`, named deterministic orthogonal initialization (`particlegan.init.deterministic_orthogonal_`), `recipe_owned_v1` prior ownership, clean live evaluation (`enumerated_prior_without_output_noise`), 24 scheduled checks, and a 5-check sustained terminal pass requirement.

---

## `interp_sparse_arc`

<a id="interp_sparse_arc"></a>

- **Tier / Role:** Tier 1 `stretch` in `discriminator_stability` (`600` updates, `batch = 128`, `timeout = 300s`).
- **Target & Capacity:** 2D $270^\circ$ circular arc ($R = 2.0$, $\theta \in [-\pi/4, 5\pi/4]$, radial $\sigma = 0.03$) with `M = 12` discrete particles in `z_dim = 2`, 2-layer 64-wide MLP generator and Fourier-2 critic.
- **Unseen Latent Query:** Undirected latent $k$-NN ($k = 3$) chords $z_{ij}(t) = (1 - t) z_i + t z_j$ at $t \in \{0.25, 0.50, 0.75\}$.
- **Frozen Gates:** `finite_fraction == 1.0`, `endpoint_hq >= 0.85`, `interp_hq >= 0.80` (rejects straight Euclidean chord-cutting across the arc interior), `interp_arc_bins >= 22` of `32` (impossible for 12 endpoints alone), and `midpoint_novelty >= 0.20` (rejects Voronoi/step-function snapping).

---

## `interp_img_patch_shift`

<a id="interp_img_patch_shift"></a>

- **Tier / Role:** Tier 2 `stretch` in `discriminator_stability` (`800` updates, `batch = 64`, `timeout = 300s`).
- **Target & Capacity:** Continuous $1 \times 8 \times 8$ grayscale images with a single unit-peak $2 \times 2$ window patch at continuous center $(u, v) \in [1.5, 6.5]^2$ (`0.01` clipped Gaussian noise), `M = 16` discrete particles in `z_dim = 4`, width-16 `residual_upsample` conv generator and conv critic.
- **Unseen Latent Query:** Undirected latent $k$-NN ($k = 2$) chords at $t \in \{0.25, 0.50, 0.75\}$ projected onto the $51 \times 51$ continuous single-patch manifold.
- **Frozen Gates:** `finite_fraction == 1.0`, `endpoint_rmse <= 0.09`, `interp_rmse <= 0.10`, `interp_peak >= 0.78` (rejects 50/50 pixel-space double-exposure ghosts whose peak drops to `0.50`), `quadrant_coverage == 4`, and `midpoint_novelty >= 0.18`.

---

## `interp_saddle_barycentric`

<a id="interp_saddle_barycentric"></a>

- **Tier / Role:** Tier 2 `stretch` in `discriminator_stability` (`800` updates, `batch = 128`, `timeout = 300s`).
- **Target & Capacity:** Non-harmonic curved 3D saddle sheet $w = 0.8(x^2 - y^2) + 0.9\cos(0.75\pi x)\cos(0.75\pi y) + \mathcal{N}(0, 0.03^2)$ over $(x, y) \in [-1, 1]^2$, `M = 24` discrete particles in `z_dim = 2`, 3D-output 2-layer 64-wide MLP generator and Fourier-2 critic.
- **Unseen Latent Query:** 2D simplex barycenters $\frac{1}{3}(z_i + z_j + z_k)$ of non-collinear latent $k$-NN ($k = 4$) triangles.
- **Frozen Gates:** `finite_fraction == 1.0`, `endpoint_surface_mae <= 0.08`, `barycenter_surface_mae <= 0.09`, `barycenter_hq >= 0.78` (rejects flat planes and 3D Euclidean simplex chord-cutting), `xy_bins_covered >= 12` of `16`, and `barycenter_novelty >= 0.18`.
