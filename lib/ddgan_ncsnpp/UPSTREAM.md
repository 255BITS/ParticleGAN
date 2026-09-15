# Official DDGAN generator

Source: https://github.com/NVlabs/denoising-diffusion-gan
Commit: `6818ded6443e10ab69c7864745457ce391d4d883`

Copied model dependencies with their original copyright headers and licenses.
Changes: local resampling import; upstream native PyTorch FIR implementation used
on CPU and CUDA instead of compiling the custom CUDA extension; optional class
embedding added to the timestep embedding after its MLP. Architecture and
initializers otherwise retained. Wrapper in `lib/image_ncsnpp.py`.
No upstream trainer, losses, diffusion schedule, optimizer, or pretrained G weights imported.
