def train(*, pairing='live', gan_factory=None, cap_factory=None, particle_l2=None, noise_policy=None) -> dict:
    """Run the original 80-step cloud experiment, including stranger arms."""
    torch.manual_seed(TOY_SEED)
    particle_l2 = LOCKED_SHARED.particle_l2 if particle_l2 is None else particle_l2
    base_critic = HostCritic()
    if noise_policy is None:
        critic = base_critic
    else:
        from benchmarks.transfer_suite.legacy_noise_adapters import wrap_input
        critic = wrap_input(base_critic, noise_policy)
    particles = nn.Parameter(torch.zeros(LOCKED_SHARED.n_particles, 1))
    if noise_policy is not None:
        noise_policy.register_generator_base(particles)
    opt_d = torch.optim.Adam(critic.parameters(), lr=TOY_LR, betas=TOY_BETAS)
    if noise_policy is not None and noise_policy.scale_parameters():
        opt_p = torch.optim.Adam([{'params': [particles]}, {'params': noise_policy.scale_parameters(), '_comparison_output_scale': True}], lr=TOY_LR, betas=TOY_BETAS)
    else:
        opt_p = torch.optim.Adam([particles], lr=TOY_LR, betas=TOY_BETAS)
    if noise_policy is not None:
        noise_policy.register_generator_optimizer(opt_p, opt_d)
    gan = (gan_factory or make_gan_loss)()
    regularizer = (cap_factory or make_b_cap)()
    real = real_batch(LOCKED_SHARED.n_particles)
    stranger = torch.linspace(-3.0, 3.0, LOCKED_SHARED.n_particles).unsqueeze(1)
    for step in range(1, TOY_STEPS + 1):
        if noise_policy is not None:
            noise_policy.set_step(step - 1)
        for _game_pass in _game_correction.passes():
            opt_d.zero_grad(set_to_none=True)
            fake = particles.detach() if pairing == 'live' else stranger
            if noise_policy is not None:
                fake = noise_policy.output(fake, generator_step=False)
            d_loss = gan.d_loss(critic(real), critic(fake))
            (d_loss + regularizer(critic, real, fake, step=step)).backward()
            schedule_optimizer(opt_d, step - 1)
            opt_d.step()
            opt_p.zero_grad(set_to_none=True)
            d_real = critic(real).detach()
            generated = particles if pairing == 'live' else stranger
            if noise_policy is not None:
                generated = noise_policy.output(generated, generator_step=True)
            paired = critic(generated)
            g_loss = gan.g_loss(paired, d_real)
            g_loss = g_loss + particle_l2 * particles.square().mean()
            g_loss.backward()
            schedule_optimizer(opt_p, step - 1)
            opt_p.step()
        checkpoint(step, lambda: {'mean_abs': float(particles.detach().abs().mean()), 'grad_med': _grad_median(base_critic, real, particles)})
    with torch.no_grad():
        mean_abs = float(particles.abs().mean())
        nearest = _nearest(particles)
    grad_med = _grad_median(base_critic, real, particles)
    return {'mean_abs': mean_abs, 'grad_med': grad_med, 'nearest': nearest, 'cover_score': LOCKED_SHARED.cover_weight * (1.0 - min(nearest, 1.0)), 'verdict': 'PASS' if cell_wins(mean_abs, grad_med) else 'FAIL'}
