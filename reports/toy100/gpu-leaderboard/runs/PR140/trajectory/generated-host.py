def train(*, pairing: str='shared', gan_factory=None, cap_factory=None, diagnostics=False, noise_policy=None) -> dict:
    """Train and measure identity error; every pairing is allowed."""
    torch.set_num_threads(1)
    torch.manual_seed(PROTOCOL['seed'])
    slow, fast = trajectories()
    index = pairing_index(pairing, slow)
    paired = fast[index]
    hidden = PROTOCOL['critic_hidden']
    generator = _Generator(slow.shape[1], PROTOCOL['z_dim'], fast.shape[1], hidden)
    critic = _Critic(slow.shape[1] + fast.shape[1], hidden)
    if noise_policy is not None:
        from benchmarks.transfer_suite.legacy_noise_adapters import wrap_input, wrap_output
        generator = wrap_output(generator, noise_policy)
        critic = wrap_input(critic, noise_policy, data_index=1)
    view = _FastView(critic)
    prior = ParticlePrior(PROTOCOL['n_particles'], PROTOCOL['z_dim'], init_std=0.1, generator=torch.Generator(device=torch.get_default_device()).manual_seed(PROTOCOL['seed']))
    gan = (gan_factory or make_gan_loss)()
    regularizer = (cap_factory or make_b_cap)()
    spread = ParticleRegularizer(weight=PROTOCOL['vicreg_weight'])
    opt_g = torch.optim.Adam(list(generator.parameters()) + list(prior.parameters()), lr=PROTOCOL['lr'], betas=(PROTOCOL['beta1'], PROTOCOL['beta2']))
    opt_d = torch.optim.Adam(critic.parameters(), lr=PROTOCOL['lr'], betas=(PROTOCOL['beta1'], PROTOCOL['beta2']))
    if noise_policy is not None:
        noise_policy.register_generator_optimizer(opt_g, opt_d)
    steps = PROTOCOL['steps']
    for step in range(1, steps + 1):
        if noise_policy is not None:
            noise_policy.set_step(step - 1)
        for _extra_phase in _extra_state.phases(step, opt_d, opt_g, locals()):
            context = noise_policy.discriminator() if noise_policy is not None else nullcontext()
            with context:
                fake = generator(slow, prior.z)
            opt_d.zero_grad(set_to_none=True)
            d_loss = gan.d_loss(critic(slow, paired), critic(slow, fake.detach()))
            view.slow = slow.detach()
            d_loss = d_loss + regularizer(view, paired, fake.detach(), step=step)
            d_loss.backward()
            if _extra_phase == 0:
                schedule_optimizer(opt_d, step - 1)
            opt_d.step()
            flags = [p.requires_grad for p in critic.parameters()]
            critic.requires_grad_(False)
            try:
                opt_g.zero_grad(set_to_none=True)
                fake = generator(slow, prior.z)
                g_loss = gan.g_loss(critic(slow, fake), critic(slow, paired).detach())
                g_loss = g_loss + PROTOCOL['cover_weight'] * _cover(fake, fast)
                g_loss = g_loss + PROTOCOL['particle_l2'] * prior.z.square().mean()
                g_loss = g_loss + spread(prior.z)
                g_loss.backward()
                if _extra_phase == 0:
                    schedule_optimizer(opt_g, step - 1)
                opt_g.step()
            finally:
                for parameter, flag in zip(critic.parameters(), flags):
                    parameter.requires_grad_(flag)

        def measure_identity():
            context = noise_policy.evaluation(step) if noise_policy is not None else nullcontext()
            with context:
                return {'identity_mse': identity_mse(generator(slow, prior.z).detach(), fast)}
        checkpoint(step, measure_identity)
    context = noise_policy.evaluation(steps) if noise_policy is not None else nullcontext()
    with torch.no_grad(), context:
        pred = generator(slow, prior.z)
        mse = identity_mse(pred, fast)
        paired_mse = identity_mse(pred, paired)
    result = {'identity_mse': mse, 'paired_target_mse': paired_mse, 'verdict': 'PASS' if passed(mse) else 'FAIL'}
    if diagnostics:
        context = noise_policy.evaluation(steps) if noise_policy is not None else nullcontext()
        with torch.no_grad(), context:
            distances = torch.cdist(pred, fast)
            result['set_cover'] = float(_cover(pred, fast))
            result['own_nearest_fraction'] = float((distances.argmin(1) == torch.arange(len(fast))).float().mean())
            result['particle_mean_square'] = float(prior.z.square().mean())
            result['particle_std_mean'] = float(prior.z.std(0).mean())
        norms = []
        context = noise_policy.evaluation(steps) if noise_policy is not None else nullcontext()
        with context:
            for batch in (paired, pred):
                point = batch.detach().requires_grad_(True)
                grad = torch.autograd.grad(view(point).sum(), point)[0]
                norms.append(grad.norm(dim=1).detach())
        norms = torch.cat(norms)
        result['critic_gradient_median'] = float(norms.median())
        result['critic_gradient_max'] = float(norms.max())
    return result
