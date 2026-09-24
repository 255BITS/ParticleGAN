def train_mode_hold(recipe: ModeHoldRecipe | None=None, *, seed: int=0, gan_factory=None, cap_factory=None, diagnostics=False, training_recipe=None, log=None, noise_policy=None) -> dict:
    """Train every requested arm and score only its generated samples."""
    recipe = ModeHoldRecipe() if recipe is None else recipe
    if training_recipe is not None:
        recipe = replace(recipe, n_particles=training_recipe.num_particles, particle_l2=0.0, vicreg_weight=training_recipe.prior_reg, beta1=training_recipe.betas[0], beta2=training_recipe.betas[1], ema=training_recipe.ema_decay, d_lr_mult=training_recipe.d_lr_mult, steps=training_recipe.total_steps)
    torch.manual_seed(seed)
    stream = torch.Generator().manual_seed(seed)
    means = ring_means()
    prior = ParticlePrior(recipe.n_particles, Z_DIM, init_std=0.5, generator=stream) if training_recipe is None else training_recipe.make_prior(generator=stream)
    generator = SimpleMLPGenerator(Z_DIM, HIDDEN, N_HIDDEN, 2)
    critic = SimpleMLPDiscriminator(2, HIDDEN, N_HIDDEN, FOURIER)
    if noise_policy is not None:
        from benchmarks.transfer_suite.legacy_noise_adapters import wrap_input, wrap_output
        generator = wrap_output(generator, noise_policy)
        critic = wrap_input(critic, noise_policy)
    gan = (gan_factory or (training_recipe.make_loss if training_recipe else make_gan_loss))()
    regularizer = (cap_factory or (training_recipe.make_gradient_penalty if training_recipe else make_b_cap))()
    vicreg = ParticleRegularizer(weight=recipe.vicreg_weight)
    opt_g = torch.optim.Adam(list(generator.parameters()) + list(prior.parameters()), lr=LR, betas=(recipe.beta1, recipe.beta2))
    opt_d = torch.optim.Adam(critic.parameters(), lr=LR * recipe.d_lr_mult, betas=(recipe.beta1, recipe.beta2))
    if training_recipe is not None:
        opt_g, opt_d = training_recipe.make_optimizers(generator, critic, prior)
    if noise_policy is not None:
        noise_policy.register_generator_optimizer(opt_g, opt_d)
    base_lrs = [[group['lr'] for group in opt.param_groups] for opt in (opt_g, opt_d)]
    batch = BATCH if training_recipe is None else training_recipe.batch_size
    ema_g = [p.detach().clone() for p in generator.parameters()]
    ema_z = prior.z.detach().clone()

    @torch.no_grad()
    def measure(step: int):
        context = noise_policy.evaluation(step) if noise_policy is not None else nullcontext()
        with context:
            latent, _ = prior.sample(EVAL_N, generator=torch.Generator().manual_seed(seed + 9))
            row = diversity(generator(latent), means, detailed=diagnostics)
            if diagnostics:
                row['support'] = diversity(generator(prior.z), means, detailed=True)
                if noise_policy is not None and noise_policy.output_std > 0:
                    row['support_scope'] = 'one noisy draw per particle'
        return row

    def snapshot(step: int) -> dict:
        saved_g = [p.detach().clone() for p in generator.parameters()]
        saved_z = prior.z.detach().clone()
        with torch.no_grad():
            for param, ema in zip(generator.parameters(), ema_g):
                param.copy_(ema)
            prior.z.copy_(ema_z)
            if noise_policy is not None and step == recipe.steps:
                noise_policy.capture_final_ema()
            row = measure(step)
        with torch.no_grad():
            for param, saved in zip(generator.parameters(), saved_g):
                param.copy_(saved)
            prior.z.copy_(saved_z)
        row.update(step=step, seed=seed)
        return row
    curve = []
    live_curve = []
    for step in range(1323, 1335):
        if noise_policy is not None:
            noise_policy.set_step(step)
        if training_recipe is not None:
            scale = learning_rate_scale(step, recipe.steps, training_recipe.lr_anneal_start, training_recipe.lr_floor)
            for opt, rates in zip((opt_g, opt_d), base_lrs):
                for group, rate in zip(opt.param_groups, rates):
                    group['lr'] = rate * scale
        for _extra_phase in _extra_state.phases(step, opt_d, opt_g, locals()):
            real = sample_ring(means, batch, SIGMA, stream)
            latent, _ = prior.sample(batch, generator=stream)
            context = noise_policy.discriminator() if noise_policy is not None else nullcontext()
            with context:
                fake = generator(latent).detach()
            d_loss = gan.d_loss(critic(real), critic(fake))
            d_loss = d_loss + regularizer(critic, real, fake, step=step + 1)
            opt_d.zero_grad()
            d_loss.backward()
            if _extra_phase == 0:
                schedule_optimizer(opt_d, step)
            opt_d.step()
            latent, _ = prior.sample(batch, generator=stream)
            fake = generator(latent)
            if gan.mode in ('rp', 'ra'):
                real_g = sample_ring(means, batch, SIGMA, stream)
                g_loss = gan.g_loss(critic(fake), critic(real_g))
            else:
                g_loss = gan.g_loss(critic(fake))
            if recipe.fm_weight > 0.0:
                real_mean = sample_ring(means, batch, SIGMA, stream).detach().mean(0)
                g_loss = g_loss + recipe.fm_weight * (fake.mean(0) - real_mean).pow(2).sum()
            g_loss = g_loss + recipe.particle_l2 * prior.z.pow(2).mean()
            g_loss = g_loss + vicreg(prior.z)
            opt_g.zero_grad()
            g_loss.backward()
            if _extra_phase == 0:
                schedule_optimizer(opt_g, step)
            opt_g.step()
        with torch.no_grad():
            for ema, param in zip(ema_g, generator.parameters()):
                ema.mul_(recipe.ema).add_(param, alpha=1.0 - recipe.ema)
            ema_z.mul_(recipe.ema).add_(prior.z, alpha=1.0 - recipe.ema)
        checkpoint(step + 1, lambda: measure(step + 1))
        if diagnostics and (step + 1) % 200 == 0:
            curve.append(snapshot(step + 1))
        if diagnostics and ((step + 1) % 200 == 0 or (step + 1 >= recipe.steps - 200 and (step + 1) % 50 == 0)):
            point = {'step': step + 1, **measure(step + 1)}
            live_curve.append(point)
            if log is not None:
                log(point)
    if noise_policy is not None:
        noise_policy.capture_final_live()
    final = snapshot(recipe.steps)
    final['verdict'] = verdict(final)
    if diagnostics:
        live = measure(recipe.steps)
        final['live'] = {**live, 'verdict': verdict(live)}
        final['curve'] = curve
        final['live_curve'] = live_curve
    return final
