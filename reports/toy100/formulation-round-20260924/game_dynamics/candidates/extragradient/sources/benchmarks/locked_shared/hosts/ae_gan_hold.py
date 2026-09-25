def train(cfg: HoldConfig, *, noise_policy=None) -> dict:
    """Train and return reconstruction/hold measurements."""
    torch.manual_seed(cfg.seed)
    recipe = make_recipe(cfg)
    prior = recipe.make_prior()
    encoder, decoder, critic = (MLP(2, 4), MLP(2, 2), MLP(2, 1))
    if noise_policy is not None:
        from benchmarks.transfer_suite.legacy_noise_adapters import wrap_input, wrap_output
        decoder = wrap_output(decoder, noise_policy)
        critic = wrap_input(critic, noise_policy)
    opt_g, opt_d = recipe.make_optimizers(decoder, critic, prior, encoder=encoder)
    if noise_policy is not None:
        noise_policy.register_generator_optimizer(opt_g, opt_d)
    gan = recipe.make_loss()
    regularizer = recipe.make_gradient_penalty(norm=cfg.reg_norm, target_anneal=cfg.target_anneal)
    stream = torch.Generator(device=torch.get_default_device()).manual_seed(0)
    torch.rand(3, generator=stream)
    torch.randn(8, 2, generator=stream)
    torch.randn(8, 2, generator=stream)
    if torch.get_default_device().type == 'cuda':
        torch.cuda.set_rng_state(stream.get_state())
    else:
        torch.set_rng_state(stream.get_state())

    def measure(step: int):
        context = noise_policy.evaluation(step) if noise_policy is not None else nullcontext()
        with context:
            return evaluate(encoder, decoder, prior, recipe)
    opened = measure(0)
    _log(cfg.name, 0, opened, extra=' phase=init')
    penalty_applied = 0
    adv_steps = 0
    for step in range(1, cfg.steps + 1):
        if noise_policy is not None:
            noise_policy.set_step(step - 1)
        for _game_pass in _game_correction.passes():
            data = sample_data(cfg.batch)
            if cfg.adversarial_weight > 0:
                codes, _ = prior.sample(cfg.batch)
                context = noise_policy.discriminator() if noise_policy is not None else nullcontext()
                with context:
                    fake = decoder(codes).detach()
                opt_d.zero_grad(set_to_none=True)
                d_loss = gan.d_loss(critic(data).squeeze(-1), critic(fake).squeeze(-1))
                penalty, stats = regularizer.penalty(critic, data, fake, step=step)
                if stats.get('applied'):
                    penalty_applied += 1
                (d_loss + penalty).backward()
                schedule_optimizer(opt_d, step - 1)
                opt_d.step()
            query, offset = encoder(data).chunk(2, dim=1)
            encoded = recipe.encode(query, prior, offset=offset)
            reconstructed = decoder(encoded.codes[:, 0])
            recon = encoded.reconstruction_loss(reconstructed[:, None], data)
            codes, _ = prior.sample(cfg.batch)
            generated = decoder(codes)
            opt_g.zero_grad(set_to_none=True)
            for param in critic.parameters():
                param.requires_grad_(False)
            loss = cfg.reconstruction_weight * recon + cfg.particle_l2 * prior.z.square().mean()
            if cfg.adversarial_weight > 0:
                real_logits = critic(data).squeeze(-1).detach()
                fake_logits = critic(generated).squeeze(-1)
                adv = gan.g_loss(fake_logits, real_logits)
                anchors = _anchors()
                cover = torch.cdist(anchors, generated).min(dim=1).values.mean()
                loss = loss + cfg.adversarial_weight * adv + cfg.cover_weight * cover
                if cfg.fm_weight > 0:
                    real_feat = critic.features(data).detach().mean(0)
                    fake_feat = critic.features(generated).mean(0)
                    loss = loss + cfg.fm_weight * (real_feat - fake_feat).square().mean()
                adv_steps += 1
            loss.backward()
            for param in critic.parameters():
                param.requires_grad_(True)
            schedule_optimizer(opt_g, step - 1)
            opt_g.step()
        checkpoint(step, lambda: measure(step))
        if step == 1 or (step % 50 == 0 and step != cfg.steps):
            snap = measure(step)
            _log(cfg.name, step, snap, extra=f' loss={float(loss.detach()):.4f}')
    final = measure(cfg.steps)
    row = {'name': cfg.name, 'cfg': cfg, 'recon_mse': final['recon_mse'], 'hold': final['hold'], 'init_recon_mse': opened['recon_mse'], 'penalty_applied': penalty_applied, 'adv_steps': adv_steps, 'steps': cfg.steps}
    return row
