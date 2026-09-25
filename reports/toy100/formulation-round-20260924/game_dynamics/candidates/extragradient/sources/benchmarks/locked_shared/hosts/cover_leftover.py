def fit_cover_leftover(recipe: CoverRecipe, *, log=None, field: LeftoverField | None=None, noise_policy=None) -> dict:
    """Train one arm. Returns the EMA residual score plus recipe pins."""
    field = field or LeftoverField()
    torch.manual_seed(recipe.seed)
    dim = field.dim
    residual = _Residual(dim)
    prior_p = ParticlePrior(recipe.knob('n_particles'), dim, init_std=recipe.knob('particle_init_std'))
    prior_m = ParticlePrior(recipe.knob('n_particles'), dim, init_std=recipe.knob('particle_init_std'))
    critic = _FourierCritic(dim, n_rand=recipe.knob('critic_n_rand'), hidden=recipe.knob('critic_hidden'), seed=recipe.seed)
    if noise_policy is not None:
        from benchmarks.transfer_suite.legacy_noise_adapters import wrap_input
        critic = wrap_input(critic, noise_policy)
    gan = GANLoss(loss_type=recipe.knob('loss_type'), mode=recipe.knob('gan_mode'))
    penalty = GradientPenalty(arm=recipe.knob('reg_arm'), coeff=recipe.knob('reg_coeff'), kappa=recipe.knob('reg_kappa'), norm=recipe.knob('reg_norm'), lazy_k=1, target_anneal='none')
    spread = ParticleRegularizer(target_std=recipe.knob('vicreg_std'), weight=recipe.knob('vicreg_weight'))
    lr = float(recipe.knob('lr'))
    betas = (float(recipe.knob('beta1')), float(recipe.knob('beta2')))
    if noise_policy is not None:
        noise_policy.register_generator_base(residual)
    generator_params = list(residual.parameters()) + (noise_policy.scale_parameters() if noise_policy is not None else [])
    opt_g = torch.optim.Adam([{'params': generator_params, 'lr': lr}, {'params': list(prior_p.parameters()) + list(prior_m.parameters()), 'lr': lr}], lr=lr, betas=betas)
    opt_d = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)
    if noise_policy is not None:
        noise_policy.register_generator_optimizer(opt_g, opt_d)
    ema = _EMA(generator_params, decay=float(recipe.knob('ema')))
    poles_p, poles_m, neu = teacher_poles(field, recipe.knob('teacher'))
    half = max(1, int(recipe.knob('batch')) // 2)
    jitter = float(recipe.knob('particle_jitter'))
    cover_w = float(recipe.knob('cover_weight'))
    particle_l2 = float(recipe.knob('particle_l2'))
    _log(log, 'cover_leftover start arm=%s steps=%s seed=%s teacher=%s cover=%.1f penalty_coeff=%.1f kappa=%.1f fm=%.1f n_particles=%s' % (recipe.arm, recipe.steps, recipe.seed, recipe.knob('teacher'), cover_w, penalty.coeff, penalty.kappa, float(recipe.knob('fm_weight')), recipe.knob('n_particles')))

    def fake_batch() -> tuple[torch.Tensor, torch.Tensor]:
        fake_p = neu + residual.delta(1.0) + _particle_batch(prior_p, half, jitter)
        fake_m = neu + residual.delta(-1.0) + _particle_batch(prior_m, half, jitter)
        return (fake_p, fake_m)
    for step in range(recipe.steps):
        if noise_policy is not None:
            noise_policy.set_step(step)
        for _game_pass in _game_correction.passes():
            scale = _delayed_cosine(step, recipe.steps, int(recipe.knob('delay')), float(recipe.knob('min_lr_ratio')))
            for group in opt_g.param_groups:
                group['lr'] = lr * scale
            for group in opt_d.param_groups:
                group['lr'] = lr * scale
            real_p = _sample_real_cloud(poles_p, neu, half, cloud_std=recipe.knob('cloud_std'), span_frac=recipe.knob('span_frac'), end_margin=recipe.knob('end_margin'))
            real_m = _sample_real_cloud(poles_m, neu, half, cloud_std=recipe.knob('cloud_std'), span_frac=recipe.knob('span_frac'), end_margin=recipe.knob('end_margin'))
            real = torch.cat([real_p, real_m], dim=0)
            fake_p, fake_m = fake_batch()
            fake = torch.cat([fake_p, fake_m], dim=0).detach()
            if noise_policy is not None:
                fake = noise_policy.output(fake, generator_step=False)
            d_loss = gan.d_loss(critic(real.detach()), critic(fake))
            cap = penalty(critic, real.detach(), fake, step=step + 1)
            d_loss = d_loss + cap
            opt_d.zero_grad()
            d_loss.backward()
            schedule_optimizer(opt_d, step)
            opt_d.step()
            fake_p, fake_m = fake_batch()
            fake = torch.cat([fake_p, fake_m], dim=0)
            if noise_policy is not None:
                fake = noise_policy.output(fake, generator_step=True)
            g_loss = gan.g_loss(critic(fake), critic(real.detach()))
            parts = torch.cat([prior_p.z, prior_m.z], dim=0)
            g_loss = g_loss + spread(parts)
            if particle_l2 > 0.0:
                g_loss = g_loss + particle_l2 * parts.pow(2).mean()
            if cover_w > 0.0:
                cover = (neu + residual.delta(1.0) - poles_p).pow(2).mean()
                cover = cover + (neu + residual.delta(-1.0) - poles_m).pow(2).mean()
                g_loss = g_loss + cover_w * cover
            opt_g.zero_grad()
            g_loss.backward()
            schedule_optimizer(opt_g, step)
            opt_g.step()
        ema.update(generator_params)
        checkpoint(step + 1, lambda: score_geometry(residual, field, poles_p, poles_m, neu))
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == recipe.steps:
            live = score_geometry(residual, field, poles_p, poles_m, neu)
            _log(log, 'cover_leftover arm=%s step=%s/%s d=%.4f g=%.4f cap=%.4f u_kept=%.3f content=%.3f leak=%.3f err=%.3f covered=%s' % (recipe.arm, step + 1, recipe.steps, float(d_loss.detach()), float(g_loss.detach()), float(cap.detach()), live['u_kept'], live['content_kept'], live['leak_ratio'], max(live['pole_rel_err_plus'], live['pole_rel_err_minus']), int(live['covered'])))
    live_score = score_geometry(residual, field, poles_p, poles_m, neu)
    if noise_policy is not None:
        noise_policy.capture_final_live()
    ema.copy_to(generator_params)
    if noise_policy is not None:
        noise_policy.capture_final_ema()
    scored = score_geometry(residual, field, poles_p, poles_m, neu)
    with torch.no_grad():
        particle_rms = float(torch.cat([prior_p.z, prior_m.z], dim=0).pow(2).mean().sqrt())
    row = {'arm': recipe.arm, 'steps': recipe.steps, 'seed': recipe.seed, 'teacher': recipe.knob('teacher'), 'cover_weight': cover_w, 'reg_coeff': float(penalty.coeff), 'reg_arm': penalty.arm, 'kappa': float(penalty.kappa), 'norm': penalty.norm, 'fm_weight': float(recipe.knob('fm_weight')), 'gan_mode': recipe.knob('gan_mode'), 'loss_type': recipe.knob('loss_type'), 'n_particles': int(recipe.knob('n_particles')), 'particle_l2': particle_l2, 'particle_rms': particle_rms, **scored, 'live': live_score}
    _log(log, 'cover_leftover DONE arm=%s pass=%s reasons=%s u_kept=%.3f content=%.3f leak=%.3f err=%.3f same_dir=%.3f particle_rms=%.3f' % (recipe.arm, int(row['pass']), row['fail_reasons'] or 'none', row['u_kept'], row['content_kept'], row['leak_ratio'], max(row['pole_rel_err_plus'], row['pole_rel_err_minus']), row['same_dir'], particle_rms))
    return row
