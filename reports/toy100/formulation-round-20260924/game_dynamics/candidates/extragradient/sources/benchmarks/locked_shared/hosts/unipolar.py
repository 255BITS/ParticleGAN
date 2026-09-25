def _fit_rpgan(student: FreeOriginResidual, critic: ScaleCritic, target: torch.Tensor, *, steps: int, recipe: UnipolarRecipe, noise_policy=None) -> tuple[list[dict], GradientPenalty]:
    """One D update then one G update, averaged over scales ``{0, +1}``."""
    gan = GANLoss(loss_type=recipe.loss_type, mode=recipe.gan_mode)
    reg = GradientPenalty(arm=recipe.reg_arm, coeff=recipe.reg_coeff, kappa=recipe.reg_kappa, norm=recipe.reg_norm, lazy_k=recipe.reg_lazy, target_anneal=recipe.target_anneal)
    if noise_policy is not None:
        noise_policy.register_generator_base(student)
    g_parameters = list(student.parameters()) + (noise_policy.scale_parameters() if noise_policy is not None else [])
    opt_g = torch.optim.Adam(g_parameters, lr=LR, betas=BETAS)
    opt_d = torch.optim.Adam(critic.parameters(), lr=LR, betas=BETAS)
    if noise_policy is not None:
        noise_policy.register_generator_optimizer(opt_g, opt_d)
    for opt in (opt_g, opt_d):
        opt.param_groups[0]['initial_lr'] = LR
    real = {0.0: _batch(torch.zeros_like(target)), 1.0: _batch(target)}
    history = []
    for step in range(steps):
        if noise_policy is not None:
            noise_policy.set_step(step)
        for _game_pass in _game_correction.passes():
            _apply_lr(opt_g, step, steps)
            _apply_lr(opt_d, step, steps)
            critic.requires_grad_(True)
            opt_d.zero_grad(set_to_none=True)
            d_loss = student.odd.new_zeros(())
            for scale in SCALES:
                fake = student.delta(scale).unsqueeze(0).expand(N_ROWS, -1).detach()
                if noise_policy is not None:
                    fake = noise_policy.output(fake, generator_step=False)
                cap, _stats = reg.penalty(lambda z, scale=scale: critic.score(z, scale), real[scale] / critic.input_scale, fake / critic.input_scale, step=step + 1)
                d_term = gan.d_loss(critic(real[scale], scale), critic(fake, scale))
                d_loss = d_loss + 0.5 * (d_term + cap)
            d_loss.backward()
            schedule_optimizer(opt_d, step)
            opt_d.step()
            critic.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            g_loss = student.odd.new_zeros(())
            with torch.no_grad():
                real_scores = {scale: critic(real[scale], scale) for scale in SCALES}
            for scale in SCALES:
                fake = student.delta(scale).unsqueeze(0).expand(N_ROWS, -1)
                if noise_policy is not None:
                    fake = noise_policy.output(fake, generator_step=True)
                g_term = gan.g_loss(critic(fake, scale), real_scores[scale])
                g_loss = g_loss + 0.5 * g_term
            g_loss.backward()
            schedule_optimizer(opt_g, step)
            opt_g.step()
        critic.requires_grad_(True)
        checkpoint(step + 1, lambda: score_residual(student))
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            row = score_residual(student)
            row.update(step=step + 1, g_loss=float(g_loss.detach()), d_loss=float(d_loss.detach()))
            history.append(row)
            print(f"unipolar_dir arm={recipe.arm} step={step + 1}/{steps} cover={row['cover']:.4f} leak={row['off_caption']:.4f} neu_hold={row['neu_hold']:.4f} cos_plus={row['cos_plus']:+.4f} hit={('PASS' if row['hit'] else 'FAIL')}", flush=True)
    return (history, reg)
