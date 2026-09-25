def _fit(arm: str, *, steps: int, seed: int, teacher: SmileTeacher, noise_policy=None) -> tuple[MidScaleResidual, dict]:
    """Train on :data:`EVAL_SCALES` (always includes ``-1``)."""
    if not _has_scale(EVAL_SCALES, -1.0):
        raise RuntimeError('training grid must include -1')
    train_arm = _train_arm_name(arm)
    torch.manual_seed(int(seed))
    student = MidScaleResidual(int(teacher.concept.numel()))
    if any((param.device != teacher.concept.device for param in student.parameters())):
        raise RuntimeError('student and teacher must share a device')
    targets = {scale: teacher.train_target(train_arm, scale) for scale in EVAL_SCALES}
    cloud = torch.stack([targets[scale] for scale in EVAL_SCALES], dim=0)
    critic = ScaleCritic(student.odd.numel(), cloud, hidden=CRITIC_HIDDEN)
    if noise_policy is not None:
        critic.noise_policy = noise_policy
    gan = GANLoss(loss_type=FORMULATION['loss_type'], mode=FORMULATION['gan_mode'])
    reg = GradientPenalty(arm=FORMULATION['reg_arm'], coeff=FORMULATION['reg_coeff'], kappa=FORMULATION['reg_kappa'], norm=FORMULATION['reg_norm'], lazy_k=FORMULATION['reg_lazy'], target_anneal=FORMULATION['target_anneal'])
    if noise_policy is not None:
        noise_policy.register_generator_base(student)
    g_parameters = list(student.parameters()) + (noise_policy.scale_parameters() if noise_policy is not None else [])
    opt_g = torch.optim.Adam(g_parameters, lr=LR, betas=BETAS)
    opt_d = torch.optim.Adam(critic.parameters(), lr=LR, betas=BETAS)
    if noise_policy is not None:
        noise_policy.register_generator_optimizer(opt_g, opt_d)
    for opt in (opt_g, opt_d):
        opt.param_groups[0]['initial_lr'] = LR
    reals = {scale: _batch(targets[scale]) for scale in EVAL_SCALES}
    cover_w = float(FORMULATION['cover_weight'])
    reg_calls = 0
    n_scales = float(len(EVAL_SCALES))
    for step in range(int(steps)):
        if noise_policy is not None:
            noise_policy.set_step(step)
        for _game_pass in _game_correction.passes():
            _apply_lr(opt_g, step, steps)
            _apply_lr(opt_d, step, steps)
            critic.requires_grad_(True)
            opt_d.zero_grad(set_to_none=True)
            d_loss = student.odd.new_zeros(())
            for scale in EVAL_SCALES:
                fake = student.state(scale).unsqueeze(0).expand(N_ROWS, -1).detach()
                if noise_policy is not None:
                    fake = noise_policy.output(fake, generator_step=False)
                cap, _stats = reg.penalty(lambda z, scale=scale: critic.score(z, scale), reals[scale] / critic.input_scale, fake / critic.input_scale, step=step + 1)
                reg_calls += 1
                d_term = gan.d_loss(critic(reals[scale], scale), critic(fake, scale))
                d_loss = d_loss + (d_term + cap) / n_scales
            d_loss.backward()
            schedule_optimizer(opt_d, step)
            opt_d.step()
            critic.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            g_loss = student.odd.new_zeros(())
            with torch.no_grad():
                real_scores = {scale: critic(reals[scale], scale) for scale in EVAL_SCALES}
            for scale in EVAL_SCALES:
                fake = student.state(scale).unsqueeze(0).expand(N_ROWS, -1)
                if noise_policy is not None:
                    fake = noise_policy.output(fake, generator_step=True)
                g_loss = g_loss + gan.g_loss(critic(fake, scale), real_scores[scale]) / n_scales
            cover = student.odd.new_zeros(())
            for scale in EVAL_SCALES:
                cover = cover + F.mse_loss(student.state(scale), targets[scale])
            g_loss = g_loss + cover_w * cover / n_scales
            g_loss.backward()
            schedule_optimizer(opt_g, step)
            opt_g.step()
        critic.requires_grad_(True)
        checkpoint(step + 1, lambda: score_hold(student, scales=_eval_scales(arm), pairing='stranger' if arm == 'stranger' else 'matched', teacher=teacher))
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == int(steps):
            preview_scales = _eval_scales(arm)
            preview = score_hold(student, scales=preview_scales, pairing='stranger' if arm == 'stranger' else 'matched', teacher=teacher)
            preview.update(arm=arm, steps=step + 1, seed=seed)
            print(format_row(preview), flush=True)
    meta = {'steps': int(steps), 'seed': int(seed), 'train_scales': [float(scale) for scale in EVAL_SCALES], 'loss_type': gan.loss_type if hasattr(gan, 'loss_type') else FORMULATION['loss_type'], 'gan_mode': FORMULATION['gan_mode'], 'reg_arm': reg.arm, 'reg_coeff': float(reg.coeff), 'reg_kappa': float(reg.kappa), 'reg_norm': reg.norm, 'reg_lazy': int(reg.lazy_k), 'reg_anneal': reg.target_anneal, 'reg_calls': int(reg_calls), 'reg_is_gradient_penalty': isinstance(reg, GradientPenalty), 'cover_weight': cover_w, 'fm_weight': float(FORMULATION['fm_weight']), 'device': str(torch.get_default_device())}
    return (student, meta)
