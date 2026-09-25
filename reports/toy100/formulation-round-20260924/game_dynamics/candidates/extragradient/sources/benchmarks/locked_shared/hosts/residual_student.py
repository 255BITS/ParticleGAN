def train(*, pairing: str='shared', echo: bool=False, log: Callable[[dict], None] | None=None, noise_policy=None) -> dict:
    """Train the residual head; score the resulting predictions."""
    torch.set_num_threads(1)
    torch.manual_seed(PROTOCOL['seed'])
    slow, fast = trajectories()
    index = pairing_index(pairing, slow)
    paired = fast[index]
    mask = both_land_mask(slow, fast, index)
    hidden = PROTOCOL['critic_hidden']
    head = ResidualHead(slow.shape[1], PROTOCOL['z_dim'], hidden)
    critic = _Critic(slow.shape[1] + fast.shape[1], hidden)
    if noise_policy is not None:
        from benchmarks.transfer_suite.legacy_noise_adapters import wrap_input, wrap_output
        head = wrap_output(head, noise_policy)
        critic = wrap_input(critic, noise_policy, data_index=1)
    view = _FastView(critic)
    prior = ParticlePrior(PROTOCOL['n_particles'], PROTOCOL['z_dim'], init_std=0.1, generator=torch.Generator(device=torch.get_default_device()).manual_seed(PROTOCOL['seed']))
    gan = GANLoss(PROTOCOL['loss_type'], PROTOCOL['gan_mode'])
    regularizer = GradRegularizer(PROTOCOL['reg_arm'], PROTOCOL['reg_coeff'], kappa=PROTOCOL['reg_kappa'], norm=PROTOCOL['reg_norm'], lazy_k=PROTOCOL['reg_lazy'], target_anneal=PROTOCOL['target_anneal'])
    spread = ParticleRegularizer(weight=PROTOCOL['vicreg_weight'])
    opt_g = torch.optim.Adam(list(head.parameters()) + list(prior.parameters()), lr=PROTOCOL['lr'], betas=(PROTOCOL['beta1'], PROTOCOL['beta2']))
    opt_d = torch.optim.Adam(critic.parameters(), lr=PROTOCOL['lr'], betas=(PROTOCOL['beta1'], PROTOCOL['beta2']))
    if noise_policy is not None:
        noise_policy.register_generator_optimizer(opt_g, opt_d)
    _emit({'event': 'config', 'family': 'residual_student', 'pairing': pairing, 'both_land_rows': int(mask.sum()), 'fm_weight': PROTOCOL['fm_weight'], 'cover_weight': PROTOCOL['cover_weight'], 'reg_arm': regularizer.arm, 'reg_coeff': regularizer.coeff, 'reg_kappa': regularizer.kappa, 'reg_norm': regularizer.norm, 'n_particles': PROTOCOL['n_particles'], 'particle_l2': PROTOCOL['particle_l2'], 'vicreg_weight': PROTOCOL['vicreg_weight'], 'residual_weight': RESIDUAL_WEIGHT, 'land_tol': LAND_TOL, 'slow_impact_max': SLOW_IMPACT_MAX, 'steps': PROTOCOL['steps'], 'seed': PROTOCOL['seed'], 'pass_identity_mse': PASS_IDENTITY_MSE, 'success_min': SUCCESS_MIN}, echo, log)
    steps = PROTOCOL['steps']
    both = int(mask.sum())
    for step in range(1, steps + 1):
        if noise_policy is not None:
            noise_policy.set_step(step - 1)
        for _game_pass in _game_correction.passes():
            context = noise_policy.discriminator() if noise_policy is not None else nullcontext()
            with context:
                fake = head(slow, prior.z)
            opt_d.zero_grad(set_to_none=True)
            d_loss = gan.d_loss(critic(slow, paired), critic(slow, fake.detach()))
            view.slow = slow.detach()
            d_loss = d_loss + regularizer(view, paired, fake.detach(), step=step)
            d_loss.backward()
            schedule_optimizer(opt_d, step - 1)
            opt_d.step()
            flags = [p.requires_grad for p in critic.parameters()]
            critic.requires_grad_(False)
            try:
                opt_g.zero_grad(set_to_none=True)
                fake = head(slow, prior.z)
                g_loss = gan.g_loss(critic(slow, fake), critic(slow, paired).detach())
                g_loss = g_loss + PROTOCOL['cover_weight'] * _cover(fake, fast)
                g_loss = g_loss + PROTOCOL['particle_l2'] * prior.z.square().mean()
                g_loss = g_loss + spread(prior.z)
                if both:
                    residual = (fake[mask] - fast[mask]).pow(2).mean()
                else:
                    residual = fake.new_zeros(())
                g_loss = g_loss + RESIDUAL_WEIGHT * residual
                g_loss.backward()
                schedule_optimizer(opt_g, step - 1)
                opt_g.step()
            finally:
                for parameter, flag in zip(critic.parameters(), flags):
                    parameter.requires_grad_(flag)

        def observe_student():
            context = noise_policy.evaluation(step) if noise_policy is not None else nullcontext()
            with torch.no_grad(), context:
                pred = head(slow, prior.z)
                return {'identity_mse': identity_mse(pred, fast), **landing_stats(pred, fast)}
        checkpoint(step, observe_student)
        if step == 1 or step % LOG_EVERY == 0 or step == steps:
            context = noise_policy.evaluation(step) if noise_policy is not None else nullcontext()
            with torch.no_grad(), context:
                pred = head(slow, prior.z)
                mse = identity_mse(pred, fast)
                stats = landing_stats(pred, fast)
            _emit({'event': 'step', 'step': step, 'pairing': pairing, 'd_loss': float(d_loss.detach()), 'g_loss': float(g_loss.detach()), 'residual_mse': float(residual.detach()), 'identity_mse': mse, 'success_rate': stats['success_rate'], 'wrong_pad_rate': stats['wrong_pad_rate']}, echo, log)
    context = noise_policy.evaluation(steps) if noise_policy is not None else nullcontext()
    with torch.no_grad(), context:
        pred = head(slow, prior.z)
        mse = identity_mse(pred, fast)
        paired_mse = identity_mse(pred, paired)
        stats = landing_stats(pred, fast)
    ok = passed(mse, stats['success_rate'], stats['wrong_pad_rate'])
    result = {'event': 'done', 'family': 'residual_student', 'pairing': pairing, 'both_land_rows': both, 'identity_mse': mse, 'paired_target_mse': paired_mse, 'success_rate': stats['success_rate'], 'wrong_pad_rate': stats['wrong_pad_rate'], 'endpoint_l2': stats['endpoint_l2'], 'pass': ok, 'pass_identity_mse': PASS_IDENTITY_MSE, 'success_min': SUCCESS_MIN, 'land_tol': LAND_TOL, 'slow_impact_max': SLOW_IMPACT_MAX, 'residual_weight': RESIDUAL_WEIGHT, 'fm_weight': PROTOCOL['fm_weight'], 'cover_weight': PROTOCOL['cover_weight'], 'reg_arm': regularizer.arm, 'reg_coeff': regularizer.coeff, 'reg_kappa': regularizer.kappa, 'reg_norm': regularizer.norm, 'n_particles': prior.num_particles, 'particle_l2': PROTOCOL['particle_l2'], 'vicreg_weight': PROTOCOL['vicreg_weight'], 'steps': steps, 'seed': PROTOCOL['seed']}
    _emit(result, echo, log)
    return result
