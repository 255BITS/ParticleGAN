def train(recipe: UnusedHoldRecipe, regularizer: GradientPenalty | None=None, *, noise_policy=None) -> dict:
    """Fit one arm. Prints a tailable line at the checkpoints."""
    torch.manual_seed(int(recipe.seed))
    student = SharedSlotStudent()
    critic = SlotCritic()
    if noise_policy is not None:
        from benchmarks.transfer_suite.legacy_noise_adapters import wrap_input
        critic = wrap_input(critic, noise_policy)
    gan = GANLoss(loss_type=recipe.loss_type, mode=recipe.gan_mode)
    reg = regularizer if regularizer is not None else _make_regularizer(recipe)
    if noise_policy is not None:
        noise_policy.register_generator_base(student)
    g_parameters = list(student.parameters()) + (noise_policy.scale_parameters() if noise_policy is not None else [])
    opt_g = torch.optim.Adam(g_parameters, lr=LR, betas=BETAS)
    opt_d = torch.optim.Adam(critic.parameters(), lr=LR, betas=BETAS)
    if noise_policy is not None:
        noise_policy.register_generator_optimizer(opt_g, opt_d)
    real = _batch(CONCEPT_DIR)
    pairs = hold_pairs(recipe.pairing)
    bcap_applied = 0
    for step in range(int(recipe.steps)):
        if noise_policy is not None:
            noise_policy.set_step(step)
        for _game_pass in _game_correction.passes():
            fake = student.embeds(1.0)[CONCEPT].unsqueeze(0).expand(N_ROWS, -1).detach()
            if noise_policy is not None:
                fake = noise_policy.output(fake, generator_step=False)
            opt_d.zero_grad(set_to_none=True)
            penalty, stats = reg.penalty(critic, real, fake, step=step + 1)
            if stats.get('applied'):
                bcap_applied += 1
            d_loss = gan.d_loss(critic(real), critic(fake)) + penalty
            d_loss.backward()
            schedule_optimizer(opt_d, step)
            opt_d.step()
            critic.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            fake_g = student.embeds(1.0)[CONCEPT].unsqueeze(0).expand(N_ROWS, -1)
            if noise_policy is not None:
                fake_g = noise_policy.output(fake_g, generator_step=True)
            g_loss = gan.g_loss(critic(fake_g), critic(real).detach())
            if float(recipe.fm_weight) != 0.0:
                real_feat = critic.features(real).detach().mean(0)
                fake_feat = critic.features(fake_g).mean(0)
                g_loss = g_loss + float(recipe.fm_weight) * (real_feat - fake_feat).pow(2).mean()
            loss = g_loss
            if float(recipe.hold_weight) != 0.0:
                embeds = student.embeds(1.0)
                loss = loss + float(recipe.hold_weight) * unused_hold_loss(embeds, student.neu, pairs)
            loss.backward()
            critic.requires_grad_(True)
            schedule_optimizer(opt_g, step)
            opt_g.step()
        checkpoint(step + 1, lambda: score_student(student))
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == int(recipe.steps):
            _log(recipe.name, step + 1, score_student(student))
    metrics = score_student(student)
    return metrics
