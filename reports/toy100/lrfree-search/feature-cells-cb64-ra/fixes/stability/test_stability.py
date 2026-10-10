"""Focused CPU regressions for actual routing, checkpoints and mass supply."""
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pkg"))
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from particlegan import GANTrainer, get_recipe
from particlegan.birth_death import ParticleBirthDeath
from particlegan.feature_cells import (FeatureCellBirthDeath, FeatureCellSnapshot,
                                      SmallPopulationReferenceBirthDeath, population_policy)

HARNESS = Path("/ml2/hypergan/lrfree-20260926/harness")
CONFIG = Path("/ml2/hypergan/gan-attempts/feature-cells-config-20260929/configs/overrides-CB64-RA.json")
SEED = 90229


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


MODE = module("stability_mode_host", HARNESS / "hosts/mode_hold_host.py")
IMAGE = module("stability_image_host", HARNESS / "hosts/image_host.py")
IMAGE_SPEC = json.loads((HARNESS / "tasks/image_task_specs.json").read_text())["img_blobs4"]


def stream(seed=SEED):
    return torch.Generator().manual_seed(seed)


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(
            a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a if key != "eval_seconds")
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b


def host_trainer(task, backend="feature_cells"):
    torch.manual_seed(0)
    options = json.loads(CONFIG.read_text())
    options["birth_death_backend"] = backend
    data_stream = stream(0)
    if task == "mode_hold":
        options.update(num_particles=12, z_dim=4, batch_size=128)
        recipe = get_recipe(**options)
        prior = recipe.make_prior(init_std=.5, generator=data_stream)
        generator = MODE.SimpleMLPGenerator(4, 96, 3, 2)
        critic = MODE.SimpleMLPDiscriminator(2, 96, 3, 3)
    else:
        spec = IMAGE_SPEC
        options.update(num_particles=spec["particles"], z_dim=spec["z_dim"], batch_size=spec["batch_size"])
        recipe = get_recipe(**options)
        generator, critic = IMAGE.Generator(spec), IMAGE.Discriminator(spec)
        prior = recipe.make_prior()
    return GANTrainer(recipe, generator, critic, prior=prior, seed=0,
                      latent_generator=data_stream, serial_backward=True)


def host_batches(task, count):
    generator = stream(0)
    if task == "mode_hold":
        means = MODE.ring_means()
        return [MODE.sample_ring(means, 128, MODE.SIGMA, generator) for _ in range(count)]
    centers = IMAGE.templates(IMAGE_SPEC)
    result = []
    for _ in range(count):
        real = centers[torch.randint(len(centers), (IMAGE_SPEC["batch_size"],), generator=generator)]
        noise = torch.randn(real.shape, generator=generator)
        result.append((real + IMAGE_SPEC["noise_std"] * noise).clamp(0., 1.))
    return result


def two_cells():
    real = torch.cat((torch.randn((512, 2), generator=stream(), dtype=torch.float64)*.02-3,
                      torch.randn((512, 2), generator=stream(1), dtype=torch.float64)*.02+3))
    real = real[torch.randperm(1024, generator=stream())]
    return real, FeatureCellSnapshot.fit(real, generator=stream(), cells=2, rank=2)


class Stability(unittest.TestCase):
    def test_exact_finite_resolution_and_boundary(self):
        for n in (6, 12, 32, 256, 512, 799):
            policy = population_policy(n)
            self.assertFalse(policy["finite_resolution_feasible"])
            self.assertEqual(policy["actual_backend"], "knn")
            self.assertGreater(policy["minimum_bh_flags"], policy["maximum_guard_flags"])
        for n in (800, 801, 1024, 20000):
            self.assertTrue(population_policy(n)["finite_resolution_feasible"])
        self.assertEqual(population_policy(12)["maximum_guard_flags"], 0)
        self.assertEqual(population_policy(800)["minimum_bh_flags"], 40)
        self.assertEqual(population_policy(800)["maximum_guard_flags"], 40)

    def test_existing_small_hosts_exact_reference_law(self):
        for task in ("mode_hold", "img_blobs4"):
            with self.subTest(task=task):
                reference = host_trainer(task, "knn")
                candidate = host_trainer(task)
                self.assertIs(type(reference.birth_death), ParticleBirthDeath)
                self.assertIs(type(candidate.birth_death), SmallPopulationReferenceBirthDeath)
                self.assertFalse(hasattr(candidate.birth_death, "perturb_latent"))
                # Six deterministic CPU updates test the real host models,
                # optimizer and reference reaction dispatch, not efficacy.
                for real in host_batches(task, 6):
                    reference.step(real, generator_real=real, collect_stats=True)
                    candidate.step(real, generator_real=real, collect_stats=True)
                left, right = reference.state_dict(), candidate.state_dict()
                left.pop("recipe"); right.pop("recipe")
                metadata = right["birth_death"].pop("population_policy")
                self.assertEqual(metadata["matching_sampler"], "controller_reference")
                self.assertTrue(same(left, right))
                g1, g2 = stream(), stream()
                latent = reference.prior.z.detach()
                a = reference._generate(reference.G, latent, reference.output_sigma(), g1)
                b = candidate._generate(candidate.G, latent, candidate.output_sigma(), g2)
                self.assertTrue(torch.equal(a, b))
                self.assertTrue(torch.equal(g1.get_state(), g2.get_state()))

    def test_small_checkpoint_reload_and_route_mismatch(self):
        trainer = host_trainer("mode_hold")
        for real in host_batches("mode_hold", 2):
            trainer.step(real, generator_real=real)
        saved = trainer.state_dict()
        restored = host_trainer("mode_hold")
        restored.load_state_dict(saved)
        self.assertTrue(same(saved, restored.state_dict()))
        for key, value in (("actual_backend", "feature_cells"), ("matching_sampler", "feature_cells"),
                           ("population", 32), ("minimum_bh_flags", 1)):
            bad = deepcopy(saved)
            bad["birth_death"]["population_policy"][key] = value
            before = restored.state_dict()
            with self.assertRaisesRegex(ValueError, "metadata"):
                restored.load_state_dict(bad)
            self.assertTrue(same(before, restored.state_dict()))
        bad = deepcopy(saved)
        bad["birth_death"].pop("population_policy")
        with self.assertRaisesRegex(ValueError, "metadata"):
            restored.load_state_dict(bad)

    def test_active_checkpoint_reload_and_mass_policy_mismatch(self):
        def construct():
            torch.manual_seed(0)
            options = json.loads(CONFIG.read_text())
            options.update(num_particles=1024,z_dim=2,batch_size=128)
            recipe = get_recipe(**options)
            generator = MODE.SimpleMLPGenerator(2,32,2,2)
            critic = MODE.SimpleMLPDiscriminator(2,32,2,2)
            return GANTrainer(recipe,generator,critic,seed=0,serial_backward=True)
        trainer = construct()
        self.assertIs(type(trainer.birth_death),FeatureCellBirthDeath)
        self.assertEqual(trainer.birth_death.population_policy["actual_backend"],"feature_cells")
        saved = trainer.state_dict()
        restored = construct(); restored.load_state_dict(saved)
        self.assertTrue(same(saved,restored.state_dict()))
        for path in ("settings","population_policy"):
            bad = deepcopy(saved)
            if path == "settings":
                bad["birth_death"][path]["mass_policy"] = "old_unconstrained"
            else:
                bad["birth_death"][path]["actual_backend"] = "knn"
            before = restored.state_dict()
            with self.assertRaisesRegex(ValueError,"backend/settings"):
                restored.load_state_dict(bad)
            self.assertTrue(same(before,restored.state_dict()))

    def test_macro_balanced_narrow_support_does_not_change_mass(self):
        fixture = module("stability_test_cost_fixture",Path(
            "/ml2/hypergan/gan-attempts/scaling-portability-20260929/scaling_a/shared_toy.py"))
        for scenario in ("nominal","rare_hole"):
            toy,query,real,oracle = fixture.make_fixture(1024,scenario)
            # Undo the known planted intervention in fixture preparation;
            # no oracle field is supplied to the controller.
            if scenario == "nominal":
                query[torch.from_numpy(oracle["planted"]),1] -= 2.2
            else:
                query[torch.from_numpy(oracle["planted"]),:2] = torch.tensor(
                    fixture.CENTERS[oracle["original_labels"][oracle["planted"]]],dtype=torch.float64)
            q,r = toy.capture(query),toy.capture(real)
            snap = FeatureCellSnapshot.fit(r,generator=stream())
            ids,_ = snap.assign(q)
            target = snap.reference_counts+snap.real_calibration_counts
            counts = torch.bincount(ids,minlength=snap.cells)
            self.assertGreater(int((counts-target).abs().sum()),600)
            group_ids = snap._mass_topology()
            self.assertEqual(snap.mass_groups,8)
            self.assertTrue(torch.equal(snap._group_counts(counts),snap._group_counts(target)))
            flags,p,_ = snap.support(q)
            comparison = snap.cell_comparison(q)
            child,parent,detail = snap.ordinary_transport(q,flags,comparison,generator=stream(),pvalues=p)
            self.assertEqual(detail["between_group_moves"],0)
            self.assertTrue(bool((group_ids[ids[child]]==group_ids[ids[parent]]).all()))
            self.assertEqual(len(parent),len(torch.unique(parent)))
            # Derived action topology leaves the original detector and count
            # evidence bit-identical on this snapshot.
            after_flags,after_p,_ = snap.support(q)
            after_comparison = snap.cell_comparison(q)
            self.assertTrue(torch.equal(flags,after_flags)); self.assertTrue(torch.equal(p,after_p))
            self.assertTrue(same(comparison,after_comparison))

    def test_ordinary_supply_budget_and_clean_balance(self):
        real, snap = two_cells()
        query = torch.cat((torch.full((900, 2), -3., dtype=torch.float64),
                           torch.full((124, 2), 3., dtype=torch.float64)))
        flags, p = torch.zeros(1024, dtype=torch.bool), torch.ones(1024)
        comparison = snap.cell_comparison(query)
        child, parent, detail = snap.ordinary_transport(query, flags, comparison, generator=stream(), pvalues=p)
        self.assertGreater(len(child), 0)
        self.assertLessEqual(len(child), 51)
        self.assertEqual(len(child), len(torch.unique(child)))
        self.assertEqual(len(parent), len(torch.unique(parent)))
        self.assertFalse(bool(torch.isin(parent, child).any()))
        self.assertTrue(bool((detail["birth_allocation"] <= detail["clean_vacancies"]).all()))
        self.assertTrue(bool((detail["birth_allocation"] <= detail["eligible_parent_counts"]).all()))
        sparse_p = p.clone(); sparse_p[:900] = .01; sparse_p[0] = 1.
        child,parent,detail = snap.ordinary_transport(query,flags,comparison,generator=stream(),pvalues=sparse_p)
        self.assertGreater(len(child),0)
        self.assertFalse(bool((child==0).any()))
        # Strong emitted evidence alone cannot move a clean table that already
        # has exactly the reference cell masses.
        balanced = real.clone()
        extreme = torch.full_like(real, -3.)
        comparison = snap.cell_comparison(extreme)
        self.assertGreater(int(comparison["excess"].sum()), 0)
        child, parent, detail = snap.ordinary_transport(balanced, flags, comparison, generator=stream(), pvalues=p)
        self.assertEqual(len(child), 0)
        self.assertTrue(bool((detail["clean_counts"] == detail["target_counts"]).all()))

    def test_isolation_supply_unique_parents_and_guard(self):
        _, snap = two_cells()
        query = torch.cat((torch.full((900, 2), -3., dtype=torch.float64),
                           torch.full((124, 2), 3., dtype=torch.float64)))
        flags = torch.zeros(1024, dtype=torch.bool); flags[:46] = True
        p = torch.full((1024,), .01); p[1000] = 1.
        child, parent, detail = snap.select_parents(query, flags, generator=stream(), pvalues=p)
        self.assertEqual(len(child), 1)
        self.assertEqual(parent.tolist(), [1000])
        self.assertEqual(detail["unfilled_repairs"], 45)
        self.assertTrue(bool(((detail["candidate_ids"] == parent[:,None]) & detail["candidate_mask"]).any(1).all()))
        too_many = flags.clone(); too_many[:52] = True
        child, parent, detail = snap.select_parents(query, too_many, generator=stream(), pvalues=p)
        self.assertEqual(len(child), 0)
        self.assertFalse(detail["guard_passed"])
        # Planned ordinary parents cannot be reused by isolation.
        child, parent, detail = snap.select_parents(query, flags, ordinary_children=torch.tensor([500]),
            ordinary_parents=torch.tensor([1000]), generator=stream(), pvalues=p)
        self.assertEqual(len(child), 0)

    def test_rare_cell_no_inflation_and_no_supported_parent_loss(self):
        generator = stream()
        real = torch.randn((1024, 2), generator=generator, dtype=torch.float64)*.01-3.
        real[-2:] += 6.
        snap = FeatureCellSnapshot.fit(real, generator=stream(), cells=2, rank=2)
        query = real.clone(); query[:46] = torch.tensor([4., 4.], dtype=torch.float64)
        flags = torch.zeros(1024, dtype=torch.bool); flags[:46] = True
        p = torch.ones(1024)
        child, parent, detail = snap.select_parents(query, flags, generator=stream(), pvalues=p)
        self.assertEqual(len(child), 46)
        self.assertEqual(len(torch.unique(parent)), 46)
        before_ids, _ = snap.assign(query)
        rare_cell = int(snap.assign(real[-1:])[0][0])
        self.assertTrue(bool((detail["parent_cell_ids"] != rare_cell).all()))
        self.assertTrue(bool((detail["parent_cell_ids"] == detail["anchor_cell_ids"]).all()))
        after = query.clone(); after[child] = query[parent]
        after_ids, _ = snap.assign(after)
        self.assertEqual(int((after_ids == rare_cell).sum()), 2)
        self.assertEqual(int((before_ids[-2:] == rare_cell).sum()), 2)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Stability)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = dict(scope="CPU contract/mechanism regressions, no CUDA quality verdict",
                   tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                   success=result.wasSuccessful())
    (ROOT / "regressions.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
