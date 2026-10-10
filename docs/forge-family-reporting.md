# Maintaining researcher-facing family reports

The current [technique inventory](../reports/forge/technique-inventory.md) links to
one generated explanation and results page per solution family. Edit the source
metadata below, then regenerate; do not edit generated family Markdown directly.

## Names and tags

`configs/forge/trainer-families.json` owns family display labels and tags. A
presentation rename keeps the family ID, candidate IDs, API preset names, URLs
and frozen scientific evidence intact.

Tags are reusable categories, not grades or qualification claims. Define each
lowercase hyphenated tag once in `configs/forge/family-tags.json`, then assign a
sorted, duplicate-free list to each family. Use tags for characteristics shared
by the registered variants; explain configuration-dependent properties in the
training-details table. For example, a critic-input gradient penalty and
parameter-gradient clipping are different mechanisms. Do not infer one from the
other. The generated inventory includes a tag directory, and inventory JSON
retains tags and definitions for future filtering.

## Technique descriptions

Each current or historical linked family has one file in
`configs/forge/family-documentation/<family-id>.json`:

- `overview`: a plain-language explanation of the mechanism and purpose.
- `symbols`: definitions of the symbols and operations used in the pseudocode.
- `pseudocode`: lines describing the training update, including special rules.
- `training_details`: explanations for `loss`, `optimizer`, `learning_rates`,
  `gradient_clipping`, `critic_penalty`, `damping_and_guards`, `noise`, and
  `averaging`. State explicitly when a mechanism is disabled.
- `configuration_notes`: differences between the family, its variants, selected
  measurements, and task-owned conditions.
- `references`: paper title, verified primary-source URL, and the component it
  supports. Mark related work clearly; a citation must not imply that a paper
  specifies the repository's complete technique. An empty list explicitly means
  no dedicated paper is cited.
- `sources`: repository-relative files used to check the explanation, or pinned
  GitHub source URLs for archived implementations.
- Optional `equations`: a list of objects with `label`, `latex`, and
  `explanation`. Supply the LaTeX body without dollar delimiters or code fences;
  the generator places it in a display-math block outside the pseudocode fence.
  Define every symbol in `symbols` before the equations, and explain each
  equation's role in plain language.
- Optional `illustration`: one repository-local PNG, JPEG, GIF, or WebP asset
  with `path`, descriptive `alt`, and `caption`. The generator places it beside
  the overview and records its file hash. Explain the conceptual mechanism;
  identify conceptual images as illustrations rather than measured results.

For example, add these optional fields to an existing description (JSON needs
doubled backslashes for LaTeX commands):

```json
{
  "equations": [
    {
      "label": "Critic objective",
      "latex": "L_D = L_{\\mathrm{adv}} + \\lambda R(D)",
      "explanation": "The critic minimizes its adversarial loss plus the penalty."
    }
  ],
  "illustration": {
    "path": "reports/forge/families/assets/bcap-explainer.png",
    "alt": "Real and generated samples feed two critic input-gradient penalties.",
    "caption": "Conceptual BCAP mechanism; this illustration is not measured data."
  }
}
```

GitHub renders LaTeX inside `$$` display-math blocks; ordinary text fences retain
readable pseudocode. See [GitHub's mathematical-expression documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
and [image and alt-text syntax](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax#images).

Read the implementation, resolved recipe overrides, and recorded source bindings
before writing. Define terms such as A2, anchors, and spike guards where they
first appear. Show the actual special rule in pseudocode rather than hiding it
behind an unexplained helper name. Distinguish task-owned losses, initialization,
priors and sampling laws from technique mechanisms. Current descriptions do not
retroactively change what an archived run executed.

## Refresh and review

From the repository root in the project's Python environment:

```sh
python reports/forge/regenerate_technique_inventory.py --refresh-publication
python -m experiments.forge compile --summaries-only
python -m experiments.forge compile --check
python -m experiments.forge validate
```

This refresh updates the existing leaderboard and family navigation without
training or regrading qualifications. Review pseudocode and tags against their
sources, verify links, and check that numerical results and scientific selection
identities are unchanged. References appear at the end of each family page.
Review formulas against the implementation as well as the prose. An image is
optional; explain the technique fully in text and mathematics so the page remains
useful without it. Existing descriptions without these fields keep their layout.
