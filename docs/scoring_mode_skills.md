# Scoring-mode development skills

A skill is a folder of written instructions (a `SKILL.md` plus references) that
a coding agent reads to follow a repeatable workflow. The repository includes
two for adding scoring modes:

- **`design-scoring-mode`** develops and tests a PyTorch reference objective
  and single-pair alignment method from a mathematical or plain-language description.
- **`accelerate-scoring-mode`** adds the mode and any new features to the
  registries, reuses or implements Numba/Triton kernels, and checks the batched
  and screening paths against the reference.

The skills and their supporting references live in
`.claude/skills/design-scoring-mode/` and
`.claude/skills/accelerate-scoring-mode/` at the repository root. Each folder
contains a `SKILL.md` entry point. Read it directly, or ask a coding agent that
supports skills to use it in this repository. To install a skill for another
agent, copy the whole folder, including its supporting references. The skills
are instructions for development; nothing in them runs when `shepherd_score` is
imported.

Both folders are in source distributions, and wheels install them under
`share/shepherd-score/skills/` in the installation's data directory
(`Path(sysconfig.get_path("data")) / "share/shepherd-score/skills"`); for
editable installs, use the repository folders.

The reference workflow tests the objective and its gradients before any kernel
work. The acceleration workflow compares fixed-pose values and gradients,
optimizer settings, returned transforms, stored features and each execution
path against the reference, and accounts for differing seed sets, sub-batch
early stopping and frozen Triton launch configurations. The included test
template fails until its gates are filled in for the new objective.

Review generated code as you would any other contribution: a self-score of one
and agreement on a few optimized pairs do not show the objective is correct.
This page covers the agent workflow; the mechanism is described under
[Adding a mode](modes.md).
