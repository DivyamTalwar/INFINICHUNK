# Upstream provenance and scope

INFINICHUNK currently vendors a snapshot of two upstream projects:

1. **VERL**, Volcano Engine's reinforcement-learning framework:
   https://github.com/volcengine/verl
2. **The Markovian Thinker / Delethink**, the direct source of the bounded
   head-tail carryover loop, trainer integration, reward manager, and example
   configurations:
   https://github.com/McGill-NLP/the-markovian-thinker

The 2026-08-23 audit compared this repository with Markovian Thinker commit
`7b870e93db7cc9beb105fd77950ff9890182b063`. After normalizing the
`delethink`/`infinichunk` names, the core agent-loop, trimmer, trainer, reward
manager, and 24K training script were identical or mechanically renamed.
Those components are inherited work, not an independent INFINICHUNK algorithm.

The vendored `pyproject.toml` still installs a package named `verl`. Do not
publish that package under a new release identity. New INFINICHUNK work should
live under a separate `infinichunk_ext` namespace and depend on a pinned
upstream, or be maintained as an explicitly identified fork until stable VERL
extension points exist.

## Result policy

Figures inherited from the upstream paper/repository must be labeled
**upstream results**. A result may be labeled **reproduced by INFINICHUNK** only
when raw trajectories, configuration, seed, hardware/software manifest, exact
commands, and the relevant upstream/INFINICHUNK commits are published.

This document is an engineering provenance record, not legal advice. The
repository license and source-file notices remain authoritative.
