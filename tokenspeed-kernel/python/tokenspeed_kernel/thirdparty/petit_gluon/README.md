# Gluon Petit MegaMoE

This directory contains the vendored runtime closure of the Gluon MegaMoE
implementation derived from
[causalflow-ai/petit-kernel](https://github.com/causalflow-ai/petit-kernel),
whose package metadata identifies version 0.0.5. The upstream BSD 3-Clause
license is reproduced in `LICENSE.txt`.

The source preserves Petit's kernel algorithms, schedules, layouts, solution
IDs, launch behavior, and lazy HIP VMM binding compilation. Imports rooted at
the upstream `lib` package and relative public-package imports were qualified
with the vendored TokenSpeed package path. Upstream Triton imports are resolved
to `tokenspeed_triton` by the package entry point.

The retained runtime supports the registered GFX950 MegaMoE configurations,
including GPT-OSS 120B (EP8, 128 experts, top-4, 2880x3072, biased OpenAI
SwiGLU) and DeepSeek V4 (EP8, 384 experts, top-6, 7168x3072, bias-free SiLU).
The HIP VMM binding is compiled on first use through
`torch.utils.cpp_extension`, so a ROCm development environment is required.

This vendored source is not registered as a TokenSpeed kernel by itself. A
separate adapter must expose it through `tokenspeed_kernel.ops`.

## Integration status

The source is intentionally kept at the upstream Gluon kernel contract. The
currently pinned `tokenspeed-triton` compiler rejects two constructs used by
that contract: numeric LDS pointer address space `3` and the 8196-word shared
allocation used by MegaMoE. Resolving those requires matching compiler support
or a separately reviewed kernel adaptation; neither is part of this direct
source port.
