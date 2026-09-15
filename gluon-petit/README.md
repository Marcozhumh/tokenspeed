## Goal

This gluon-petit project is rewrite the native petit ( petit-kernel/ ), which is written in C++/HIP, again in gluon.

## Requirement

- Faithfully port the native Petit code, function by function to gluon
- Match the file organization of the native petit

## Test Flow

- Build petit-kernel with Clang23, put petit-kernel build under petit-kernel/build/
- Correctness tests must pass
- Use rocprofv3 to capture the kernel latency, report the numbers vs the native petit