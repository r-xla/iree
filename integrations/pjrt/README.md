# IREE PJRT Plugin

This directory contains an experimental PJRT plugin library which can bridge
Jax (and TensorFlow in the future) to IREE.

# Developing

Support for dynamically loaded PJRT plugins is brand new as of 12/21/2022 and
there are sharp edges still. The following procedure is being used to develop.

There are multiple development workflows, ranked from easiest to hardest (but
most powerful).

## Install a compatible version of Jax and the IREE compiler

```shell
pip install -r requirements.txt

pip install jax==0.6.1
```

Verify that your Jax install is functional like:

```shell
python -c "import jax; a = jax.numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9]); print(a + a);"
```

## Install the plugin of your choice (in this example 'cpu')

```shell
pip install -v --no-deps -e python_packages/iree_cpu_plugin
```

## Verify basic functionality

```shell
JAX_PLATFORMS=iree_cpu python -c "import jax; a = jax.numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9]); print(a + a);"
```

## Required compiler options for f64 / i64 inputs (r-xla fork)

IREE's default input pipeline **demotes f64 to f32 and i64 to i32**, and in
doing so rewrites the public function signature. The runtime then rejects the
f64 buffers a caller hands it:

```
input0 element type mismatch; expected f32 (21000020) but have f64 (21000040)
```

The compiler does warn (`ConvertTypesPass ... changed public function
signatures`), but nothing fails at compile time, so a client that only checks
for compile errors sees a successful compile and an execution failure.

For a caller whose natural types are 64-bit -- R's `numeric` is f64 and its
`integer` maps to i64 in this stack -- pass:

```shell
export IREE_PJRT_IREE_COMPILER_OPTIONS="\
  --iree-input-demote-f64-to-f32=false \
  --iree-llvmcpu-link-embedded=false \
  --iree-opt-const-eval=false"
```

All three are required, and the second two only because of the first.

`--iree-llvmcpu-link-embedded=false`: keeping f64 makes the generated kernels
call the f64 libm entry points (`exp`, `log`, `sin`, `cos`, `tanh`, `atan2`,
`fmod` -- that is the complete set), and IREE's *embedded* ELF ships only the
f32 ones (`runtime/src/iree/builtins/musl/` contains two checked-in bitcode
blobs whose math symbols are all `*f`). Linking otherwise fails with
`iree-lld: error: undefined symbol: exp`. Turning embedded linking off
produces a system dylib that resolves them against the host libm through the
`system-library` loader, which has to be enabled in the build (it is, by
default).

`--iree-opt-const-eval=false`: without it, a program whose f64 transcendentals
land in a const-eval'd global initializer *still* fails to link, with the same
`undefined symbol: exp` and a giveaway `jit_eval_*` symbol name. The reason is
that the compile-time const-eval JIT builds its own target via
`LLVMTarget::createForHost()`, which hardcodes `requestLinkEmbedded=true` and
does not consult the flag above. This is not CPU-specific: const-eval always
resolves to the host `local` device, so a CUDA or ROCm client with f64
preserved hits the same wall.

`--iree-input-demote-i64-to-i32=false` is **not** needed: the plugin already
sets it for every compile (see `SetupJob` in `common/iree_compiler.cc`), and
IREE's own default is false.

Known to remain broken at f64 with these flags: `cbrt` (IREE's math lowering
has no f64 `math.cbrt` path -- an MLIR-level gap, not a link error) and
`arith.sitofp` from i32 to f64 (unimplemented in the VM conversion, though the
opcode exists; keep host-side integers at i64 to avoid it).

A cleaner fix for the whole f64 story would be `--iree-link-bitcode`, which
*does* reach const-eval and would let embedded linking stay on. It needs an
f64 libm bitcode module of real quality: it is linked with `OverrideFromSrc`
and `alwaysinline`, so anything less than musl-grade sources would shadow the
host libm for runtime kernels too and trade a loud link error for quiet
numerical error.

These are deliberately left as options rather than defaults: they are the
right choice for a 64-bit host caller and the wrong one for a client that
wants f32 performance, and only the caller knows which it is.

## Advanced settings

To pass additional compile options to IREE during JIT compilation, you can use
the `IREE_PJRT_IREE_COMPILER_OPTIONS` environment variable. This variable can
be set to a space-delimited list of flags that would be passed to the
`iree-compile` command-line tool.

For example:
```shell
export IREE_PJRT_IREE_COMPILER_OPTIONS=--iree-scheduling-dump-statistics-format=csv
JAX_PLATFORMS=iree_cpu python -c "import jax; a = jax.numpy.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9]); print(a + a);"
```

Besides, to control logging levels in the IREE PJRT plugin,
you can set `IREE_PJRT_LOG_LEVEL` to `debug` or `error` (default: `debug`).

## Incrementally developing

If you did an editable install (`-e`) above, then you should be able to incrementally
make changes and build the native component with no further interaction needed.

```shell
cd python_packages/iree_cpu_plugin/build/cmake
ninja
```

## Running the Jax test suite

The JAX test suite can be run with pytest. We recommend using `pytest-xdist`
as it spawns tests in workers which can be restarted in the event of individual
test case crashes.

Setup:

```
# Install pytest
pip install pytest pytest-xdist

# Install the ctstools package from this repo (`-e` makes it editable).
pip install -e ctstools
```

Example of running tests:

```
JAX_PLATFORMS=iree_cuda pytest -n4 --max-worker-restart=9999 \
  -p openxla_pjrt_artifacts --openxla-pjrt-artifact-dir=/tmp/foobar \
  ~/src/jax/tests/nn_test.py
```

Note that you will typically want a small number of workers (`-n4` above) for
CUDA and a larger number can be tolerated for cpu.

The plugin `openxla_pjrt_artifacts` is in the `ctstools` directory and
performs additional manipulation of the environment in order to save
compilation artifacts, reproducers, etc.

## Communication channels

* Please submit feature requests and bug reports about the plugin in [GitHub Issues](https://github.com/iree-org/iree/issues).
* Discuss the development of the plugin at `#jax` or `#pjrt-plugin` channel of [IREE Discord server](https://discord.gg/wEWh6Z9nMU).
* Check the [OpenXLA/XLA](https://github.com/openxla/xla) repo and [its communication channels](https://github.com/openxla/community?tab=readme-ov-file#communication-channels) for PJRT APIs and clients.

## License

IREE PJRT plugin is licensed under the terms of the Apache 2.0 License with
LLVM Exceptions. See [LICENSE](../../LICENSE) for more information.

[PJRT C API](./third_party/pjrt_c_api) comes from
[OpenXLA/XLA](https://github.com/openxla/xla) and is licensed under
the Apache 2.0 License. See its own [LICENSE](./third_party/pjrt_c_api/LICENSE) for more information.
