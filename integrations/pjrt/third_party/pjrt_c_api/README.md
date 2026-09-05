# pjrt_c_api

This directory contains a fork of the C headers and .proto files
needed to build a PJRT plugin.

`pjrt_c_api.h` is copied verbatim. Re-vendor with
`./revendor.py --from <checkout>`, which also regenerates the stub
table in `src/iree_pjrt/common/stubs.inc`; see the script's docstring
for why that has to be generated rather than maintained by hand.

Last synced from:

* https://github.com/openxla/xla.git
* commit: 6b73c4c49eb0c5aee1963c1e34c7cc8ba92ac202
* PJRT API version: 0.114
