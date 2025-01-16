# pytroch-playground

## Build a custom op library with triton

Currerntly a todo list:

- Generate `op lists` by `op.yaml` and op `shim` and pointwise op (may also with reduction op)

- Compile kernels (cubin) in python with `Triton compile` and cache them in cpp

- Add flaggems kernels && flash attention kernels

- ~~Add third party gtest and glog~~

Done:

- use torch LOG, ENV : `TORCH_CPP_LOG_LEVEL=INFO`
