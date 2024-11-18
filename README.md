# dubcmp
```julia
using Pkg
Pkg.generate("DubCmp")
Pkg.add(path="/Users/tangjianfeng/code/julia_work/dubcmp/DubCmp")
Pkg.activate("DubCmp")
Pkg.add("Test")
Pkg.instantiate()
```

# 编译lib
```julia
julia> using Pkg

julia> Pkg.activate(".")
  Activating new project at `~/code/julia_work/dubcmp`

julia> using PackageCompiler

julia> create_library("DubCmp", "build/dubcmp", lib_name="libDubCmp")
```