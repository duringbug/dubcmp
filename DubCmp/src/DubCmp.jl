module DubCmp
export greet, Train
Base.@ccallable function greet()::Cstring
    msg = "Hello World!" * '\0'  # 确保字符串以空字符结尾
    return pointer(msg)  # 返回 C 字符串指针
end
include("math/Train.jl")
end # module DubCmp
