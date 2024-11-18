# test_dubcmp.jl

using Test
using DubCmp  # 引入模块
using LinearAlgebra
using Random  # 用于生成随机数

# 测试 greet() 函数是否返回 "Hello World!"
# @testset "DubCmp Tests" begin
#     # 调用 greet 函数
#     output = DubCmp.greet()

#     # 断言输出是 "Hello World!"
#     @test output == "Hello World!"
# end

@testset "Sys Tests" begin
    output = Sys.WORD_SIZE
    @test output == 64
end

@testset "Large Matrix Multiplication" begin
    # 生成两个 1024x1024 的随机矩阵
    Random.seed!(1234)  # 固定种子以确保结果可重复
    A = rand(1024, 1024) * 100
    B = rand(1024, 1024) * 100
    
    # 测量矩阵乘法时间
    println("Starting matrix multiplication...")
    time_taken = @elapsed C = A * B
    println("Matrix multiplication completed in $(time_taken) seconds.")

    
    # 验证维度
    @test size(C) == (1024, 1024)
    
    # 验证计算结果
    # 因为直接比较浮点数可能有问题，可以使用 isapprox 进行近似比较
    @test isapprox(C, A * B, atol=1e-12)
end
