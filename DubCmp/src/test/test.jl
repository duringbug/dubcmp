import Pkg
Pkg.add("Distributions")
Pkg.add("StatsBase")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Plots")
Pkg.add("MLDatasets")

using Test
using LinearAlgebra
using Random  # 用于生成随机数
using Distributions
using StatsBase
using CSV
using DataFrames
using Plots
using MLDatasets

# test_dubcmp.jl
include("../DubCmp.jl")
using .DubCmp  # 引入模块


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
    A = rand(1024, 64) * 100
    B = rand(64, 1024) * 100
    
    time_taken = @elapsed C = A * B
    println("Matrix multiplication completed in $(time_taken) seconds.")

    
    # 验证维度
    @test size(C) == (1024, 1024)
    
    # 验证计算结果
    # 因为直接比较浮点数可能有问题，可以使用 isapprox 进行近似比较
    @test isapprox(C, A * B, atol=1e-12)
end

@testset "SVM" begin
    # 生成两个 1024x1024 的随机矩阵
    Random.seed!(1234)  # 固定种子以确保结果可重复
    W = rand(1, 1024) * 100
    x = rand(1024, 1) * 100
    b = 0.0
    println(typeof(W))


    time_taken = @elapsed y = W * x .+ b
    println("Matrix multiplication completed in $(time_taken) seconds.")

    
    # 验证维度
    @test size(y) == (1, 1)
    
    # 验证计算结果
    # 因为直接比较浮点数可能有问题，可以使用 isapprox 进行近似比较
    @test isapprox(y, W * x .+ b, atol=1e-12)
end

@testset "PLA" begin
    input_dim = 2048
    pla = DubCmp.Train.PLA{Float64}(input_dim)

    x1 = rand(input_dim, 1024) * 1 .- 0.5
    W1 = rand(1, input_dim) * 1
    W2 = rand(1, input_dim) * 1
    b1 = 25.0
    y1 = W1 * x1 .+ b1
    true_labels1 = ones(Int64, size(y1, 2))

    x2 = rand(input_dim, 1024) * 1 .- 0.5
    b2 = -25.0
    y2 = W2 * x2 .+ b2
    true_labels2 = zeros(Int64, size(y2, 2))


    X = hcat(x1, x2)
    Y = hcat(y1, y2)
    true_labels = vcat(true_labels1, true_labels2)

    indices = randperm(size(X, 2))
    X_shuffled = X[:, indices]
    Y_shuffled = Y[:, indices]
    true_labels_shuffled = true_labels[indices] 

    DubCmp.Train.train(pla, X_shuffled, Y_shuffled, max_iter = 100, learning_rate = 5e-3, bar = false)
    predictions = DubCmp.Train.predict(pla, X_shuffled)
    
    predicted_labels = predictions .> 0
    true_labels_shuffled_bool = true_labels_shuffled .> 0

    # 计算准确度
    correct_predictions = sum((true_labels_shuffled_bool[:] .== predicted_labels[:]))
    accuracy = correct_predictions / length(true_labels)
    @show accuracy  # 显示准确度
end

# 生成数据函数
function generate_gaussian_data(input_dim::Int, num_samples::Int, num_classes::Int, num_test_samples::Int)
    samples_per_class = num_samples ÷ num_classes
    samples_test_per_class = num_test_samples ÷ num_classes
    X = zeros(Float64, num_samples, input_dim)
    y = Vector{Int}(undef, num_samples)
    # 生成测试数据 (随机生成 64 个样本)
    X_test = zeros(Float64, num_test_samples, input_dim)
    y_test = Vector{Int}(undef, num_test_samples)
    
    # 生成训练测试数据
    for i in 1:num_classes
        mean = randn(input_dim) * 5  # 每个类别的均值
        A = randn(input_dim, input_dim)
        cov = A * A' + I * 0.1  # 保证对称性并且正定，+ I * 0.1 确保数值稳定性
        start_idx = (i - 1) * samples_per_class + 1
        end_idx = i * samples_per_class
        start_idx_test = (i - 1) * samples_test_per_class + 1
        end_idx_test = i * samples_test_per_class
        
        # 使用广播生成每一类的数据
        X[start_idx:end_idx, :] .= rand(MvNormal(mean, cov), samples_per_class)'  # 使用广播生成每一类的数据
        y[start_idx:end_idx] .= i  # 标签为类别

        start_idx = (i - 1) * (num_test_samples ÷ num_classes) + 1
        end_idx = i * (num_test_samples ÷ num_classes)
        
        # 生成测试数据
        X_test[start_idx_test:end_idx_test, :] .= rand(MvNormal(mean, cov), num_test_samples ÷ num_classes)'  # 使用广播生成每一类的数据
        y_test[start_idx_test:end_idx_test] .= i  # 标签为类别
    end
    
    return X, y, X_test, y_test
end

@testset "KNN" begin
    input_dim = 1024
    num_samples = 1024
    num_classes = 8
    num_test_samples = 16 * num_classes  # 测试样本数量
    k = 5  # 设置超参数 k
    
    # 生成数据
    X, y, X_test, y_test = generate_gaussian_data(input_dim, num_samples, num_classes, num_test_samples)
    
    # 检查数据
    @test size(X) == (num_samples, input_dim)
    @test length(y) == num_samples
    @test unique(y) == 1:num_classes
    
    # 实例化 KNN
    knn = DubCmp.Train.KNN{Float64}(X, y, k)
    @test knn.k == k
    y_predict = DubCmp.Train.predict(knn,X_test)

    accuracy = sum(y_predict .== y_test) / length(y_test)
    @show accuracy 
end

# KDTree容易出现维度灾难问题
@testset "KDTree" begin
    input_dim = 64 
    num_samples = 1024
    num_classes = 8
    num_test_samples = 16 * num_classes  # 测试样本数量
    k = 5  # 设置超参数 k
    
    # 生成数据
    X, y, X_test, y_test = generate_gaussian_data(input_dim, num_samples, num_classes, num_test_samples)

    # 检查数据
    @test size(X) == (num_samples, input_dim)
    @test length(y) == num_samples
    @test unique(y) == 1:num_classes

    # 实例化 KDTree
    kdtree = DubCmp.Train.KDTree{Float64}(X, y, input_dim)

    # 检查 KDTree 的构造
    @test kdtree.k == input_dim
    @test size(kdtree.points) == (num_samples, input_dim)
    @test length(kdtree.labels) == num_samples
    
    # 使用 KDTree 进行预测
    y_predict = DubCmp.Train.predict(kdtree, X_test)
    
    # 计算准确率
    accuracy = sum(y_predict .== y_test) / length(y_test)
    @show accuracy
end

@testset "NativeBayes" begin
    data = CSV.File("resources/iris/iris.data", header=false)
    df = DataFrame(data)

    train_ratio = 0.7
    num_samples = nrow(df)  # 获取数据集的总行数
    train_size = Int(floor(train_ratio * num_samples))  # 计算训练集的大小

    random_indices = randperm(num_samples)

    train_indices = random_indices[1:train_size]  # 前 train_size 个作为训练集
    test_indices = random_indices[train_size+1:end]  # 剩下的作为测试集

    train_df = df[train_indices, :]
    test_df = df[test_indices, :]

    nativeBayes = DubCmp.Train.NativeBayes(train_df, 5)
    results = DubCmp.Train.predict(nativeBayes, test_df)

    actual = test_df[:, 5]

    correct_predictions = sum(results .== actual)
    accuracy = correct_predictions / length(actual)

    @show accuracy
end

@testset "DecisionTree" begin
    data = CSV.File("resources/WineQT/WineQT.csv", header=true)
    df = DataFrame(data)

    train_ratio = 0.7
    num_samples = nrow(df)  # 获取数据集的总行数
    train_size = Int(floor(train_ratio * num_samples))  # 计算训练集的大小

    random_indices = randperm(num_samples)

    train_indices = random_indices[1:train_size]  # 前 train_size 个作为训练集
    test_indices = random_indices[train_size+1:end]  # 剩下的作为测试集

    train_df = df[train_indices, :]
    test_df = df[test_indices, :]
end

@testset "TSEN" begin

    # 加载 MNIST 数据集
    train_x, train_y = MNIST(split=:train)[:]
    train_x_flattened = reshape(train_x, 28 * 28, 60000)'

    # 从训练集中随机选取样本
    sample_size = 1000
    sample_indices = rand(1:size(train_x_flattened, 1), sample_size)  # 修正为一维的索引
    sample_data = train_x_flattened[sample_indices, :]
    sample_labels = train_y[sample_indices]

    # 创建 t-SNE 实例并拟合数据
    tsne_model = DubCmp.Train.TSNE(Matrix{Float32}(sample_data), eta=0.03, tol=1e-3, sigma=0.9)
    DubCmp.Train.fit!(tsne_model)

    # 有10个标签（0到9）
    unique_labels = 0:9  # 标签从0到9

    # 创建一个颜色列表，为每个标签指定一个颜色
    label_colors = [:red, :blue, :green, :yellow, :orange, :purple, :cyan, :magenta, :brown, :black]

    # 将标签映射到颜色
    colors = [label_colors[l+1] for l in sample_labels]  # sample_labels是标签数组，假设标签从0开始

    # 可视化降维后的数据
    scatter(tsne_model.Y[:, 1], tsne_model.Y[:, 2],
        color=colors,  # 为每个标签指定的颜色
        title="t-SNE on MNIST",
        xlabel="Dimension 1", ylabel="Dimension 2", 
        legend=false,
        markersize=2,  # 设置小点
        alpha=0.5)     # 设置透明度

    # 保存图像
    savefig("tsne_plot.png")
end
