import Pkg
# Pkg.add("Distributions")
# Pkg.add("StatsBase")
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("Plots")
# Pkg.add("PyPlot")
# Pkg.add("MLDatasets")

using Test
using LinearAlgebra
using Random  # 用于生成随机数
using Distributions
using StatsBase
using CSV
using DataFrames
using Plots
using PyPlot
# gr()
pyplot()
using MLDatasets

# test_dubcmp.jl
include("../DubCmp.jl")
using .DubCmp  # 引入模块
label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

function showImg(train_images_split, train_labels_split)
    idx = 2
    img = train_images_split[idx, :, :, :]  # 获取训练集的第一张图片，注意这是一个 32x32x3 的图像
    img = permutedims(img, (2, 3, 1))  # 转置维度顺序，使其符合 RGB 格式
    img = Float32.(img)  # 确保数据是 Float32 类型
    label = label_names[train_labels_split[idx] + 1]
    imshow(img)  # 使用 heatmap 显示图片
    PyPlot.text(0.5, 0.95, label, color="white", ha="center", fontsize=14, transform=gca().transAxes)
    show()
end

@testset "NN Tests" begin
    train_images, train_labels = CIFAR10(split=:train)[:]
    test_images, test_labels = CIFAR10(split=:test)[:]
    train_images = permutedims(train_images, (4, 3, 2, 1))
    test_images = permutedims(test_images, (4, 3, 2, 1))
    train_images_split, train_labels_split, val_images_split, val_labels_split = DubCmp.Train.DubNN.splitDataSet(train_images, train_labels, 0.8, 408)
    println("训练集图像尺寸: ", size(train_images_split))
    println("训练集标签尺寸: ", size(train_labels_split))
    println("验证集图像尺寸: ", size(val_images_split))
    println("验证集标签尺寸: ", size(val_labels_split))
    println("测试集图像尺寸: ", size(test_images))
    println("测试集标签尺寸: ", size(test_labels))
    conv = DubCmp.Train.DubNN.ConvolutionalNetworks(train_images, train_labels)
    # showImg(train_images_split, train_labels_split)

    conv_param = Dict("stride" => 1, "pad" => 1)
    F, C, WW, HH = 10, 3, 3, 3  # 假设有10个卷积核，大小为3x3，通道数为3（CIFAR-10是RGB图片）
    w = rand(Float32, F, C, WW, HH)  # 随机生成卷积核权重
    b = rand(Float32, F)  # 随机生成偏置
    out, cache_conv = DubCmp.Train.DubNN.conv_forward(train_images_split, w, b, conv_param)
    out, cache_relu = DubCmp.Train.DubNN.relu_forward(out)
    pool_param = Dict("pool_width" => 1, "pool_height" => 2, "stride" => 2)
    out, cache_pool = DubCmp.Train.DubNN.max_pool_forward(out, pool_param)
    println("卷积后的输出形状: ", size(out))
    @show size(out[2, 2, :, :])
    img = out[2, 2, :, :]
    img = Float32.(img)
    imshow(img, cmap="gray")
    axis("off")
    show()
end
