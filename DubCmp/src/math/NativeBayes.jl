using CSV
using DataFrames
using Distributions  # 引入分布模块，用于高斯分布
using Statistics  # 引入 Statistics 模块

export NativeBayes

# 朴素贝叶斯模型
mutable struct NativeBayes
    """
        likelihoods: 每个特征对应的条件概率(Array{Array{Normal{Float64}}}) 
        classes: 类标签(Vector{String})
    """
    priors::Vector{Float64}      # 先验概率
    likelihoods::Array{Array{Normal{Float64}}}  # 每个特征对应的条件概率（每个类的高斯分布）
    classes::Vector{String}      # 类标签
    num_features::Int            # 特征数量
    class_column::Int            # 类标签所在列的索引
    
    # 从数据框初始化
    function NativeBayes(df::DataFrame, class_column::Int)
        classes = unique(df[:, class_column])  # 提取所有的类标签
        num_features = size(df, 2) - 1  # 特征的数量 (去除类标签列)
        
        priors = zeros(Float64, length(classes))  # 初始化先验概率
        likelihoods = [Array{Normal{Float64}}(undef, 0) for _ in 1:num_features]  # 初始化条件概率数组
        
        # 计算先验概率
        for (i, c) in enumerate(classes)
            priors[i] = sum(df[:, class_column] .== c) / length(df[:, class_column])
        end
        
        # 计算每个特征在每个类下的条件概率（使用高斯分布）
        for j in 1:num_features
            for (i, c) in enumerate(classes)
                feature_values = df[df[:, class_column] .== c, j]
                mean_val = mean(feature_values)  # 计算均值
                std_dev = std(feature_values)  # 计算标准差
                push!(likelihoods[j], Normal(mean_val, std_dev))  # 存储每个特征的高斯分布
            end
        end
        
        return new(priors, likelihoods, classes, num_features, class_column)
    end
    
    # 从给定的参数初始化模型
    function NativeBayes(priors::Vector{Float64}, likelihoods::Array{Array{Normal{Float64}}}, classes::Vector{String}, num_features::Int, class_column::Int)
        return new(priors, likelihoods, classes, num_features, class_column)
    end
end

# 预测函数
function predict(model::NativeBayes, samples::DataFrame)
    predictions = String[]  # 用于存储预测结果
    
    for row in eachrow(samples)  # 使用 eachrow 遍历每一行
        sample = row[:]  # 获取单个样本的特征
        
        # 计算每个类的后验概率
        posteriors = Float64[]  # 后验概率
        for (i, class) in enumerate(model.classes)
            posterior = model.priors[i]  # P(class)
            
            # 对每个特征计算 P(feature_j | class)，使用高斯分布的 PDF
            for j in 1:model.num_features
                likelihood = pdf(model.likelihoods[j][i], sample[j])  # 计算该特征在该类下的条件概率
                posterior *= likelihood
            end
            push!(posteriors, posterior)
        end
        
        # 找到后验概率最大的类
        predicted_class = model.classes[argmax(posteriors)[1]]
        push!(predictions, predicted_class)
    end
    
    return predictions
end
