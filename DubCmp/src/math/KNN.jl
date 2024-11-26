using LinearAlgebra
using StatsBase
using Base.Threads

export KNN

mutable struct KNN{T<:Number}
    X::Matrix{T}
    y::Vector{Int64}
    k::Int64

    # 构造函数
    function KNN{T}(X::Matrix{T}, y::Vector{Int64}, k::Int64) where {T<:Number}
        @assert size(X, 1) == length(y) "Number of rows in X must match the length of y"
        return new{T}(X, y, k)
    end
end

# 计算欧几里得距离
function euclidean_distance(x1::SubArray{T}, x2::SubArray{T}) where {T<:Number}
    return sqrt(sum((x1 .- x2) .^ 2))
end

# 预测单个样本
function predict_one(knn::KNN{T}, x::SubArray{T, 1, Matrix{T}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}}, true}) where {T<:Number}
    distances = euclidean_distance.(eachrow(knn.X), Ref(x))  # 广播计算每一行与 x 的距离
    sorted_indices = sortperm(distances)[1:knn.k]           # 取最近的 k 个邻居索引
    nearest_labels = knn.y[sorted_indices]
    return mode(nearest_labels)  # 返回最常见的标签
end

# 预测多个样本
function predict(knn::KNN{T}, X_new::Matrix{T}) where {T<:Number}
    return predict_one.(Ref(knn), eachrow(X_new))  # 直接传递 SubArray
end
