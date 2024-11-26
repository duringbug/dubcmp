using LinearAlgebra
using StatsBase

export KDTree, knn_search

mutable struct KDTree{T<:Number}
    points::Matrix{T}   # 数据点
    labels::Vector{Int64} # 数据点的标签
    k::Int64            # 数据点的维度
    tree::Union{Nothing, Any} # 树的结构

    function KDTree{T}(points::Matrix{T}, labels::Vector{Int64}, k::Int64) where {T<:Number}
        @assert size(points, 1) == length(labels) "Number of points must match the number of labels"
        @assert size(points, 2) == k "The dimension of points must match the value of k"
        tree = build_kdtree(points, labels, k)
        return new{T}(points, labels, k, tree)
    end
end

# 构建 KDTree
function build_kdtree(points::Matrix{T}, labels::Vector{Int64}, k::Int64) where T
    return build_tree(points, labels, 1, k)
end

function build_tree(points::Matrix{T}, labels::Vector{Int64}, depth::Int64, k::Int64) where T
    n = size(points, 1)
    if n == 0
        return nothing
    end
    
    # 选择当前维度来划分
    axis = (depth - 1) % k + 1
    
    # 按照当前维度排序
    sorted_indices = sortperm(points[:, axis])  # 按照当前维度排序
    sorted_points = points[sorted_indices, :]
    sorted_labels = labels[sorted_indices]

    # 选择中位点作为当前节点
    median_idx = div(n - 1, 2) + 1  # 计算中位数索引，避免返回 0 索引
    median_point = sorted_points[median_idx, :]
    median_label = sorted_labels[median_idx]

    # 递归构建左右子树
    left_tree = n > 1 ? build_tree(sorted_points[1:median_idx-1, :], sorted_labels[1:median_idx-1], depth + 1, k) : nothing
    right_tree = n > 1 ? build_tree(sorted_points[median_idx+1:end, :], sorted_labels[median_idx+1:end], depth + 1, k) : nothing

    return (point=median_point, label=median_label, left=left_tree, right=right_tree)
end





# 计算欧几里得距离
function euclidean_distance(x1::Vector{T}, x2::Vector{T}) where T
    return sqrt(sum((x1 .- x2) .^ 2))
end

# 进行最近邻搜索
function knn_search(tree, point::Vector{T}, k::Int64, depth::Int64, k_dim::Int64) where T
    if tree === nothing
        return [], Inf
    end

    # 当前节点的点和标签
    node_point = tree.point
    node_label = tree.label

    # 计算当前点与树节点的距离
    dist = euclidean_distance(point, node_point)

    # 递归搜索左右子树
    axis = (depth - 1) % k_dim + 1
    next_branch = point[axis] < node_point[axis] ? tree.left : tree.right
    other_branch = point[axis] < node_point[axis] ? tree.right : tree.left

    # 递归搜索最近的分支
    nearest_points, nearest_distance = knn_search(next_branch, point, k, depth + 1, k_dim)

    # 将当前节点加入
    push!(nearest_points, (node_point, node_label, dist))
    nearest_points = sort!(nearest_points, by = x -> x[3])

    if length(nearest_points) > k
        pop!(nearest_points)
    end

    # 判断是否需要搜索另一分支
    if length(nearest_points) < k || abs(point[axis] - node_point[axis]) < nearest_distance
        other_points, other_distance = knn_search(other_branch, point, k, depth + 1, k_dim)
        nearest_points = vcat(nearest_points, other_points)
        nearest_points = sort!(nearest_points, by = x -> x[3])
        if length(nearest_points) > k
            pop!(nearest_points)
        end
    end

    return nearest_points, nearest_points[end][3]
end


# 预测单个样本
function predict_one(kdtree::KDTree{T}, x::SubArray{T, 1, Matrix{T}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}}, true}) where T
    # 检查 x 是否是 SubArray 类型，如果是，转化为 Vector{T}
    x_vector = typeof(x) <: SubArray ? Vector{T}(x) : x

    # 使用 KDTREE 进行最近邻搜索
    nearest_points, _ = knn_search(kdtree.tree, x_vector, kdtree.k, 1, kdtree.k)

    # 获取最近邻的标签
    nearest_labels = [point[2] for point in nearest_points]

    # 返回最常见的标签
    return mode(nearest_labels)
end


# 预测多个样本
function predict(kdtree::KDTree{T}, X_new::Matrix{T}) where T
    return predict_one.(Ref(kdtree), eachrow(X_new))  # 对每一行样本进行预测
end