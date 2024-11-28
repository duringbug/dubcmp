using Random
using LinearAlgebra

mutable struct TSNE
    perplexity::Float32
    n_iter::Int
    eta::Float32
    tol::Float32
    sigma::Float32
    Y::Array{Float32,2}  # 低维空间的嵌入
    P::Array{Float32,2}  # 高维空间的概率分布
    data::Array{Float32,2}  # 高维数据

    function TSNE(data::Array{Float32, 2}; perplexity=30.0, n_iter=1000, eta=200, tol=1e-5, sigma = 1.0)
        # 初始化 t-SNE 参数
        new(perplexity, n_iter, eta, tol, sigma, randn(size(data, 1), 2), zeros(size(data, 1), size(data, 1)), data)
    end
end


# 计算每对点的联合概率分布
function compute_joint_probabilities(tsne::TSNE)
    n = size(tsne.data, 1)
    for i in 1:n
        dists = sum((tsne.data .- tsne.data[i, :]').^2, dims=2)
        tsne.P[i, :] = exp.(-dists / (2 * tsne.sigma^2))
        tsne.P[i, i] = 0.0
        tsne.P[i, :] /= sum(tsne.P[i, :])
    end
end

# 使用梯度下降优化低维嵌入
function compute_low_dimensional_representation(tsne::TSNE)

    pbar = ProgressBar(tsne.n_iter)

    n = size(tsne.P, 1)
    for iter in 1:tsne.n_iter
        # 初始化一个 n × n 的矩阵来存储每对样本之间的距离
        dists = zeros(Float32, n, n)

        # 计算低维空间的欧几里得距离（逐元素计算每对点之间的差异）
        Threads.@threads for i in 1:n
            for j in 1:n
                @inbounds dists[i, j] = sum((tsne.Y[i, :] .- tsne.Y[j, :]).^2)
            end
        end
        # 取欧几里得距离
        dists = sqrt.(dists)

        # 计算Q（基于t-SNE的公式）
        Q = 1.0 ./ (1.0 .+ dists)
        Q[tril(ones(n, n), -1).==1] .= 0  # 对角线不需要
        Q /= sum(Q)

        # 计算梯度
        grad = zeros(Float32, n, 2)  # 这里存储每个点的梯度
        Threads.@threads for i in 1:n
            for j in 1:n
                # 计算每个点的梯度
                @inbounds grad[i, :] .+= -4 * (tsne.Y[i, :] .- tsne.Y[j, :]) * (tsne.P[i, j] - Q[i, j]) * dists[i, j]
            end
        end

        # 更新 Y
        tsne.Y += tsne.eta * grad
        norm_grad = norm(grad)
        if iter % 10 == 0
            @show norm_grad
        end
        if norm_grad < tsne.tol
            println("Converged at iteration $iter")
            break
        end
        next!(pbar)
    end
end


# t-SNE 主函数
function fit!(tsne::TSNE)
    compute_joint_probabilities(tsne)
    compute_low_dimensional_representation(tsne)
end
