using LinearAlgebra
using Random
export  PLA, train

# 进度条模块
mutable struct ProgressBar
    max_val::Int
    current_val::Int
    width::Int
end

function ProgressBar(max_val::Int, width::Int=50)
    return ProgressBar(max_val, 0, width)
end

function next!(pbar::ProgressBar)
    pbar.current_val += 1
    print_progress(pbar)
end

function print_progress(pbar::ProgressBar)
    # 计算百分比
    percent = pbar.current_val / pbar.max_val
    # 确保百分比不会小于 0
    percent = max(0, min(1, percent))  # 确保百分比在 0 到 1 之间

    # 计算进度条的显示字符数
    bar = "█" ^ round(Int, pbar.width * percent) * " " ^ round(Int, pbar.width * (1 - percent))
    # 输出进度条
    print("\r|$bar| $(round(percent * 100, digits=1))%")
    flush(stdout)
end


mutable struct PLA{T<:Number}
    W::Matrix{T}
    b::T

    # 构造函数
    function PLA{T}(input_dim::Int) where {T<:Number}
        W = fill(T(100), 1, input_dim)  # 动态设置 W 的维度，根据输入数据的维度初始化
        b = zero(T)                     # T 类型的零值
        return new{T}(W, b)
    end
end

# train 方法
function train(self::PLA{T}, X::Matrix{T}, Y::Matrix{T}; max_iter::Int=200, learning_rate::T=1.0, bar::Bool=true) where {T<:Number}
    # X: 输入特征矩阵，Y: 标签矩阵
    # max_iter: 最大迭代次数
    # learning_rate: 学习率

    # 获取样本数量
    n_samples = size(X, 2)

    pbar = ProgressBar(max_iter)

    # 训练过程的循环
    for epoch in 1:max_iter
        # 假设整个数据集一次性通过
        total_loss = 0.0
        for i in 1:n_samples
            # 获取当前样本
            x_i = X[:, i]
            y_i = Y[1, i]  # 获取该样本对应的标签

            # 计算预测值
            y_pred = self.W * x_i .+ self.b
            y_pred = y_pred[1, 1]

            # 计算当前样本的损失
            loss = (y_pred - y_i)^2
            total_loss += loss

            # 误分类检查：如果预测值和标签不同
            if (y_pred > 0.0) != (y_i > 0.0)
                # 更新权重和偏置
                self.W .= self.W .+ learning_rate * y_i * x_i'
                self.b += learning_rate * y_i
            end           
        end

        # 计算并输出每个 epoch 的平均损失
        avg_loss = total_loss / n_samples
        next!(pbar)
        if bar
            println("Epoch $epoch, Loss: $avg_loss")
        end
    end
    println()
    return self
end

function predict(self::PLA{T}, X_test)where {T<:Number}
    return self.W * X_test .+ self.b
end