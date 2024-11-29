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

function conv_forward(x, w, b, conv_param)
  N, C, W, H = size(x)
  F, C, WW, HH = size(w)
  stride = conv_param["stride"]
  pad = conv_param["pad"]
  W_out = 1 + (H + 2 * pad - WW) ÷ stride
  H_out = 1 + (W + 2 * pad - HH) ÷ stride
  x_padded = zeros(N, C, W + 2 * pad, H + 2 * pad)
  x_padded[:, :, pad+1:end-pad, pad+1:end-pad] .= x
  out = zeros(N, F, W_out, H_out)
  x_cols = zeros(WW * HH * C, W_out * H_out * N)
  w_cols = reshape(w, F, C * WW * HH)
  idx = 1
  pbar = ProgressBar(N)
  for n in 1:N
      for i in 1:W_out
          for j in 1:H_out
              w_start = (i - 1) * stride + 1
              h_start = (j - 1) * stride + 1
              x_slice = x_padded[n, :, w_start:w_start + WW - 1, h_start:h_start + HH - 1]
              x_cols[:, idx] .= reshape(x_slice, WW * HH * C)
              out[n, :, i, j] .= w_cols * x_cols[:, idx] .+ b
              idx += 1
          end
      end
      next!(pbar)
  end
  println()
  cache = (x, w, b, conv_param)
  return out, cache
end

function conv_backward(dout, cache)
  x, w, b, conv_param = cache
  N, C, W, H = size(x)
  F, C, WW, HH = size(w)
  stride = conv_param["stride"]
  pad = conv_param["pad"]
  W_out = size(dout, 3)
  H_out = size(dout, 4)
  x_padded = zeros(N, C, W + 2 * pad, H + 2 * pad)
  x_padded[:, :, pad+1:end-pad, pad+1:end-pad] .= x
  dx_padded = zeros(N, C, W + 2 * pad, H + 2 * pad)
  
  dw = zeros(F, C, WW, HH)
  db = zeros(F)
  for n in 1:N  # 对每个样本进行遍历
      for f in 1:F  # 对每个卷积核进行遍历
          for i in 1:W_out  # 输出宽度遍历
              for j in 1:H_out  # 输出高度遍历
                  # 当前输出位置对应的梯度
                  dout_slice = dout[n, f, i, j]
                  
                  # 卷积窗口的起始和结束位置
                  w_start = (i - 1) * stride + 1
                  h_start = (j - 1) * stride + 1
                  x_slice = x_padded[n, :, w_start:w_start + WW - 1, h_start:h_start + HH - 1]
                  
                  # 计算卷积核和输入数据窗口的梯度
                  dw[f, :, :, :] .+= dout_slice * x_slice
                  dx_padded[n, :, w_start:w_start + WW - 1, h_start:h_start + HH - 1] .+= dout_slice * w[f, :, :, :]
              end
          end
          # 偏置的梯度是 dout 在输出维度上的求和
          db[f] .+= sum(dout[n, f, :, :])
      end
  end
  
  # 将填充部分的梯度去除，得到 dx
  dx = dx_padded[:, :, pad+1:end-pad, pad+1:end-pad]  # 去除填充部分
  
  return dx, dw, db
end

function max_pool_forward(x, pool_param)
      # 从池化参数中提取池化的高度、宽度和步幅
      pool_height = pool_param["pool_height"]
      pool_width = pool_param["pool_width"]
      stride = pool_param["stride"]
  
      # 输入数据的形状
      N, C, W, H = size(x)  # 注意：这里W在第三个维度，H在第四个维度
  
      # 计算输出的高度和宽度
      W_out = 1 + (W - pool_width) ÷ stride
      H_out = 1 + (H - pool_height) ÷ stride
  
      # 初始化输出矩阵
      out = zeros(N, C, W_out, H_out)
  
      # 进行最大池化操作
      pbar = ProgressBar(N)
      for n in 1:N  # 遍历每个样本
          for c in 1:C  # 遍历每个通道
              for i in 1:W_out  # 遍历输出的宽度
                  for j in 1:H_out  # 遍历输出的高度
                      # 当前池化区域的起始和结束位置
                      w_start = (i - 1) * stride + 1
                      w_end = w_start + pool_width - 1
                      h_start = (j - 1) * stride + 1
                      h_end = h_start + pool_height - 1
  
                      # 在池化区域内取最大值
                      out[n, c, i, j] = maximum(x[n, c, w_start:w_end, h_start:h_end])
                  end
              end
          end
          next!(pbar)
      end
      println()
  
      # 返回输出和缓存
      cache = (x, pool_param)
      return out, cache
end

function max_pool_backward(dout, cache)
  # 从缓存中提取输入数据和池化参数
  x, pool_param = cache
  pool_height = pool_param["pool_height"]
  pool_width = pool_param["pool_width"]
  stride = pool_param["stride"]
  
  # 输入数据的形状
  N, C, W, H = size(x)
  
  # 输出的梯度形状与输入数据相同
  dx = zeros(N, C, W, H)
  
  # 计算输出的高度和宽度
  W_out = size(dout, 3)
  H_out = size(dout, 4)
  
  # 进行最大池化的反向传播
  pbar = ProgressBar(N)
  for n in 1:N  # 遍历每个样本
      for c in 1:C  # 遍历每个通道
          for i in 1:W_out  # 遍历输出的宽度
              for j in 1:H_out  # 遍历输出的高度
                  # 当前池化区域的起始和结束位置
                  w_start = (i - 1) * stride + 1
                  w_end = w_start + pool_width - 1
                  h_start = (j - 1) * stride + 1
                  h_end = h_start + pool_height - 1
                  
                  # 提取池化区域
                  x_slice = x[n, c, w_start:w_end, h_start:h_end]
                  
                  # 找到池化区域的最大值的索引
                  max_index = Argmax(x_slice)  # 获取最大值的索引
                  
                  # 将dout的梯度传递给最大值所在的位置
                  dx[n, c, w_start + max_index[1], h_start + max_index[2]] += dout[n, c, i, j]
              end
          end
      end
      next!(pbar)
  end
  
  return dx
end

function relu_forward(x)
  N = size(x, 1)  # 假设 N 是第一个维度 (batch size)，你可以根据实际情况调整
  out = similar(x)  # 创建与 x 相同大小的输出数组
  pbar = ProgressBar(N)  # 创建一个进度条，N 是迭代次数
  for i in 1:N
      out[i, :, :, :] = max.(0, x[i, :, :, :])  # 对每个样本进行 ReLU 操作
      next!(pbar)
  end
  println()
  
  cache = x  # 保存输入 x，用于反向传播
  return out, cache
end

function relu_backward(dout, cache)
    x = cache  # 获取缓存中的输入 x
    dx = similar(x)  # 创建与输入相同形状的 dx
    
    # 计算梯度：只有当 x > 0 时，dout 才会传递给 dx，否则 dx 为 0
    dx .= (x .> 0) .* dout  # 使用广播操作，计算 ReLU 的梯度
    
    return dx
end



