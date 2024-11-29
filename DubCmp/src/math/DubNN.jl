module DubNN
    using DataFrames
    using Random
    using LinearAlgebra
    using Base.Iterators: partition
    using Base.Threads
    include("Core/Layer.jl")
    include("NN/ConvolutionalNetworks.jl")
    function splitDataSet(dataset::AbstractArray, label::AbstractVector, split_ratio, seed)
        Random.seed!(seed)
        num_train = size(dataset, 1)
        num_train_split = Int64(floor(split_ratio * num_train))
        indices = randperm(num_train)
        train_indices = indices[1:num_train_split]
        val_indices = indices[num_train_split+1:end]
        
        # 对于Array类型的切分
        train_images_split = dataset[train_indices, :, :, :]
        train_labels_split = label[train_indices]
        val_images_split = dataset[val_indices, :, :, :]
        val_labels_split = label[val_indices]
        
        return train_images_split, train_labels_split, val_images_split, val_labels_split
    end
    function train()
        
    end
end