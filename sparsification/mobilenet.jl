using Flux, Metalhead, Statistics
using Flux.Tracker: back!, update!, grad
using Flux: onehotbatch, onecold, crossentropy, throttle, loadparams!
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
using Images

alpha = 1.0
dropout = 0.001
depth_multiplier = 1
layers = []

#Convolution block

function conv_block(input, filters)
  filters = Int64(floor(filters * alpha))
  push!(layers, x -> padarray(x[:, :, :, :], Fill(0, (0, 0, 0, 0), (1, 1, 0, 0))))  #ZeroPadding2d equivalent
  push!(layers, Conv((3, 3), input => filters, pad=(1, 1), stride=(2, 2)))
  push!(layers, BatchNorm(filters))
  push!(layers, x -> map(a -> min(max(0, a), 6.0), Tracker.data(x)))    #ReLu 6.0
end

#Depthwise Convolution block

function depth_conv_block(input, pointwise_conv_filters; strides=(1, 1))
  pointwise_conv_filters = Int64(floor(pointwise_conv_filters * alpha))
  if strides != (1, 1)
    push!(layers, x -> padarray(x[:, :, :, :], Fill(0, (0 ,0, 0, 0), (1, 1, 0, 0))))  #ZeroPadding2d equivalent
  end
  new_input = Int64(input * depth_multiplier)
  push!(layers, DepthwiseConv((3, 3), input => depth_multiplier, pad=(1, 1), stride=strides))  #Depthwise Convolution
  push!(layers, BatchNorm(new_input))
  push!(layers, x -> map(a -> min(max(0, a), 6.0), Tracker.data(x)))
  push!(layers, Conv((1, 1), new_input => pointwise_conv_filters, pad=(1, 1), stride=(1, 1)))
  push!(layers, BatchNorm(pointwise_conv_filters))
  push!(layers, x -> map(a -> min(max(0, a), 6.0), Tracker.data(x)))
end

# Adding MobileNet Layers

  conv_block(3, Int64(floor(32*alpha))),
  depth_conv_block(Int64(floor(32*alpha)), 64),
  depth_conv_block(Int64(floor(64*alpha)), 128, strides=(2, 2)),
  depth_conv_block(Int64(floor(128*alpha)), 128),
  depth_conv_block(Int64(floor(128*alpha)), 256, strides=(2, 2)),
  depth_conv_block(Int64(floor(256*alpha)), 256),
  depth_conv_block(Int64(floor(256*alpha)), 512, strides=(2, 2)),
  depth_conv_block(Int64(floor(512*alpha)), 512),
  depth_conv_block(Int64(floor(512*alpha)), 512),
  depth_conv_block(Int64(floor(512*alpha)), 512),
  depth_conv_block(Int64(floor(512*alpha)), 512),
  depth_conv_block(Int64(floor(512*alpha)), 512),
  depth_conv_block(Int64(floor(512*alpha)), 1024, strides=(2, 2)),
  depth_conv_block(Int64(floor(1024*alpha)), 1024),
  push!(layers, x -> mean(x, dims=[1, 2]))
"  Dropout(dropout),
  #Reshape2d
  Conv((1, 1), `Reshapeout` => 1000, pad=(1, 1), stride=(1, 1)),
  x -> reshape(x, (1000,)),
  softmax
"
# Building MobileNet

model(x) = foldl((x, m) -> m(x), layers, init=x)

# Function to convert the RGB image to Float64 Arrays
getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))

# Fetching the train and validation data and getting them into proper shape
X = trainimgs(CIFAR10)
imgs = [getarray(X[i].img) for i in 1:50000] 
labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)
train = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:49000, 100)]

model(train[1][1])
"valset = collect(49001:50000)
valX = cat(imgs[valset]..., dims = 4) |> gpu
valY = labels[:, valset] |> gpu

loss(x, y) = crossentropy(model(x), y)

accuracy(x, y) = mean(onecold(model(x), 1:10) .== onecold(y, 1:10))

opt = Descent(0.1)

# Starting to train models
for epoch = 1:50
  for d in train
    l = gpu(loss(d...))
    back!(l)
    for p in params(model)
      update!(opt, p, grad(p))
    end
    @show(accuracy(valX,valY))
  end
end"