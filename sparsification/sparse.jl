using Flux, Metalhead, Statistics
using Flux.Tracker: back!, update!, grad, extract_grad!, data
using Flux: onehotbatch, onecold, crossentropy, throttle, loadparams!
using Metalhead: trainimgs
using Images: channelview
using CuArrays
using BSON: @save, @load
using Statistics: mean
using Base.Iterators: partition

# VGG16 and VGG19 models

vgg16() = Chain(
  Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(64),
  Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(64),
  MaxPool((2, 2)),
  Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128),
  Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(128),
  MaxPool((2,2)),
  Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(256),
  MaxPool((2, 2)),
  Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  MaxPool((2, 2)),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
  BatchNorm(512),
  MaxPool((2, 2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(512, 4096, relu),
  Dropout(0.5),
  Dense(4096, 4096, relu),
  Dropout(0.5),
  Dense(4096, 10),
  softmax )

getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))

X = trainimgs(CIFAR10)
imgs = [getarray(X[i].img) for i in 1:50000] 
labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)
train = [(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:49000, 100)]

loss(x, y) = crossentropy(model(x), y)

m = vgg16()
lambda = 1.0
opt = Descent(0.1)

#Unspecific Bounded Insensitivity
function unspecific(d, lambda)
    output = m(d[1])
    back!(output[1], once=false)
    gs = data.(extract_grad!.(params(m)))
    gs = map(x -> abs.(x), gs)
    for o = 2:length(output)
        back!(output[o], once=false)
        grads = data.(extract_grad!.(params(m)))
        grads = map(x -> abs.(x), grads)
        gs = gs .+ grads
    end

    l = loss(d...)
    back!(l)

    gs = 1/length(output) .* gs         #Sensitivity
    gs = max(0, 1 .- gs)                #Unspecific bounded Insensitivity
    gs = gs .* params(m)
    gs = gs .* lambda

    for p in params(m)
        Δ = -apply!(opt, p, data(grad(p)))
        update!(p, Δ)                        #Update parameters
        update!(p, grads[p])                 #Adding extra insensitivity factor
    end
end

#Specific Bounded Insensitivity
function specific(d, lambda)
    output = m(d[1])
    reqd_index = 0
    for o = 1:length(output)
        if d[2][o]:
            reqd_index = o
            break
        end
    end
    back!(output[reqd_index], once=false)
    gs = data.(extract_grad!.(params(m)))
    gs = map(x -> abs.(x), gs)
    for o = reqd_index:length(output)
        if d[2][o]
            back!(output[o], once=false)
            grads = data.(extract_grad!.(params(m)))
            grads = map(x -> abs.(x), grads)
            gs = gs .+ grads
        end
    end    

    l = loss(d...)
    back!(l)

    gs = max(0, 1 .- gs)                  #Specific bounded Insensitivity
    gs = gs .* params(m)
    gs = gs .* lambda

    for p in params(m)
        Δ = -apply!(opt, p, data(grad(p)))
        update!(p, Δ)                            #Update parameters
        update!(p, grads[p])                     #Adding extra insensitivity factor
    end
end


```for d in train
    unspecific(d, lambda)
    #specific(d, lambda)
    #Thresholding value to actual zero
end```