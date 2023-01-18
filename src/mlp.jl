module MLP

using Flux
using Flux: onehot, onehotbatch, Embedding

# Define the data

# words = readlines("chinese_names.txt")
words = readlines("names.txt")
words = words[2:end]
chars = Set(join(words))
chars = chars |> collect
# chars = 'a':'z' |> collect
chars = insert!(chars, 1, '.')
stoi = Dict(c => i for (i, c) in enumerate(chars))
itos = Dict(i => c for (i, c) in enumerate(chars))

block_size = 2

wrap_word(w) = "." ^ block_size * w * "." |> collect
ngrams(w) = [(w[i:i+block_size-1], w[i+block_size]) for i in 1:length(w)-block_size]
wrap_word(words[1]) |> ngrams


function ngram_data(words, block_size)
    all_ngrams = words .|> wrap_word .|> ngrams
    all_ngrams = vcat(all_ngrams...)
    X = all_ngrams .|> first 
    X = hcat(X...)
    Y = last.(all_ngrams)
    return X, Y
end

using MLUtils

X, Y = ngram_data(words, block_size)

X, Y = shuffleobs((X, Y))
train_data, valid_data, test_data = splitobs((X, Y), at = (0.8, 0.1))


# X
# Y
# # Define the model


# # using Flux: Embedding, onehotbatch, onehot
# # emb = Embedding(26, 2)
# # x1 = rand('a':'z', 10)
# # x1 = onehotbatch(x1, 'a':'z')
# # emb(x1)
# x2 = rand('a':'z', (3, 10))
# x2 = onehotbatch(x2, 'a':'z')
# # crossentropy(x2, x2)
# # works with arbitary shaped tensors - any batch dimensions are possible
# # emb(x2)


# emb = randn(2, length(chars))
# # emb = Embedding(27, 2)
# # emb(onehotbatch(Y, chars))
# # q = onehotbatch(Y, chars)
# # isa(q, AbstractMatrix{Bool})
# # k = onehotbatch(X, chars)
# # k2 = reshape(k, size(k, 1), :)
# # emb(k2)
# # isa(k, AbstractVector)
# # # emb(k) # broken report issue in Flux
# # emb(rand(1:26, (10, 1, 12)))
# emb * onehotbatch(Y, chars)
# # onehotbatch(X, chars)
# # emb
# emb * onehotbatch(X, chars)

# emb[:, [1,2,5,5]] # indexing particular cols
# emb[[1,1], [1,2,4,4]]
# # cartisian indexing is like gather?
# X_enc = onehotbatch(X, chars)
# indexing_t = rand(1:26, (2, 3, 4))
# Z = emb[[1,1, 1], indexing_t] # awesome it works - very complex indexing
# # result above is of shape size(first_axis_t) x size(second_axis_t)
# # I still don't have a strong model of indexing and how the tensor is actually stored
# X
# X2 = X .|> (x -> stoi[x])
# x_embedded = emb[:, X2]

# W1 = rand(100, 2*3) # verified with how dense layer arranges its weights
# b1 = rand(100)

# x_embedded
# x_embedded_2, old_size = reshape(x_embedded, 6, :), size(x_embedded)
# # x_embedded_2[1, 1] = 0.0
# x_embedded_2
# x_embedded_2
# x_embedded[:, 1, :]
# x_embedded_3 = cat(x_embedded[:, 1, :], x_embedded[:, 2, :], x_embedded[:, 3, :]; dims=1)
# x_embedded_4 = cat(eachslice(x_embedded, dims=2)...; dims=1)
# # unbind is like eachslice but inefficient as new tensor is created
# # know that reshape works like a reverse loop
# # isequal as alternative to == for missing nan values
# pointer(x_embedded) == pointer(x_embedded_2)
# # check underlying memeory location
# eachindex(x_embedded) == eachindex(x_embedded_2)
# # element wise comparison. not trivial as ==
# reshape(x_embedded, 6, 32) .== x_embedded_3
# reshape(x_embedded, 6, 32) == x_embedded_2

# # reshape in julia does not create new array so it is same as pytorch view
# h = tanh.(W1 * x_embedded_2 .+ b1)
# # in julia broadcasting starts with matching dimensions to the left
# W2 = randn(27, 100)
# b2 = randn(27)

# y = W2 * h .+ b2
# # subracting logits by a fixed constant doesn't change result of softmax
# counts = exp.(y) # risky if one of your logits is large you will get a nan
# counts ./= sum(counts, dims=1)

# counts2 = softmax(y)

# ŷ = Y .|> (y -> stoi[y])

# Y

# relevant_probs = counts[[CartesianIndex(l, i) for (i, l) in enumerate(ŷ)]]

using Statistics

# loss = -mean(log.(relevant_probs))

using Flux.Losses: logitcrossentropy, crossentropy

# logitcrossentropy(y, onehotbatch(Y, chars))
# crossentropy(counts, onehotbatch(Y, chars))
# forward pass more efficent due to fused kernel
# backward more efficient because of math trick
# stable due to subraction before softmax

function make_model(emb_size=2, hidden_dim=100, vocab_size=27, block_size=3)
    emb = randn(emb_size, vocab_size)
    W1 = randn(hidden_dim, emb_size*block_size)
    b1 = randn(hidden_dim)
    W2 = randn(vocab_size, hidden_dim)
    b2 = randn(vocab_size)

    function model(X)
        x_embedded = emb(onehotbatch(X, chars))
        x_embedded = reshape(x_embedded, emb_size*block_size, :)
        h = tanh.(W1 * x_embedded .+ b1)
        y = W2 * h .+ b2
    end

    return model
end



struct Model
    emb
    W1
    b1
    W2
    b2
end

Flux.@functor Model

function Model(;emb_size=2, hidden_dim=100, vocab_size=27, block_size=3)
    emb = Embedding(vocab_size, emb_size)
    W1 = randn(hidden_dim, emb_size*block_size)
    b1 = randn(hidden_dim)
    W2 = randn(vocab_size, hidden_dim)
    b2 = randn(vocab_size)

    return Model(emb, W1, b1, W2, b2)
end

function (m::Model)(X_onehot)
    x_embedded = m.emb(X_onehot)
    hidden_dim_input = size(m.W1, 2)
    x_embedded = reshape(x_embedded, hidden_dim_input, :)
    h = tanh.(m.W1 * x_embedded .+ m.b1)
    y = m.W2 * h .+ m.b2
end


    
lossfn(y, Y) = logitcrossentropy(y, onehotbatch(Y, chars))
lossfng(y, Y_onehot) = logitcrossentropy(y, Y_onehot)
# function loss(y, Y)
#     ŷ = onehotbatch(Y, chars)
#     logitcrossentropy(y, ŷ)
# end

using ProgressMeter

# model = Model(hidden_dim=100, emb_size=30, vocab_size=length(chars), block_size=2)
# opt = Flux.Optimise.Descent(0.2)
# opt_state = Flux.setup(opt, model)
# opt_state

using Optimisers
# opt_state
# typeof(model)

using Functors

function train()
    η = 0.01
    @showprogress for i in 1:100000
        # old_grads = nothing     #
        bs = 32
        ix = rand(1:size(X, 2), bs)
        X_batch = X[:, ix]
        Y_batch = Y[ix]
        l, grads = Flux.withgradient(m -> lossfng(m(X_batch), Y_batch), model)
        # if isnothing(old_grads)
        #     old_grads = grads
        # else
        #     old_grads = Functors.map(old_grads, grads) do x, y
        #         x .+ y
        #     end
        # end
        # accumlation example
        # grads2 = grads[1]
        # fmap(grads2, grads[1]) do p, g
        #     p .+= g
        # end
        Flux.update!(opt_state, model, grads[1])
        # fmap(model, grads[1]) do p, g
        #     p .= p .- η .* g
        # end       
    end
    println(lossfn(model(X), Y))
end

using CUDA


function train2(model)
    X_train, Y_train = train_data
    X_train
    Y_train
    data = Flux.Data.DataLoader((X_train, Y_train), batchsize=32, shuffle=true)
    # is equal to taking batch, forward pass, backward pass and optim update
    X_val, Y_val = valid_data
    # X_val = X_val |> gpu
    # Y_val = Y_val |> gpu
    lr = 0.2
    x = X_train |> gpu
    y = Y_train |> gpu
    m = model |> gpu
    onehotbatch(y, chars) |> summary 
    logitcrossentropy(m(x), onehotbatch(y, chars))
    function valid_loss()
        model = model |> cpu
        l = lossfn(model(X_val), Y_val)
        println(l)
        return l
    end

    println("loss before training $(valid_loss())")
    should_reduce_lr = Flux.early_stopping(valid_loss, 5, min_dist=0.05)


    
    @showprogress for epoch in 1:10
        model = model |> gpu
        Flux.train!(model, data, opt_state) do m, x, y
            lossfng(m(x |> gpu), y)
        end
        X_val, Y_val = valid_data


        
        # Figure out how to drop LR at Plateu 
        if should_reduce_lr() && false
            should_reduce_lr = Flux.early_stopping(valid_loss, 5, min_dist=0.05)
            lr = lr * 0.5
            println("Reducing LR to $(lr)")
            Optimisers.adjust!(opt_state, lr)
        end
    end
end


function train3(model, train_data, valid_data, opt_state, epochs=10, lr=0.2)
    X_val, Y_val = valid_data
    X_val, Y_val = onehotbatch(X_val, chars), onehotbatch(Y_val, chars)

    function valid_loss()
        println("Validating")
        model = model |> cpu
        l = lossfng(model(X_val), Y_val)
        println("Validation loss: $(l)")
        return l
    end

    println("loss before training $(valid_loss())")
    should_reduce_lr = Flux.early_stopping(valid_loss, 5, min_dist=0.05)


    data = Flux.Data.DataLoader(train_data, batchsize=64, shuffle=true)
    for epoch in 1:epochs
        println("Epoch $epoch")
        opt_state, model = single_epoch(model, data, opt_state)
        # println("Last Train Batch Loss: $l")
        # Figure out how to drop LR at Plateu 
        if should_reduce_lr() && false
            should_reduce_lr = Flux.early_stopping(valid_loss, 5, min_dist=0.05)
            lr = lr * 0.5
            println("Reducing LR to $(lr)")
            Optimisers.adjust!(opt_state, lr)
        end
    end
    return model, opt_state
end
    

using CUDA

model = Model(hidden_dim=100, emb_size=10, vocab_size=length(chars), block_size=2) |> gpu
opt = Flux.Optimise.Descent(0.01)
opt_state = Flux.setup(opt, model)

# model, opt_state = train3(model, train_data, valid_data, opt_state)


# for d in data
#     println(d)
# end
data = Flux.Data.DataLoader(train_data, batchsize=64, shuffle=true)

x, y = first(data)


function single_epoch(model, data, opt_state)
    m = model |> gpu
    X_train, Y_train = train_data
    # X_train |> summary
    # Y_train |> summary
    # X_train .|> (x -> stoi[x]) |> float |> gpu
    println("Size of data is $(length(data))")

    # @showprogress for (x, y) in data
    #     x = x .|> (c -> stoi[c]) |> gpu
    #     y = onehotbatch(y, chars)  |> gpu
    #     l, grads = Flux.withgradient(m -> lossfng(m(x), y), m)
    #     Flux.update!(opt_state, m, grads[1] |> gpu)
    #     lossfng(m(x), y)
    #     return opt_state, m , l
    # end

    @showprogress for i in 1:10000
        ix = rand(1:size(X_train, 2), 32)
        x, y = X_train[:, ix], Y_train[ix]
        x = x .|> (c -> stoi[c]) |> gpu
        y = onehotbatch(y, chars)  |> gpu
        l, grads = Flux.withgradient(m -> lossfng(m(x), y), m)
        # grads[1] |> summary
        # yes they are on cuda
        opt_state, m = Flux.update!(opt_state, m, grads[1])
    end
    return opt_state, m 
end




using StatsBase



# Sample a word using trained model

function sample_word(max_len=20, block_size=2)
    word = repeat(".", block_size) |> collect
    for i in 1:max_len
        y = model(word[end-2:end])
        y = softmax(y)
        new_char = sample(chars, Weights(y[:, 1]))
        if new_char == '.'
            return word[block_size+1:end] |> join
        end
        push!(word, new_char)
    end
    return join(prefix)
end

# sampled_words = [sample_word() for i = 1:10]
train2
# train2(model)
# train()

using Plots

# Plot a  2D scatter plot of the embedding

# learned_embedding = model.emb.weight

# learned_embedding

# scatter(learned_embedding[1, :], learned_embedding[2, :])
# annotate!.(learned_embedding[1, :], learned_embedding[2, :], text.(chars, 7, :center, :white, :sans))

# gui()


end
