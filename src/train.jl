module Train


include("model.jl")


using Flux
using Flux.Optimise
using Flux.Losses: logitcrossentropy
using Flux: onehotbatch, @functor

using StatsBase
using MLUtils
using Random
using Fire

using ProgressMeter
using BSON: @save

using BFloat16s

using CUDA

using Lazy

mutable struct ScaleValue{T} <: Flux.Optimise.AbstractOptimiser
    ratio::T
end

apply!(o::ScaleValue, x, Δ) = Δ .* o.ratio

using BenchmarkTools


struct Packed
    features
    mask
end


# Base.:+(a::Packed, b::Packed) = Packed(a.features + b.features, a.mask)
# Broadcast.broadcastable(f, x::Packed, y::Packed) = Packed(f.(x.features, y.features), x.mask)

# (mha::Model.MHSelfAttention)(x::Packed) = Packed(mha(x.features, x.mask), x.mask)
# (m::Dense)(x::Packed) = Packed(m(x.features), x.mask)
# (m::LayerNorm)(x::Packed) = Packed(m(x.features), x.mask)
# (m::Dropout)(x::Packed) = Packed(m(x.features), x.mask)
# (m::Embedding)(x::Packed) = Packed(m(x.features), x.mask)

# Base.length(x::Packed) = length(x.features)


# Non-CLI arguments
Random.seed!(1337)
device = gpu

# Data


text = read("input.txt", String)
chars = Set(text) |> collect |> sort

#higher order function here would be good I think
stoi = Dict(c => i for (i, c) in enumerate(chars))
itos = Dict(i => c for (i, c) in enumerate(chars))
encode(text) = [stoi[c] for c in text]
encode_char(c) = stoi[c]
decode(code) = [itos[i] for i in code] |> join
    


function prepare_data()
    
    text = read("input.txt", String)
    chars = Set(text) |> collect |> sort
    
    #higher order function here would be good I think
    stoi = Dict(c => i for (i, c) in enumerate(chars))
    itos = Dict(i => c for (i, c) in enumerate(chars))
    encode(text) = [stoi[c] for c in text]
    encode_char(c) = stoi[c]
    decode(code) = [itos[i] for i in code] |> join
    
    data = encode(text)

    return data, chars, encode, decode
end


loss(model, xs, ys_oh) = logitcrossentropy(model(xs), ys_oh)

    
function generate(model, start_tokens, max_len, block_size=512)

    # start_tokens= 
    # start_tokens = unsqueeze(start_tokens, 2)
    sample_helper(p) = sample(1:length(chars), Weights(p))
    tokens = start_tokens
    for i in 1:max_len
        logits = model(tokens[end-(block_size-1):end, :] |> device) # this device thing can also be moved inside custom model itself?
        last_time_step = logits[:, end, :]
        probs = softmax(last_time_step)
        probs = probs |> cpu
        next_token = sample_helper.(eachcol(probs)) # sampling does not work on gpu
        tokens = cat(tokens, next_token', dims=1)
    end
    return tokens[block_size:end]
end



function get_batch(data, batch_size, block_size, mask)
    start_offsets = rand(1:length(data)-block_size, batch_size)
    xs = hcat([data[start:start+block_size-1] for start in start_offsets]...)
    ys = hcat([data[start+1:start+block_size] for start in start_offsets]...)
    ys = onehotbatch(ys, 1:length(chars))
    xs, ys = xs |> device, ys |> device
    return (xs, mask), ys
end

    
function train_step!(model, xs, ys, opt_state)
    # println("forward pass")
    # @time model(xs)
    # println("loss")
    # @time l = loss(model, xs, ys)
    # println("loss direct")
    # @time logitcrossentropy(model(xs), ys)
    # println("backward pass")
    # @time Flux.withgradient(m -> loss(m, xs, ys), model)
    l, gs = Flux.withgradient(m -> loss(m, xs, ys), model)
    opt_state, model = Flux.update!(opt_state, model, gs[1]) # can consider scaling the gradient here
    return model, opt_state, l
end

function train(model, train_data, valid_data, opt_state, epochs, block_size, batch_size; train_iters=500, valid_iters=100)

    mask = Model.make_decoder_mask(block_size) |> device
    
    function estimate_loss()
        # we don't need to worry about modes here becaue we are not calculating gradients
        # when we calculate gradients flux will automatically switch to train mode
        valid_loss = 0
        train_loss = 0
        for i in 1:valid_iters
            xs, ys = get_batch(valid_data, batch_size, block_size, mask)
            # outs = model(xs)
            # logitcrossentropy(outs, ys)
            valid_loss += loss(model, xs, ys)
            xs, ys = get_batch(train_data, batch_size, block_size, mask)
            train_loss += loss(model, xs, ys)
        end
        valid_loss, train_loss = valid_loss/100, train_loss/100
        return valid_loss |> cpu, train_loss |> cpu
        
    end

    # my_xs, my_ys = get_batch(train_data, batch_size, block_size)

    # make_dataset() = [get_batch(train_data, batch_size, block_size) for i in 1:train_iters]

    
    for epoch in 1:epochs
        # @showprogress for i in 1:train_iters
        # dataset = make_dataset()

        # Doing so that julia compliles the code first and we can compare the report
        # xs, ys = get_batch(train_data, batch_size, block_size)
        # # xs, ys = my_xs, my_ys
        # # t = @async get_batch(train_data, batch_size, block_size)
        # model, opt_state, l = train_step!(model, xs, ys, opt_state)

        
        
        progress = Progress(train_iters, showspeed=true)
        #            @showprogress for i in 1:train_iters

        for i in 1:train_iters
            # xs, ys = fetch(t)
            xs, ys = get_batch(train_data, batch_size, block_size, mask)
            # xs, ys = my_xs, my_ys
            # t = @async get_batch(train_data, batch_size, block_size)
            
            # @benchmark begin
            # @time train_step!(model, xs, ys, opt_state)
            model, opt_state, l = train_step!(model, xs, ys, opt_state)
            # Flux.update!(opt_state, model, Flux.gradient(m -> loss(m, xs, ys), model)[1])
            # end
            next!(progress)
                # println("loss: ", l)
        end

        

        valid_loss, train_loss = estimate_loss()
        println("Epoch: $epoch, Train Loss: $train_loss, Valid Loss: $valid_loss")

        start_tokens = ones(Int, block_size, 1) |> device
        generate(model, start_tokens, 1000, block_size) |> cpu |> decode |> println

    end
    return model, opt_state
end


"Train a Bigram Language Model"
@main function main(filename::AbstractString; epochs::Integer=10,
                    block_size::Integer=10, batch_size::Integer=32,
                    n_layers::Integer=3, n_embed::Integer=32,
                    train_iters::Integer=500, valid_iters::Integer=100,
                    lr::Float64=0.001, n_head::Integer=4)

    println("Loading data...")
    data, chars, encode, decode = prepare_data()
    println("Done loading data")

    

    println("Preparing model")
    model = Model.Decoder(vocab_size=length(chars), n_layers=n_layers, n_embed=n_embed, n_head=n_head,
                          block_size=block_size, attention_dropout=0.2, residual_dropout=0.2)
    # model = Flux.paramtype(BFloat16, model) |> device
    model = model |> device
    println("Setting up optimizer")
    # model = Flux.Embedding(length(chars), length(chars)) |> device # 
    # Consider adding scaling for mix precison
    # opt = Optimiser(AdamW(BFloat16(lr)))
    opt = Optimiser(AdamW(Float32(lr)))
    opt_state = Flux.setup(opt, model)
    println("Done setting up optimizer")

    train_data, valid_data = splitobs(data, at = 0.9)
    model, opt_state = train(model, train_data, valid_data, opt_state, epochs,
                             block_size, batch_size, train_iters=train_iters, valid_iters=valid_iters)

    # start_tokens = ones(Int, block_size, 1) |> device
    # generate(model, start_tokens, 1000, block_size) |> cpu |> decode |> println

    model = model |> cpu

    println("Saving model to $filename")
    @save filename model
    println("Done!")
end   


end
