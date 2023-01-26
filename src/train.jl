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
    return tokens
end


function get_batch(data, batch_size, block_size)
    start_offsets = rand(1:length(data)-block_size, batch_size)
    xs = hcat([data[start:start+block_size-1] for start in start_offsets]...)
    ys = hcat([data[start+1:start+block_size] for start in start_offsets]...)
    ys = onehotbatch(ys, 1:length(chars))
    return xs |> device, ys |> device
end

    
function train_step!(model, xs, ys, opt_state)
    l, gs = Flux.withgradient(m -> loss(m, xs, ys), model)
    opt_state, model = Flux.update!(opt_state, model, gs[1])
    return model, opt_state, l
end

function train(model, train_data, valid_data, opt_state, epochs, block_size, batch_size; train_iters=500, valid_iters=100)
    
    function estimate_loss()
        # we don't need to worry about modes here becaue we are not calculating gradients
        # when we calculate gradients flux will automatically switch to train mode
        valid_loss = 0
        train_loss = 0
        for i in 1:valid_iters
            xs, ys = get_batch(valid_data, batch_size, block_size)
            # outs = model(xs)
            # logitcrossentropy(outs, ys)
            valid_loss += loss(model, xs, ys)
            xs, ys = get_batch(train_data, batch_size, block_size)
            train_loss += loss(model, xs, ys)
        end
        valid_loss, train_loss = valid_loss/100, train_loss/100
        return valid_loss |> cpu, train_loss |> cpu
        
    end

    for epoch in 1:epochs
        @showprogress for i in 1:train_iters
            xs, ys = get_batch(train_data, batch_size, block_size)
            model, opt_state, l = train_step!(model, xs, ys, opt_state)
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
                    block_size=block_size, attention_dropout=0.2, residual_dropout=0.2, decoder_mask=true) |> device

    # model = Flux.Embedding(length(chars), length(chars)) |> device
    opt = AdamW(lr)
    opt_state = Flux.setup(opt, model)


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
