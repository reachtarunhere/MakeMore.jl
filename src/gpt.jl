module GPT

using Flux
using Flux.Losses: logitcrossentropy
using Flux: onehotbatch


text = read("input.txt", String)
chars = Set(text) |> collect |> sort

stoi = Dict(c => i for (i, c) in enumerate(chars))
itos = Dict(i => c for (i, c) in enumerate(chars))
encode(text) = [stoi[c] for c in text]
encode_char(c) = stoi[c]
decode(code) = [itos[i] for i in code] |> join

data = encode(text)

device = gpu

using BenchmarkTools

# @btime encode(text)
# @btime encode_char.(text |> collect) # note that strings are not broadcastable because treated as scalars

using MLUtils


using Random
# Random.seed!(1337)



function train_examples(chunk)
end    
    
    
    


function get_batch(data, batch_size, block_size)
    start_offsets = rand(1:length(data)-block_size, batch_size)
    xs = hcat([data[start:start+block_size-1] for start in start_offsets]...)
    ys = hcat([data[start+1:start+block_size] for start in start_offsets]...)
    # chunks = hcat([data[ix:ix+block_size] for ix in start_offsets]...)
    ys = onehotbatch(ys, 1:length(chars))
    return xs |> device, ys |> device
end


# xb, yb = get_batch(train_data, batch_size, block_size)

# model(xb)

# yb_oh = onehotbatch(yb, 1:length(chars))
# logitcrossentropy(model(xb), yb_oh)


function loss(model, xs, ys_oh)
    return logitcrossentropy(model(xs), ys_oh)
end



using StatsBase


function generate(model, start_tokens, max_len)

    sample_helper(p) = sample(1:length(chars), Weights(p))
    tokens = start_tokens
    for i in 1:max_len
        logits = model(tokens|>device) # this device thing can also be moved inside custom model itself?
        last_time_step = logits[:, end, :]
        probs = softmax(last_time_step)
        probs = probs |> cpu
        next_token = sample_helper.(eachcol(probs)) # sampling does not work on gpu
        tokens = cat(tokens, next_token', dims=1)
    end
    return tokens
end


using Flux.Optimise



function train_step!(model, xs, ys, opt_state)
    l, gs = Flux.withgradient(m -> loss(m, xs, ys), model)
    opt_state, model = Flux.update!(opt_state, model, gs[1])
    return model, opt_state, l
end



using ProgressMeter

function train(model, train_data, valid_data, opt_state, epochs, block_size, batch_size)


    function estimate_loss()
        # we don't need to worry about modes here becaue we are not calculating gradients
        # when we calculate gradients flux will automatically switch to train mode
        valid_loss = 0
        train_loss = 0
        for i in 1:100
            xs, ys = get_batch(valid_data, batch_size, block_size)
            valid_loss += loss(model, xs, ys)
            xs, ys = get_batch(train_data, batch_size, block_size)
            train_loss += loss(model, xs, ys)
        end
        valid_loss, train_loss = valid_loss/100, train_loss/100
        return valid_loss |> cpu, train_loss |> cpu
    end

    
    for epoch in 1:epochs
        l = 0
        @showprogress for i in 1:100
            xs, ys = get_batch(train_data, batch_size, block_size)
            model, opt_state, l = train_step!(model, xs, ys, opt_state)
        end

        valid_loss, train_loss = estimate_loss()
        println("Epoch: $epoch, Train Loss: $train_loss, Valid Loss: $valid_loss")
    end
    return model, opt_state
end





using Fire
using BSON: @save

"Train a Bigram Language Model"
@main function main(filename::AbstractString, epochs::Integer=10,
                    block_size::Integer=8, batch_size::Integer=4)


    train_data, valid_data = splitobs(data, at = 0.9)

    model = Flux.Embedding(length(chars), length(chars)) |> device
    opt = AdamW()
    opt_state = Flux.setup(opt, model)
    model, opt_state = train(model, train_data, valid_data, opt_state, epochs,
                             block_size, batch_size)

    start_tokens = ones(Int, 1, 1) |> device
    generate(model, start_tokens, 1000) |> cpu |> decode |> println

    
    model = model |> cpu

    println("Saving model to $filename")
    @save filename model
    println("Done!")
    
end   

end
