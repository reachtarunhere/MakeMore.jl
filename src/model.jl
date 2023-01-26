module Model

# using Flux: @functor, Dense, Chain, Parallel, Dropout, LayerNorm
using Flux
using Flux: @functor

using LinearAlgebra: tril


using CUDA

CUDA.allowscalar(false)

to_batched_matrix(A) = reshape(A, size(A)[1:2]..., :), size(A)[3:end]
restore_batched_matrix(A, dims) = reshape(A, size(A)[1:2]..., dims...)


function Flux.batched_mul(A::AbstractArray, B::AbstractArray)
    A, dims = to_batched_matrix(A)
    B, dims = to_batched_matrix(B)
    C = batched_mul(A, B)
    restore_batched_matrix(C, dims)
end

function Flux.batched_transpose(A::AbstractArray)
    permutedims(A, (2, 1, 3:length(size(A))...))
end


methods(batched_mul)


get_fn_name(fn) = Symbol(fn) |> String 
make_from_config(constructor) = config -> constructor(config[get_fn_name(constructor)]...)



function dot_attention(q, k, v, mask)
    A = (batched_transpose(k) ⊠ q) .+ mask
    A = softmax(A / √size(k, 1), dims=1)
    return A
end


struct MHSelfAttention
    n_embed::Int
    n_head::Int
    head_size::Int
    MH_QKV::Dense
    MH_O::Dense
    mask::AbstractArray
    attention_dropout::Dropout
    residual_dropout::Dropout
end

@functor MHSelfAttention

Flux.trainable(m::MHSelfAttention) = (m.MH_QKV, m.MH_O)

make_decoder_mask(block_size) = tril(fill(-1f9, block_size, block_size), -1)

function MHSelfAttention(;n_embed, n_head, block_size, attention_dropout, residual_dropout, decoder_mask=false)
    head_size = n_embed ÷ n_head
    MH_QKV = Dense(n_embed, 3 * head_size * n_head)
    MH_O = Dense(n_embed, n_embed)
    mask = decoder_mask ? make_decoder_mask(block_size) : zeros(block_size, block_size)
    MHSelfAttention(n_embed, n_head, head_size, MH_QKV, MH_O, mask, Dropout(attention_dropout), Dropout(residual_dropout))
end

function split_fused_heads(x, n_head, head_size, n_fused=3) # seperate heads and qkv
    x = reshape(x, head_size, n_fused, n_head, size(x)[2:end]...)
    x = permutedims(x, (1, 2, 4, 3, 5))
    return x[:, 1, :, :, :], x[:, 2, :, :, :], x[:, 3, :, :, :] # bad as it makes a copy
    # should note ultra terrible since we allocate even after a dense op
end



function (mha::MHSelfAttention)(x)
    # qkv = mha.MH_QKV(x) # 3 * head_size * n_head, T, BS
    q, k, v = split_fused_heads(mha.MH_QKV(x), mha.n_head, mha.head_size)
    A = dot_attention(q, k, v, mha.mask) |> mha.attention_dropout # T, T, n_head, BS
    o = v ⊠ A # head_size, T, n_head, BS
    # now fuse heads
    o = vcat(eachslice(o, dims=3)...) # head_size * n_head, T, BS
    o = mha.MH_O(o) |> mha.residual_dropout
    return o
end


function Block(;n_embed, n_head, block_size, attention_dropout, residual_dropout, decoder_mask=false)
    mha = MHSelfAttention(n_embed=n_embed, n_head=n_head, block_size=block_size, attention_dropout=attention_dropout,
                          residual_dropout=residual_dropout, decoder_mask=decoder_mask)
    MLP = Chain(Dense(n_embed, 4 * n_embed, gelu), Dense(4 * n_embed, n_embed), Dropout(residual_dropout))

    return Chain(SkipConnection(Chain(LayerNorm(n_embed), mha), +),
                 SkipConnection(Chain(LayerNorm(n_embed), MLP), +))
end

function PositionalAwareEmbedding(vocab_size, embed_size, block_size)
    embed = Flux.Embedding(vocab_size, embed_size)
    pos_embed = Chain(x -> 1:block_size, Flux.Embedding(block_size, embed_size))
    return Parallel(.+, embed, pos_embed)
end

function Decoder(;vocab_size, n_layers, n_embed, n_head, block_size, attention_dropout, residual_dropout, decoder_mask=true)
    embed = PositionalAwareEmbedding(vocab_size, n_embed, block_size)
    blocks = [Block(n_embed=n_embed, n_head=n_head, block_size=block_size, attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout, decoder_mask=decoder_mask) for _ in 1:n_layers]
    lm_head = Dense(embed[1].weight') # n_embed, vocab_size (weight tying)
    return Chain(embed, blocks..., LayerNorm(n_embed), lm_head)
end


m = Decoder(vocab_size=100, n_layers=2, n_embed=512, n_head=8, block_size=512, attention_dropout=0.1, residual_dropout=0.1)
xs = rand(1:100, 512, 2)
m(xs)

# TODO: implement model cropping
# TODO: implement data parallelism
# TODO: implement more optimizer details
# TODO: implement more sampling methods
# TODO: count total number of params

end # module
