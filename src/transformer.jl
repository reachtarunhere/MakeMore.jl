module Transformer

using LinearAlgebra
using Flux 
using Flux: @functor
# using Zygote

sentence_batch = rand(2, 5, 3) # 2 features, 5 words, 3 sentences
bow_mask = UpperTriangular(ones(5, 5)) |> Array
bow = sentence_batch ⊠ bow_mask
sentence_batch[:, 1, 1] .+ sentence_batch[:, 2, 1] == bow[:, 2, 1]
bow_mask = bow_mask ./ sum(bow_mask, dims=1)

function make_bow_mask(n)
    bow_mask = UpperTriangular(ones(n, n)) |> Array
    bow_mask = bow_mask ./ sum(bow_mask, dims=1)
end


function make_bow_mask_softmax(n)
    bow_mask = zeros(n, n)
    zero_places = UpperTriangular(ones(n, n)) .!= 1
    bow_mask[zero_places] .= -Inf
    bow_mask = softmax(bow_mask, dims=1)
end

function bow_mask_softmax(n)
    bow_mask = UpperTriangular(ones(n, n)) |> Array
    bow_mask[bow_mask .== 0] .= -Inf
    bow_mask = softmax(bow_mask, dims=1)
end

# function make_inf_mask(n) # still breaks ad with mutation error 
#     mask = ones(n, n) |> triu
#     buffer = Zygote.Buffer(mask)
#     buffer[buffer .== 0] .= -Inf # Indexing gets fucked here
#     return copy(buffer)
# end

make_inf_mask(n) = tril(fill(-Inf, n, n), -1) # works without mutation
# make_inf_mask_2(n) = (1 .- triu(ones(n, n))) .* -Inf # nan issues

# make_inf_mask_2(3)
make_inf_mask(2)

function attention_mask(A) # custom implementation of masked_fill
    T, _, B = size(A)
    A = make_inf_mask(T) .+ A # not required for encoder as we want all nodes to talk to each other
    softmax(A, dims=1)
end


# function attention_mask(A) # custom implementation of masked_fill
#     T, _, B = size(A)
   
#     mask = UpperTriangular(ones(T, T)) .!= 1
#     A[mask, :] .= -Inf
#     softmax(A, dims=1)
# end



# struct SelfAttention
#     key::Dense
#     query::Dense
#     value::Dense
# end

# @functor SelfAttention

# function SelfAttention(dim_f, head_size) # consider declaring mask asn non trainable parameter in constructor
#     key = Dense(dim_f, head_size, bias=false)
#     query = Dense(dim_f, head_size, bias=false)
#     value = Dense(dim_f, head_size, bias=false)
#     SelfAttention(key, query, value)
# end

# function (sa::SelfAttention)(x)
#     k = sa.key(x) # head_size, T, B
#     q = sa.query(x) # head_size, T, B
#     v = sa.value(x)
#     A = batched_transpose(k) ⊠ q # T, T, B
#     A = attention_mask(A)
#     v ⊠ A # T, F, B
# end

# function (sa::SefAttention)(x, y) # cross attention
#     k = sa.key(x) # head_size, T, B
#     q = sa.query(y) # head_size, T, B
#     v = sa.value(x)
#     A = batched_transpose(k) ⊠ q # T, T, B
#     A = attention_mask(A)
#     v ⊠ A # T, F, B
# end



# slf_attn = SelfAttention(3, 2)

# A = slf_attn(rand(3, 10, 2))

# mask = UpperTriangular(ones(10, 10)) |> Array
# mask[mask .== 0] .= -Inf
# mask[mask .== 1] .= 0


# test_model = Chain(SelfAttention(3, 2), sum)


# Flux.withgradient(test_model) do m # works with proper attention mask
#     m(rand(3, 10, 2))
# end





make_inf_mask(3)
x = rand(3, 3, 2)
make_inf_mask(3) .+ x


# Feature, Time, Head, Batch

feature = 3
x = rand(3, 4, 5, 6)
layer = Dense(feature, 3 * feature)
qkv = layer(x) # 3F, T, H, B
remaining_dims = size(qkv)[2:end]
qkv = reshape(qkv, (feature, 3, remaining_dims...)) # F, 3, T, H, B

q2, k2, v2 = eachslice(qkv, dims=2) #
q2

# the above is not correct as we are using the same parameters for all heads

feature = 3
t = 4
head = 5
batch = 6

layer = Dense(feature, 3 * feature * head) |> gpu
x = rand(feature, t, batch) |> gpu

qkv = layer(x) # 3 * feature * head, t, batch
qkv = reshape(qkv, (feature, 3, head, t, batch)) # feature, 3, head, t, batch
size(qkv)
qkv = permutedims(qkv, (1, 2, 4, 3, 5))
# qkv = reshape(qkv, (feature, 3, t, :))
q, k, v = eachslice(qkv, dims=2) # feature, head, t, batch
q
# A = batched_transpose(k) ⊠ q # t, t, head*batch
# A = reshape(A, (t, t, head, batch))



to_batched_matrix(A) = reshape(A, size(A)[1:2]..., :), size(A)[3:end]
restore_batched_matrix(A, dims) = reshape(A, size(A)[1:2]..., dims...)

# function batched_transpose(A::AbstractArray)
#     dims = size(A)
#     A = reshape(A, (dim[1], dim[2], :))
#     A = batched_transpose(A)
#     reshape(A, size(A)[1:2]..., dims[3:end]...)
# end

# function batched_mul(A::AbstractArray, B::AbstractArray)
#     dims = size(A)
#     A = reshape(A, (dim[1], dim[2], :))
#     B = reshape(B, (dim[1], dim[2], :))
#     C = batched_mul(A, B)
#     reshape(C, size(C)[1:2]..., dims[3:end]...)
# end

import NNlib: batched_mul, batched_transpose

function NNlib.batched_mul(A::AbstractArray, B::AbstractArray)
    A, dims = to_batched_matrix(A)
    B, dims = to_batched_matrix(B)
    C = batched_mul(A, B)
    restore_batched_matrix(C, dims)
end

function NNlib.batched_transpose(A::AbstractArray)
    A, dims = to_batched_matrix(A)
    A = batched_transpose(A)
    restore_batched_matrix(A, dims)
end



# using NNlib: batched_mul, batched_transpose

#batched_mul(rand(3, 4, 5, 6), rand(4,3,5,6))
methods(batched_mul)
methods(batched_transpose)
v

A = batched_transpose(k) ⊠ q
out = v ⊠ A


function dot_attention(q, k, v, mask)
    A = (batched_transpose(k) ⊠ q) .+ mask
    A = softmax(A / √size(k, 1), dims=1)
    return v ⊠ A , A
end

struct MultiHeadSelfAttention
    in_dim::Int
    out_dim::Int # size of feature for output
    num_heads::Int
    context_len::Int
    MH_QKV::Dense
    mask::AbstractArray
    attention_dropout::Dropout
end

@functor MultiHeadSelfAttention
Flux.trainable(sa::MultiHeadSelfAttention) = (sa.MH_QKV,)

function MultiHeadSelfAttention(in_dim, out_dim, num_heads, context_len, decoder_mask=false)
    MH_QKV = Dense(in_dim, 3 * out_dim * num_heads)
    mask = decoder_mask ? make_inf_mask(context_len) : zeros(context_len, context_len)
    MultiHeadAttention(in_dim, out_dim, num_heads, context_len, MH_QKV, mask)
end


function (mha::MultiHeadSelfAttention)(x)
    qkv = mha.MH_QKV(x) # 3 * out_dim * num_heads, t, batch
    qkv = reshape(qkv, (mha.out_dim, 3, mha.num_heads, mha.context_len, :)) # out_dim, 3, num_heads, t, batch
    qkv = permutedims(qkv, (1, 2, 4, 3, 5)) # out_dim, 3, t, num_heads, batch
    q, k, v = eachslice(qkv, dims=2) # out_dim, t, head,  batch
    out, A = dot_attention(q, k, v, mha.mask)
    return out
end


struct MultiHeadCrossAttention
    in_dim_x::Int
    in_dim_y::Int
    out_dim::Int # size of feature for output
    num_heads::Int
    context_len::Int
    MH_Q::Dense
    MH_KV::Dense
    mask::AbstractArray
    dropout::Dropout
end

@functor MultiHeadCrossAttention
Flux.trainable(mha::MultiHeadCrossAttention) = (mha.MH_Q, mha.MH_KV)


function MultiHeadCrossAttention(in_dim_x, in_dim_y, out_dim, num_heads, context_len, decoder_mask=false)
    MH_Q = Dense(in_dim_x, out_dim * num_heads)
    MH_KV = Dense(in_dim_y, 2 * out_dim * num_heads)
    mask = decoder_mask ? make_inf_mask(context_len) : zeros(context_len, context_len)
    MultiHeadCrossAttention(in_dim_x, in_dim_y, out_dim, num_heads, context_len, MH_Q, MH_KV, mask)
end

function (mha::MultiHeadCrossAttention)(x, y)
    q = mha.MH_Q(x) # out_dim * num_heads, t, batch
    q = reshape(q, (mha.out_dim, mha.num_heads, mha.context_len, :)) # out_dim, num_heads, t, batch
    q = permutedims(q, (1, 3, 2, 4)) # out_dim, t, num_heads, batch
    kv = mha.MH_KV(y) # 2 * out_dim * num_heads, t, batch
    kv = reshape(kv, (mha.out_dim, 2, mha.num_heads, mha.context_len, :)) # out_dim, 2, num_heads, t, batch
    kv = permutedims(kv, (1, 2, 4, 3, 5)) # out_dim, 2, t, num_heads, batch
    k, v = eachslice(kv, dims=2) # out_dim, t, head,  batch
    out, A = dot_attention(q, k, v, mha.mask)
    return out
end

# remember to make mask non trainable

function Block(in_dim, out_dim, num_heads, context_len, mlp_expand = 4,
               decoder_mask=false)
    mha = MultiHeadSelfAttention(in_dim, out_dim, num_heads, context_len, decoder_mask)
    MLP = Chain(Dense(out_dim, out_dim * mlp_expand, gelu), 
                Dense(out_dim * mlp_expand, out_dim),
                Dropout(0.1))

    
    return Chain(SkipConnection(Chain(LayerNorm(in_dim), mha), +),
                 SkipConnection(Chain(LayerNorm(out_dim), MLP), +))
end


# maybe a good idea to give vaswani defaults here


function PositionalAwareEmbedding(vocab_size, embed_size, pos_size, block_size)
    embed = Flux.Embedding(vocab_size, embed_size)
    pos_embed = Chain(x -> 1:block_size, Flux.Embedding(block_size, pos_size))
    return Parallel(.+, embed, pos_embed)
end


function TransformerDecoder(in_dim, out_dim, num_heads,
                            context_len, num_layers, vocab_size,
                            mlp_expand = 4)
    embed = Embedding(vocab_size, in_dim) #make positional embedding
    blocks = [Block(in_dim, out_dim, num_heads, context_len, mlp_expand) for _ in 1:num_layers]
    return Chain(embed, blocks...)
end

# for each block input and output dims have to be same because output of one is input of other













                  



    
    



    




function transpose_magic(A, dim1::Int, dim2::Int)
    dims = collect(1:length(size(A)))
    dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
    permutedims(A, dims)
end

transpose_magic(k, 1, 2)

end
