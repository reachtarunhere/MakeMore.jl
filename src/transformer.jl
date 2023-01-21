module Transformer

using LinearAlgebra
using Flux
using Flux: @functor
# using Zygote

sentence_batch = rand(2, 5, 3) # 2 features, 5 words, 3 sentences
bow_mask = UpperTriangular(ones(5, 5)) |> Array
bow = sentence_batch ⊠ bow_mask
sentence_batch[:, 1, 1] .+ sentence_batch[:, 2, 1] == bow[:, 2, 1]
bow_mask = bow_mask ./ sum(bow_mask, dims=1) #

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



struct SelfAttention
    key::Dense
    query::Dense
    value::Dense
end

@functor SelfAttention

function SelfAttention(dim_f, head_size)
    key = Dense(dim_f, head_size, bias=false)
    query = Dense(dim_f, head_size, bias=false)
    value = Dense(dim_f, head_size, bias=false)
    SelfAttention(key, query, value)
end

function (sa::SelfAttention)(x)
    k = sa.key(x) # head_size, T, B
    q = sa.query(x) # head_size, T, B
    v = sa.value(x)
    A = batched_transpose(k) ⊠ q # T, T, B
    A = attention_mask(A)
    v ⊠ A # T, F, B
end

slf_attn = SelfAttention(3, 2)

A = slf_attn(rand(3, 10, 2))

mask = UpperTriangular(ones(10, 10)) |> Array
mask[mask .== 0] .= -Inf
mask[mask .== 1] .= 0


test_model = Chain(SelfAttention(3, 2), sum)


# Flux.withgradient(test_model) do m # works with proper attention mask
#     m(rand(3, 10, 2))
# end





make_inf_mask(3)
x = rand(3, 3, 2)
make_inf_mask(3) .+ x

end
