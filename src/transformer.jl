module Transformer

using LinearAlgebra
using Flux

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

end
