module MakeMore

words = readlines("names.txt")
num_words = length(words)
min_word_len = minimum(length.(words))
max_word_len = maximum(length.(words))

wrap_seq(seq) = ["<S>";  seq...; "<E>"]
bigrams(seq) = zip(seq, seq[2:end]) |> collect
# use zip above to avoid allocating an array
# there is information about how to start and when to end in the data implicitly

words_sample = words[1:10]

# wrap_seq.(words_sample) .|> bigrams


flatten(xss) = vcat(xss...)

all_bigrams = wrap_seq.(words) .|> bigrams |> flatten

# can be more space efficient by doing it word by word

function count_items(items)
    counts = Dict{eltype(items), Int}()
    for item in items
        counts[item] = get(counts, item, 0) + 1
    end
    return counts |> collect
end

function char_bigram_counts(words)
    counts = Dict()
    for word in words
        word_bigrams = bigrams(wrap_seq(word))
        for bigram in word_bigrams
            counts[bigram] = get(counts, bigram, 0) + 1
        end
    end
    return counts |> collect
end
        

bigram_counts = count_items(all_bigrams) 
bigram_counts2 = char_bigram_counts(words)
length(bigram_counts) == length(bigram_counts2)
bigram_counts == bigram_counts2
sorted_bigram_counts = sort(bigram_counts, by = x -> x[2], rev = true)




chars = collect('a':'z')
itos = Dict{Int32, Any}(pairs(chars))
itos[27] = "<S>"
itos[28] = "<E>"
stoi = Dict(c => i for (i, c) in itos)

function get_bigram_counts(words)
    counts = zeros(Int32, 28, 28)
    for word in words
        word_bigrams = bigrams(wrap_seq(word))
        for bigram in word_bigrams
            counts[stoi[bigram[1]], stoi[bigram[2]]] += 1
        end
    end
    return counts
end

bigram_counts3 = get_bigram_counts(words)

using Plots

function plot_bigram_counts(counts)

    heatmap(bigram_counts3, c=:Blues_9, xticks = (1:28, [chars; "S" ; "E"]), yticks = (1:28, [chars; "S" ; "E"]), size = (800, 800), yflip = true)


    annotations_labels = [text(itos[i] * itos[j], :gray, 3, :center, :top) for i in 1:28, j in 1:28] |> flatten
    annotations_counts = [text(bigram_counts3[i, j], :gray, 3, :center, :bottom) for i in 1:28, j in 1:28] |> flatten


    indexes = [(i, j) for i in 1:28, j in 1:28] |> flatten
    xs = [x[1] for x in indexes]
    ys = [x[2] for x in indexes]

    annotate!(ys, xs, annotations_labels)
    annotate!(ys, xs, annotations_counts)
    gui()
end

plot_bigram_counts(bigram_counts3)

start_bigram_counts = bigram_counts3[stoi["<S>"], :]
p = start_bigram_counts ./ sum(start_bigram_counts)

using Distributions
using Random

rng = Random.seed!(2147483647)



d = Multinomial(1, p)
custom_sample() = itos[rand(rng, d) |> argmax]
samples = [custom_sample() for i in 1:1000]

# rowwise operation julia
using LinearAlgebra

P = mapslices(x -> x ./ sum(x), bigram_counts3, dims = 2)

row_sums = sum(bigram_counts3, dims = 2)
#row_sums_expanded = row_sums .* ones(1, 28)
# doing repeat manually
row_sums_expanded = repeat(row_sums, 1, 28)
P2 = bigram_counts3 ./ row_sums_expanded
P3 = (bigram_counts3 .+ 1) ./ row_sums # in broadcasting the smaller dimension is repeated no extra memory is allocated unlike in repeat. unlike python the broadcasting is explicit
# rowwise operation via repeat
P4 = bigram_counts3 ./ row_sums' # transpose gives wrong result


bigram_counts3[stoi["<S>"], :]
replace_nan(x) = isnan(x) ? 0 : x

replace_nan.(P2) ≈ replace_nan.(P)

function sample_next_char(char)
    # p = bigram_counts3[stoi[char], :] ./ sum(bigram_counts3[stoi[char], :])
    p = P[stoi[char], :]
    d = Multinomial(1, p)
    return itos[rand(rng, d) |> argmax]
end

samples = [sample_next_char("<S>") for i in 1:1000]
count(x -> x == 'a', samples) / 1000

function sample_word()
    word = ""
    char = "<S>"
    while char != "<E>"
        char = sample_next_char(char)
        word = word * char
    end
    return word[1:end-3]
end

hundered_words = [sample_word() for i in 1:100]

function negative_log_likelihood(word)
    word_bigrams = bigrams(wrap_seq(word))
    log_likelihood = 0.0
    for bigram in word_bigrams
        log_likelihood += log(P3[stoi[bigram[1]] , stoi[bigram[2]]])
    end
    return -log_likelihood/length(word_bigrams)
end

negative_log_likelihood("andrej")

log_likelihood_dataset = negative_log_likelihood.(words) |> mean


# above can also be done by using samples with weights
end
