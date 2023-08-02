#pragma once

#include <torch/script.h>

std::tuple<torch::Tensor, torch::Tensor> forced_align(
    const torch::Tensor& logProbs,
    const torch::Tensor& targets,
    const torch::Tensor& inputLengths,
    const torch::Tensor& targetLengths,
    const int64_t blank,
    double inter_word_blank_penalty,
    double inter_word_blank_penalty,
    const torch::Tensor& word_start_index);
