#include <torch/script.h>
#include <torchaudio/csrc/forced_align/compute.h>

std::tuple<torch::Tensor, torch::Tensor> forced_align(
    const torch::Tensor& logProbs,
    const torch::Tensor& targets,
    const torch::Tensor& inputLengths,
    const torch::Tensor& targetLengths,
    const int64_t blank,
    double inter_word_blank_penalty,
    double inter_word_blank_penalty,
    const torch::Tensor& word_start_index) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("torchaudio::forced_align", "")
                       .typed<decltype(forced_align)>();
  return op.call(logProbs, targets, inputLengths, targetLengths, blank, inter_word_blank_penalty, inter_word_blank_penalty, word_start_index);
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "forced_align(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank, float inter_word_blank_penalty, float inter_word_blank_penalty, Tensor word_start_index) -> (Tensor, Tensor)");
}
