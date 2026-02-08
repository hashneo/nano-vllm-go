// PyTorch C++ wrapper for Go CGo integration
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <memory>
#include <cmath>
#include <random>

extern "C" {

// Load a TorchScript model
void* load_model(const char* path) {
    try {
        auto module = new torch::jit::script::Module;
        *module = torch::jit::load(path);
        module->eval();
        return static_cast<void*>(module);
    } catch (const c10::Error& e) {
        return nullptr;
    }
}

// Free the model
void free_model(void* model) {
    if (model) {
        delete static_cast<torch::jit::script::Module*>(model);
    }
}

// Run inference
void* run_inference(void* model, long* input_ids, int batch_size, int seq_len, float temperature) {
    if (!model || !input_ids) {
        return nullptr;
    }

    try {
        auto module = static_cast<torch::jit::script::Module*>(model);

        // Create input tensor
        std::vector<int64_t> input_vec(input_ids, input_ids + batch_size * seq_len);
        auto options = torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(torch::kCPU);

        auto input_tensor = torch::from_blob(
            input_vec.data(),
            {batch_size, seq_len},
            options
        ).clone();

        // Run model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        auto output = module->forward(inputs).toTensor();

        // Get last token logits
        auto last_logits = output.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});

        // Allocate and return logits (caller must free)
        auto logits_flat = last_logits.contiguous().data_ptr<float>();
        int logits_size = last_logits.numel();
        float* logits_copy = new float[logits_size];
        std::memcpy(logits_copy, logits_flat, logits_size * sizeof(float));

        return static_cast<void*>(logits_copy);

    } catch (const c10::Error& e) {
        return nullptr;
    }
}

// Sample token from logits with temperature
long sample_token(void* logits_ptr, int vocab_size, float temperature) {
    if (!logits_ptr) {
        return -1;
    }

    float* logits = static_cast<float*>(logits_ptr);

    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }

    // Softmax
    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float sum_exp = 0.0f;
    std::vector<float> probs(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }

    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum_exp;
    }

    // Sample
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());

    int sampled_token = dist(gen);

    // Clean up
    delete[] logits;

    return static_cast<long>(sampled_token);
}

} // extern "C"
