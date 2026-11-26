#include "sam3.h"

#include <cstdlib>
#include <iostream>
#include <string>

static void usage(const char* exe) {
    std::cerr << "Usage: " << exe
              << " --model <model.gguf> --image <image> --prompt <text>"
              << " --out <mask.png> [--device auto|cuda|vulkan|cpu]\n";
}

int main(int argc, char** argv) {
    sam3_params params;
    std::string image_path, prompt, output_path;
    params.n_threads = 4;
    params.device = SAM3_DEVICE_AUTO;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) params.model_path = argv[++i];
        else if (arg == "--image" && i + 1 < argc) image_path = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
        else if (arg == "--out" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) params.n_threads = std::atoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) {
            const std::string device = argv[++i];
            if (device == "auto") params.device = SAM3_DEVICE_AUTO;
            else if (device == "cuda") params.device = SAM3_DEVICE_CUDA;
            else if (device == "vulkan") params.device = SAM3_DEVICE_VULKAN;
            else if (device == "cpu") params.device = SAM3_DEVICE_CPU;
            else { std::cerr << "Unknown device: " << device << "\n"; return 2; }
        }
        else if (arg == "--help") { usage(argv[0]); return 0; }
        else { std::cerr << "Unknown or incomplete argument: " << arg << "\n"; usage(argv[0]); return 2; }
    }

    if (params.model_path.empty() || image_path.empty() || prompt.empty() || output_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    sam3_image image = sam3_load_image(image_path);
    if (image.data.empty()) {
        std::cerr << "Could not load image: " << image_path << "\n";
        return 1;
    }

    std::cerr << "Loading model...\n";
    auto model = sam3_load_model(params);
    if (!model) {
        std::cerr << "Could not load model: " << params.model_path << "\n";
        return 1;
    }
    if (sam3_is_visual_only(*model)) {
        std::cerr << "Text prompts require a full SAM 3 checkpoint, not a visual-only model.\n";
        sam3_free_model(*model);
        return 1;
    }

    auto state = sam3_create_state(*model, params);
    if (!state) {
        std::cerr << "Could not create SAM 3 inference state.\n";
        sam3_free_model(*model);
        return 1;
    }

    std::cerr << "Encoding " << image.width << "x" << image.height
              << " image on " << sam3_backend_name(*model) << "...\n";
    if (!sam3_encode_image(*state, *model, image)) {
        std::cerr << "Image encoding failed.\n";
        return 1;
    }

    sam3_pcs_params segmentation;
    segmentation.text_prompt = prompt;
    segmentation.score_threshold = 0.25f;
    segmentation.nms_threshold = 0.1f;
    sam3_result result = sam3_segment_pcs(*state, *model, segmentation);

    if (result.detections.empty()) {
        std::cerr << "No detections for prompt: " << prompt << "\n";
        return 3;
    }

    int saved = 0;
    for (size_t i = 0; i < result.detections.size(); ++i) {
        std::string path = output_path;
        if (i > 0) {
            const size_t dot = path.find_last_of('.');
            path.insert(dot == std::string::npos ? path.size() : dot,
                        "-" + std::to_string(i + 1));
        }
        if (!sam3_save_mask(result.detections[i].mask, path)) {
            std::cerr << "Could not save mask: " << path << "\n";
            continue;
        }
        const auto& detection = result.detections[i];
        std::cout << "{\"index\":" << (i + 1)
                  << ",\"score\":" << detection.score
                  << ",\"x0\":" << detection.box.x0
                  << ",\"y0\":" << detection.box.y0
                  << ",\"x1\":" << detection.box.x1
                  << ",\"y1\":" << detection.box.y1
                  << ",\"mask\":\"" << path << "\"}\n";
        ++saved;
    }

    std::cerr << "Saved " << saved << " mask(s).\n";
    return saved > 0 ? 0 : 4;
}
