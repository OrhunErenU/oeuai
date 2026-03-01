#include "detector.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <NvOnnxParser.h>

namespace tv {

static const std::vector<std::string> CLASS_NAMES_LIST = {
    "drone", "tank", "human", "weapon", "vehicle", "aircraft",
    "bird", "smoke", "fire", "explosion", "soldier", "civilian",
    "rifle", "pistol", "barrel"
};

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TRT] " << msg << std::endl;
    }
}

Detector::Detector() = default;

Detector::~Detector() {
    if (buffers_[0]) cudaFree(buffers_[0]);
    if (buffers_[1]) cudaFree(buffers_[1]);
}

bool Detector::load_engine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "[Detector] Engine file not found: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
        std::cerr << "[Detector] Failed to deserialize engine" << std::endl;
        return false;
    }

    context_.reset(engine_->createExecutionContext());

    // Get input/output tensor info
    auto input_name = engine_->getIOTensorName(0);
    auto output_name = engine_->getIOTensorName(1);
    auto input_dims = engine_->getTensorShape(input_name);
    auto output_dims = engine_->getTensorShape(output_name);

    input_h_ = input_dims.d[2];
    input_w_ = input_dims.d[3];

    // Output: [1, num_classes+4, num_detections] for YOLOv11
    int output_total = 1;
    for (int i = 0; i < output_dims.nbDims; i++) {
        output_total *= output_dims.d[i];
    }
    output_elements_ = output_total;

    // Allocate GPU buffers
    size_t input_size = 1 * 3 * input_h_ * input_w_ * sizeof(float);
    size_t output_size = output_total * sizeof(float);

    cudaMalloc(&buffers_[0], input_size);
    cudaMalloc(&buffers_[1], output_size);

    // Set tensor addresses
    context_->setTensorAddress(input_name, buffers_[0]);
    context_->setTensorAddress(output_name, buffers_[1]);

    std::cout << "[Detector] Engine loaded: " << engine_path << std::endl;
    std::cout << "  Input: " << input_w_ << "x" << input_h_ << std::endl;
    std::cout << "  Output elements: " << output_elements_ << std::endl;

    // Warmup
    std::vector<float> dummy(3 * input_h_ * input_w_, 0.0f);
    cudaMemcpy(buffers_[0], dummy.data(), input_size, cudaMemcpyHostToDevice);
    context_->enqueueV3(0);
    cudaDeviceSynchronize();
    std::cout << "  Warmup done" << std::endl;

    return true;
}

bool Detector::build_engine(const std::string& onnx_path, const std::string& engine_path,
                             bool fp16) {
    std::cout << "[Detector] Building engine from: " << onnx_path << std::endl;

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(logger_));
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, logger_));

    if (!parser->parseFromFile(onnx_path.c_str(),
                                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "[Detector] Failed to parse ONNX" << std::endl;
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB

    if (fp16 && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "  FP16 enabled" << std::endl;
    }

    auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        std::cerr << "[Detector] Failed to build engine" << std::endl;
        return false;
    }

    std::ofstream file(engine_path, std::ios::binary);
    file.write(static_cast<const char*>(serialized->data()), serialized->size());
    std::cout << "[Detector] Engine saved to: " << engine_path << std::endl;

    return load_engine(engine_path);
}

cv::Mat Detector::preprocess(const cv::Mat& frame) {
    // Letterbox resize
    int fw = frame.cols, fh = frame.rows;
    float scale = std::min(static_cast<float>(input_w_) / fw,
                           static_cast<float>(input_h_) / fh);
    int nw = static_cast<int>(fw * scale);
    int nh = static_cast<int>(fh * scale);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh));

    cv::Mat padded(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    int dx = (input_w_ - nw) / 2;
    int dy = (input_h_ - nh) / 2;
    resized.copyTo(padded(cv::Rect(dx, dy, nw, nh)));

    return padded;
}

std::vector<Detection> Detector::detect(const cv::Mat& frame, float conf_thresh,
                                         float iou_thresh) {
    if (!context_) return {};

    int fw = frame.cols, fh = frame.rows;
    float scale = std::min(static_cast<float>(input_w_) / fw,
                           static_cast<float>(input_h_) / fh);
    float pad_x = (input_w_ - fw * scale) / 2.0f;
    float pad_y = (input_h_ - fh * scale) / 2.0f;

    // Preprocess
    cv::Mat padded = preprocess(frame);

    // Convert to float blob [1, 3, H, W]
    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);

    // Copy to GPU
    size_t input_size = blob.total() * sizeof(float);
    cudaMemcpy(buffers_[0], blob.ptr<float>(), input_size, cudaMemcpyHostToDevice);

    // Inference
    context_->enqueueV3(0);
    cudaDeviceSynchronize();

    // Copy output from GPU
    std::vector<float> output(output_elements_);
    cudaMemcpy(output.data(), buffers_[1], output_elements_ * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Post-process
    return postprocess(output.data(), output_elements_, conf_thresh, iou_thresh,
                       scale, scale, pad_x, pad_y);
}

std::vector<Detection> Detector::postprocess(float* output, int output_size,
                                              float conf_thresh, float iou_thresh,
                                              float scale_x, float scale_y,
                                              float pad_x, float pad_y) {
    std::vector<Detection> detections;

    // YOLOv11 output format: [1, 4+num_classes, num_detections]
    // Transposed: rows = 4+num_classes, cols = num_detections
    int num_dets = output_size / (4 + num_classes_);
    if (num_dets <= 0) return detections;

    for (int i = 0; i < num_dets; i++) {
        // YOLOv11 output is transposed: [4+nc, num_dets]
        float cx = output[0 * num_dets + i];
        float cy = output[1 * num_dets + i];
        float w = output[2 * num_dets + i];
        float h = output[3 * num_dets + i];

        // Find best class
        int best_cls = 0;
        float best_conf = 0.0f;
        for (int c = 0; c < num_classes_; c++) {
            float score = output[(4 + c) * num_dets + i];
            if (score > best_conf) {
                best_conf = score;
                best_cls = c;
            }
        }

        if (best_conf < conf_thresh) continue;

        // Convert from letterbox coords to original frame coords
        float x1 = (cx - w / 2.0f - pad_x) / scale_x;
        float y1 = (cy - h / 2.0f - pad_y) / scale_y;
        float bw = w / scale_x;
        float bh = h / scale_y;

        Detection det;
        det.class_id = best_cls;
        det.confidence = best_conf;
        det.bbox = cv::Rect2f(x1, y1, bw, bh);
        det.center = cv::Point2f(x1 + bw / 2.0f, y1 + bh / 2.0f);
        det.class_name = (best_cls < CLASS_NAMES_LIST.size()) ?
                          CLASS_NAMES_LIST[best_cls] : "unknown";
        detections.push_back(det);
    }

    nms(detections, iou_thresh);
    return detections;
}

void Detector::nms(std::vector<Detection>& detections, float iou_thresh) {
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            if (detections[i].class_id != detections[j].class_id) continue;

            // Compute IoU
            cv::Rect2f inter = detections[i].bbox & detections[j].bbox;
            float inter_area = inter.area();
            float union_area = detections[i].bbox.area() + detections[j].bbox.area() - inter_area;
            if (union_area > 0 && inter_area / union_area > iou_thresh) {
                suppressed[j] = true;
            }
        }
    }

    std::vector<Detection> filtered;
    for (size_t i = 0; i < detections.size(); i++) {
        if (!suppressed[i]) filtered.push_back(detections[i]);
    }
    detections = std::move(filtered);
}

const std::vector<std::string>& Detector::class_names() {
    return CLASS_NAMES_LIST;
}

} // namespace tv
