#include "detector.h"
#include <iostream>
#include <algorithm>

namespace tv {

static const std::vector<std::string> CLASS_NAMES_LIST = {
    "Drone", "Tank", "Human", "Weapon", "Vehicle", "Aircraft",
    "Bird", "Smoke", "Fire", "Explosion", "Soldier", "Civilian",
    "Rifle", "Pistol", "Barrel"
};

Detector::Detector() = default;
Detector::~Detector() = default;

bool Detector::load(const std::string& onnx_path, bool use_gpu) {
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "TankVision");

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // GPU (CUDA) if available
        if (use_gpu) {
            try {
                OrtCUDAProviderOptions cuda_opts;
                cuda_opts.device_id = 0;
                opts.AppendExecutionProvider_CUDA(cuda_opts);
                std::cout << "[Detector] CUDA execution provider enabled" << std::endl;
            } catch (const Ort::Exception& e) {
                std::cout << "[Detector] CUDA not available, using CPU: " << e.what() << std::endl;
            }
        }

        session_ = std::make_unique<Ort::Session>(*env_, onnx_path.c_str(), opts);

        // Get input info
        size_t num_inputs = session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; i++) {
            auto name = session_->GetInputNameAllocated(i, allocator_);
            input_names_str_.push_back(name.get());
            input_names_.push_back(input_names_str_.back().c_str());

            auto shape = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() == 4) {
                input_h_ = static_cast<int>(shape[2]);
                input_w_ = static_cast<int>(shape[3]);
            }
        }

        // Get output info
        size_t num_outputs = session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            auto name = session_->GetOutputNameAllocated(i, allocator_);
            output_names_str_.push_back(name.get());
            output_names_.push_back(output_names_str_.back().c_str());
        }

        std::cout << "[Detector] ONNX model loaded: " << onnx_path << std::endl;
        std::cout << "  Input: " << input_w_ << "x" << input_h_ << std::endl;

        // Warmup
        cv::Mat dummy(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
        detect(dummy, 0.5f, 0.5f);
        std::cout << "  Warmup done" << std::endl;

        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "[Detector] Load failed: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat Detector::preprocess(const cv::Mat& frame, float& scale, float& pad_x, float& pad_y) {
    int fw = frame.cols, fh = frame.rows;
    scale = std::min(static_cast<float>(input_w_) / fw,
                     static_cast<float>(input_h_) / fh);
    int nw = static_cast<int>(fw * scale);
    int nh = static_cast<int>(fh * scale);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(nw, nh));

    cv::Mat padded(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    pad_x = (input_w_ - nw) / 2.0f;
    pad_y = (input_h_ - nh) / 2.0f;
    resized.copyTo(padded(cv::Rect(static_cast<int>(pad_x), static_cast<int>(pad_y), nw, nh)));

    return padded;
}

std::vector<Detection> Detector::detect(const cv::Mat& frame, float conf_thresh,
                                         float iou_thresh) {
    if (!session_) return {};

    float scale, pad_x, pad_y;
    cv::Mat padded = preprocess(frame, scale, pad_x, pad_y);

    // Convert to float blob [1, 3, H, W] - NCHW format
    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);

    // Create input tensor
    std::vector<int64_t> input_shape = {1, 3, input_h_, input_w_};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.ptr<float>(), blob.total(),
        input_shape.data(), input_shape.size());

    // Run inference
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), output_names_.size());

    // Get output data
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    // YOLOv11 output: [1, 4+num_classes, num_detections]
    int num_dets = 0;
    if (output_shape.size() == 3) {
        num_dets = static_cast<int>(output_shape[2]);
        num_classes_ = static_cast<int>(output_shape[1]) - 4;
    }

    return postprocess(output_data, num_dets, conf_thresh, iou_thresh, scale, pad_x, pad_y);
}

std::vector<Detection> Detector::postprocess(float* output, int num_dets,
                                              float conf_thresh, float iou_thresh,
                                              float scale, float pad_x, float pad_y) {
    std::vector<Detection> detections;
    if (num_dets <= 0) return detections;

    int stride = num_dets; // Transposed layout

    for (int i = 0; i < num_dets; i++) {
        float cx = output[0 * stride + i];
        float cy = output[1 * stride + i];
        float w  = output[2 * stride + i];
        float h  = output[3 * stride + i];

        // Find best class
        int best_cls = 0;
        float best_conf = 0.0f;
        for (int c = 0; c < num_classes_; c++) {
            float score = output[(4 + c) * stride + i];
            if (score > best_conf) {
                best_conf = score;
                best_cls = c;
            }
        }

        if (best_conf < conf_thresh) continue;

        // Convert from letterbox to original coords
        float x1 = (cx - w / 2.0f - pad_x) / scale;
        float y1 = (cy - h / 2.0f - pad_y) / scale;
        float bw = w / scale;
        float bh = h / scale;

        Detection det;
        det.class_id = best_cls;
        det.confidence = best_conf;
        det.bbox = cv::Rect2f(x1, y1, bw, bh);
        det.center = cv::Point2f(x1 + bw / 2.0f, y1 + bh / 2.0f);
        det.class_name = (best_cls < static_cast<int>(CLASS_NAMES_LIST.size())) ?
                          CLASS_NAMES_LIST[best_cls] : "Unknown";
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
