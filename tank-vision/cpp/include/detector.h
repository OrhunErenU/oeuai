#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace tv {

struct Detection {
    int class_id;
    float confidence;
    cv::Rect2f bbox;        // x, y, w, h (pixel)
    cv::Point2f center;     // center point
    std::string class_name;
};

class Detector {
public:
    Detector();
    ~Detector();

    // Load ONNX model (GPU if available, else CPU)
    bool load(const std::string& onnx_path, bool use_gpu = true);

    // Run inference
    std::vector<Detection> detect(const cv::Mat& frame, float conf_thresh = 0.3f,
                                   float iou_thresh = 0.45f);

    // Get class names
    static const std::vector<std::string>& class_names();

    int input_width() const { return input_w_; }
    int input_height() const { return input_h_; }
    bool is_loaded() const { return session_ != nullptr; }

private:
    cv::Mat preprocess(const cv::Mat& frame, float& scale, float& pad_x, float& pad_y);

    std::vector<Detection> postprocess(float* output, int num_dets,
                                        float conf_thresh, float iou_thresh,
                                        float scale, float pad_x, float pad_y);

    void nms(std::vector<Detection>& detections, float iou_thresh);

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;

    int input_w_ = 640;
    int input_h_ = 640;
    int num_classes_ = 15;
};

} // namespace tv
