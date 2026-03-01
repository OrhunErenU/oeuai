#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>

namespace tv {

struct Detection {
    int class_id;
    float confidence;
    cv::Rect2f bbox;        // x, y, w, h (pixel)
    cv::Point2f center;     // center point
    std::string class_name;
};

// Custom TensorRT logger
class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class Detector {
public:
    Detector();
    ~Detector();

    // Load TensorRT engine from file
    bool load_engine(const std::string& engine_path);

    // Build engine from ONNX model
    bool build_engine(const std::string& onnx_path, const std::string& engine_path,
                      bool fp16 = true);

    // Run inference
    std::vector<Detection> detect(const cv::Mat& frame, float conf_thresh = 0.3f,
                                   float iou_thresh = 0.45f);

    // Get class names
    static const std::vector<std::string>& class_names();

    int input_width() const { return input_w_; }
    int input_height() const { return input_h_; }

private:
    // Preprocess image for YOLO
    cv::Mat preprocess(const cv::Mat& frame);

    // Post-process YOLO output
    std::vector<Detection> postprocess(float* output, int output_size,
                                        float conf_thresh, float iou_thresh,
                                        float scale_x, float scale_y,
                                        float pad_x, float pad_y);

    // NMS
    void nms(std::vector<Detection>& detections, float iou_thresh);

    TRTLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    void* buffers_[2] = {nullptr, nullptr}; // GPU buffers [input, output]
    int input_idx_ = 0;
    int output_idx_ = 1;
    int input_w_ = 640;
    int input_h_ = 640;
    int num_classes_ = 15;
    int output_elements_ = 0;
};

} // namespace tv
