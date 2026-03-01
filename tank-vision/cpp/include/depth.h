#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>

namespace tv {

class DepthEstimator {
public:
    DepthEstimator();

    // Load depth estimation model (MiDaS small ONNX)
    bool load(const std::string& model_path);

    // Estimate depth map from frame
    cv::Mat estimate(const cv::Mat& frame);

    // Get depth value at a point (normalized 0-1, 0=close, 1=far)
    float depth_at(const cv::Mat& depth_map, cv::Point2f point);

    // Get average depth in a region
    float depth_in_region(const cv::Mat& depth_map, cv::Rect2f bbox);

    bool is_loaded() const { return loaded_; }

private:
    cv::dnn::Net net_;
    bool loaded_ = false;
    int input_w_ = 256;
    int input_h_ = 256;
};

} // namespace tv
