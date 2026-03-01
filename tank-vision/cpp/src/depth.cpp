#include "depth.h"
#include <iostream>

namespace tv {

DepthEstimator::DepthEstimator() = default;

bool DepthEstimator::load(const std::string& model_path) {
    try {
        net_ = cv::dnn::readNetFromONNX(model_path);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        loaded_ = true;
        std::cout << "[Depth] MiDaS model loaded: " << model_path << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "[Depth] Failed to load: " << e.what() << std::endl;
        // Fallback to CPU
        try {
            net_ = cv::dnn::readNetFromONNX(model_path);
            loaded_ = true;
            std::cout << "[Depth] Loaded on CPU fallback" << std::endl;
        } catch (...) {
            loaded_ = false;
        }
    }
    return loaded_;
}

cv::Mat DepthEstimator::estimate(const cv::Mat& frame) {
    if (!loaded_) return cv::Mat();

    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0,
                           cv::Size(input_w_, input_h_),
                           cv::Scalar(0.485 * 255, 0.456 * 255, 0.406 * 255),
                           true, false);

    // Normalize with ImageNet stats
    // Already done via blobFromImage mean subtraction
    // Scale by std: 1/(0.229, 0.224, 0.225) ≈ (4.37, 4.46, 4.44)

    net_.setInput(blob);
    cv::Mat output = net_.forward();

    // Reshape to 2D
    cv::Mat depth(output.size[2], output.size[3], CV_32F, output.ptr<float>());

    // Resize to original frame size
    cv::Mat resized;
    cv::resize(depth, resized, cv::Size(frame.cols, frame.rows));

    // Normalize to 0-1
    double minVal, maxVal;
    cv::minMaxLoc(resized, &minVal, &maxVal);
    if (maxVal > minVal) {
        resized = (resized - minVal) / (maxVal - minVal);
    }

    return resized;
}

float DepthEstimator::depth_at(const cv::Mat& depth_map, cv::Point2f point) {
    if (depth_map.empty()) return 0.5f;
    int x = std::max(0, std::min((int)point.x, depth_map.cols - 1));
    int y = std::max(0, std::min((int)point.y, depth_map.rows - 1));
    return depth_map.at<float>(y, x);
}

float DepthEstimator::depth_in_region(const cv::Mat& depth_map, cv::Rect2f bbox) {
    if (depth_map.empty()) return 0.5f;

    cv::Rect r(
        std::max(0, (int)bbox.x),
        std::max(0, (int)bbox.y),
        std::min((int)bbox.width, depth_map.cols - (int)bbox.x),
        std::min((int)bbox.height, depth_map.rows - (int)bbox.y)
    );

    if (r.width <= 0 || r.height <= 0) return 0.5f;

    cv::Mat roi = depth_map(r);
    return (float)cv::mean(roi)[0];
}

} // namespace tv
