#pragma once

#include <vector>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "detector.h"

namespace tv {

struct Track {
    int id;
    int class_id;
    std::string class_name;
    cv::Rect2f bbox;
    cv::Point2f center;
    cv::Point2f velocity;       // pixels/frame
    float speed;                // pixels/frame magnitude
    float heading;              // degrees (0=up, 90=right)
    float confidence;
    bool is_approaching;
    int age;                    // frames since creation
    int time_since_update;      // frames since last detection
    std::vector<cv::Point2f> trail; // last N positions

    // Kalman filter state
    cv::KalmanFilter kf;
    bool kf_initialized;
};

class ByteTracker {
public:
    ByteTracker(float high_thresh = 0.5f, float low_thresh = 0.1f,
                int max_lost = 30, int max_trail = 100);

    // Update tracker with new detections, returns active tracks
    std::vector<Track> update(const std::vector<Detection>& detections);

    // Get all active tracks
    const std::vector<Track>& tracks() const { return active_tracks_; }

private:
    // IoU between two bounding boxes
    float iou(const cv::Rect2f& a, const cv::Rect2f& b);

    // Hungarian-style greedy matching
    std::vector<std::pair<int, int>> match(
        const std::vector<Track>& tracks,
        const std::vector<Detection>& detections,
        float iou_thresh);

    // Initialize Kalman filter for a track
    void init_kalman(Track& track);

    // Predict next position using Kalman filter
    void predict(Track& track);

    float high_thresh_;
    float low_thresh_;
    int max_lost_;
    int max_trail_;
    int next_id_ = 1;

    std::vector<Track> active_tracks_;
    std::vector<Track> lost_tracks_;
};

} // namespace tv
