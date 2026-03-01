#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "tracker.h"
#include "threat.h"

namespace tv {

class MilitaryHUD {
public:
    MilitaryHUD();

    // Draw complete HUD overlay on frame
    void draw(cv::Mat& frame,
              const std::vector<Track>& tracks,
              const std::vector<ThreatInfo>& threats,
              float fps,
              int frame_number);

private:
    // Draw crosshair at center
    void draw_crosshair(cv::Mat& frame);

    // Draw detection boxes with tracking info
    void draw_detections(cv::Mat& frame, const std::vector<Track>& tracks,
                         const std::vector<ThreatInfo>& threats);

    // Draw minimap in corner showing relative positions
    void draw_minimap(cv::Mat& frame, const std::vector<Track>& tracks,
                      const std::vector<ThreatInfo>& threats);

    // Draw threat level bar
    void draw_threat_bar(cv::Mat& frame, ThreatLevel overall);

    // Draw target trails
    void draw_trails(cv::Mat& frame, const std::vector<Track>& tracks);

    // Draw info panel (FPS, target count, time)
    void draw_info_panel(cv::Mat& frame, float fps, int target_count,
                         int frame_number);

    // Draw compass rose
    void draw_compass(cv::Mat& frame);

    // Helper: draw text with background
    void draw_label(cv::Mat& frame, const std::string& text,
                    cv::Point pos, cv::Scalar color,
                    float font_scale = 0.5, int thickness = 1);

    // HUD colors
    static constexpr int HUD_ALPHA = 180;
    cv::Scalar hud_green_ = cv::Scalar(0, 255, 0);
    cv::Scalar hud_red_ = cv::Scalar(0, 0, 255);
    cv::Scalar hud_yellow_ = cv::Scalar(0, 255, 255);
    cv::Scalar hud_cyan_ = cv::Scalar(255, 255, 0);
    cv::Scalar hud_white_ = cv::Scalar(255, 255, 255);
    cv::Scalar hud_orange_ = cv::Scalar(0, 165, 255);

    std::chrono::steady_clock::time_point start_time_;
};

} // namespace tv
