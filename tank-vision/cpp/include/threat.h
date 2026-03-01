#pragma once

#include <string>
#include <vector>
#include "tracker.h"

namespace tv {

enum class ThreatLevel {
    LOW = 0,
    MEDIUM = 1,
    HIGH = 2,
    CRITICAL = 3
};

struct ThreatInfo {
    int track_id;
    ThreatLevel level;
    float score;            // 0.0 - 1.0
    std::string description;
    cv::Scalar color;       // Display color
};

class ThreatAssessor {
public:
    ThreatAssessor();

    // Assess threat for a single track
    ThreatInfo assess(const Track& track, int frame_w, int frame_h);

    // Assess all tracks, sorted by threat level (highest first)
    std::vector<ThreatInfo> assess_all(const std::vector<Track>& tracks,
                                        int frame_w, int frame_h);

    // Get overall threat level (maximum of all tracks)
    ThreatLevel overall_level(const std::vector<ThreatInfo>& threats);

    // Get color for threat level
    static cv::Scalar level_color(ThreatLevel level);
    static std::string level_name(ThreatLevel level);

private:
    // Class-based base threat weights
    float class_weight(int class_id);

    // Proximity factor (closer = higher threat)
    float proximity_factor(const cv::Rect2f& bbox, int frame_w, int frame_h);

    // Approach factor (coming towards camera = higher)
    float approach_factor(const Track& track);

    // Size change factor (growing bbox = approaching)
    float size_factor(const Track& track);
};

} // namespace tv
