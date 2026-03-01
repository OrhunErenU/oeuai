#include "threat.h"
#include <algorithm>
#include <cmath>

namespace tv {

ThreatAssessor::ThreatAssessor() = default;

float ThreatAssessor::class_weight(int class_id) {
    // Higher = more threatening
    switch (class_id) {
        case 0:  return 0.9f;  // drone — highest air threat
        case 1:  return 0.95f; // tank
        case 2:  return 0.3f;  // human (unclassified)
        case 3:  return 0.7f;  // weapon
        case 4:  return 0.4f;  // vehicle
        case 5:  return 0.8f;  // aircraft
        case 6:  return 0.05f; // bird
        case 7:  return 0.2f;  // smoke
        case 8:  return 0.3f;  // fire
        case 9:  return 0.6f;  // explosion
        case 10: return 0.6f;  // soldier
        case 11: return 0.1f;  // civilian
        case 12: return 0.65f; // rifle
        case 13: return 0.55f; // pistol
        case 14: return 0.75f; // barrel (RPG etc)
        default: return 0.2f;
    }
}

float ThreatAssessor::proximity_factor(const cv::Rect2f& bbox, int fw, int fh) {
    // Larger bbox relative to frame = closer = higher threat
    float area_ratio = bbox.area() / (fw * fh);
    return std::min(1.0f, area_ratio * 10.0f); // Scale up, cap at 1
}

float ThreatAssessor::approach_factor(const Track& track) {
    if (track.speed < 2.0f) return 0.0f;
    // Moving down in frame = approaching camera
    return std::max(0.0f, std::min(1.0f, track.velocity.y / 10.0f));
}

float ThreatAssessor::size_factor(const Track& track) {
    // If bbox is growing over time = approaching
    if (track.trail.size() < 5) return 0.0f;
    // Compare recent bbox area trend (simplified)
    return track.is_approaching ? 0.3f : 0.0f;
}

ThreatInfo ThreatAssessor::assess(const Track& track, int fw, int fh) {
    ThreatInfo info;
    info.track_id = track.id;

    float w_class = class_weight(track.class_id);
    float w_prox = proximity_factor(track.bbox, fw, fh);
    float w_approach = approach_factor(track);
    float w_size = size_factor(track);

    // Weighted combination
    info.score = w_class * 0.4f + w_prox * 0.25f + w_approach * 0.2f + w_size * 0.15f;
    info.score = std::max(0.0f, std::min(1.0f, info.score));

    // Determine level
    if (info.score >= 0.75f) info.level = ThreatLevel::CRITICAL;
    else if (info.score >= 0.5f) info.level = ThreatLevel::HIGH;
    else if (info.score >= 0.25f) info.level = ThreatLevel::MEDIUM;
    else info.level = ThreatLevel::LOW;

    info.color = level_color(info.level);
    info.description = track.class_name + " [" + level_name(info.level) + "]";

    return info;
}

std::vector<ThreatInfo> ThreatAssessor::assess_all(const std::vector<Track>& tracks,
                                                     int fw, int fh) {
    std::vector<ThreatInfo> threats;
    threats.reserve(tracks.size());
    for (auto& t : tracks) {
        threats.push_back(assess(t, fw, fh));
    }
    // Sort by score descending
    std::sort(threats.begin(), threats.end(),
              [](const ThreatInfo& a, const ThreatInfo& b) { return a.score > b.score; });
    return threats;
}

ThreatLevel ThreatAssessor::overall_level(const std::vector<ThreatInfo>& threats) {
    ThreatLevel max_level = ThreatLevel::LOW;
    for (auto& t : threats) {
        if (t.level > max_level) max_level = t.level;
    }
    return max_level;
}

cv::Scalar ThreatAssessor::level_color(ThreatLevel level) {
    switch (level) {
        case ThreatLevel::LOW:      return cv::Scalar(0, 255, 0);     // Green
        case ThreatLevel::MEDIUM:   return cv::Scalar(0, 255, 255);   // Yellow
        case ThreatLevel::HIGH:     return cv::Scalar(0, 165, 255);   // Orange
        case ThreatLevel::CRITICAL: return cv::Scalar(0, 0, 255);     // Red
        default:                    return cv::Scalar(200, 200, 200);
    }
}

std::string ThreatAssessor::level_name(ThreatLevel level) {
    switch (level) {
        case ThreatLevel::LOW:      return "LOW";
        case ThreatLevel::MEDIUM:   return "MEDIUM";
        case ThreatLevel::HIGH:     return "HIGH";
        case ThreatLevel::CRITICAL: return "CRITICAL";
        default:                    return "UNKNOWN";
    }
}

} // namespace tv
