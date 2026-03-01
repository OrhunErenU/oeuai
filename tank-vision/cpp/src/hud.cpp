#include "hud.h"
#include <cmath>
#include <sstream>
#include <iomanip>

namespace tv {

MilitaryHUD::MilitaryHUD()
    : start_time_(std::chrono::steady_clock::now()) {}

void MilitaryHUD::draw_label(cv::Mat& frame, const std::string& text,
                              cv::Point pos, cv::Scalar color,
                              float font_scale, int thickness) {
    int baseline = 0;
    cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &baseline);
    cv::rectangle(frame, cv::Point(pos.x - 2, pos.y - sz.height - 4),
                  cv::Point(pos.x + sz.width + 2, pos.y + baseline + 2),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, text, pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, color, thickness);
}

void MilitaryHUD::draw_crosshair(cv::Mat& frame) {
    int cx = frame.cols / 2, cy = frame.rows / 2;
    int sz = 30, gap = 8;

    // Crosshair lines
    cv::line(frame, cv::Point(cx - sz, cy), cv::Point(cx - gap, cy), hud_green_, 1);
    cv::line(frame, cv::Point(cx + gap, cy), cv::Point(cx + sz, cy), hud_green_, 1);
    cv::line(frame, cv::Point(cx, cy - sz), cv::Point(cx, cy - gap), hud_green_, 1);
    cv::line(frame, cv::Point(cx, cy + gap), cv::Point(cx, cy + sz), hud_green_, 1);

    // Center dot
    cv::circle(frame, cv::Point(cx, cy), 2, hud_green_, cv::FILLED);

    // Corner brackets
    int bsz = 15;
    // Top-left
    cv::line(frame, cv::Point(cx - sz, cy - sz), cv::Point(cx - sz + bsz, cy - sz), hud_green_, 1);
    cv::line(frame, cv::Point(cx - sz, cy - sz), cv::Point(cx - sz, cy - sz + bsz), hud_green_, 1);
    // Top-right
    cv::line(frame, cv::Point(cx + sz, cy - sz), cv::Point(cx + sz - bsz, cy - sz), hud_green_, 1);
    cv::line(frame, cv::Point(cx + sz, cy - sz), cv::Point(cx + sz, cy - sz + bsz), hud_green_, 1);
    // Bottom-left
    cv::line(frame, cv::Point(cx - sz, cy + sz), cv::Point(cx - sz + bsz, cy + sz), hud_green_, 1);
    cv::line(frame, cv::Point(cx - sz, cy + sz), cv::Point(cx - sz, cy + sz - bsz), hud_green_, 1);
    // Bottom-right
    cv::line(frame, cv::Point(cx + sz, cy + sz), cv::Point(cx + sz - bsz, cy + sz), hud_green_, 1);
    cv::line(frame, cv::Point(cx + sz, cy + sz), cv::Point(cx + sz, cy + sz - bsz), hud_green_, 1);
}

void MilitaryHUD::draw_detections(cv::Mat& frame, const std::vector<Track>& tracks,
                                   const std::vector<ThreatInfo>& threats) {
    // Build threat map by track_id
    std::unordered_map<int, const ThreatInfo*> threat_map;
    for (auto& t : threats) threat_map[t.track_id] = &t;

    for (auto& track : tracks) {
        cv::Scalar color = hud_green_;
        std::string threat_str = "";

        auto it = threat_map.find(track.id);
        if (it != threat_map.end()) {
            color = it->second->color;
            threat_str = " " + ThreatAssessor::level_name(it->second->level);
        }

        // Draw bounding box with corner markers
        cv::Rect r(
            (int)track.bbox.x, (int)track.bbox.y,
            (int)track.bbox.width, (int)track.bbox.height
        );

        // Clamp to frame
        r &= cv::Rect(0, 0, frame.cols, frame.rows);
        if (r.width <= 0 || r.height <= 0) continue;

        // Corner markers instead of full rectangle
        int csz = std::min(15, std::min(r.width, r.height) / 3);
        // Top-left
        cv::line(frame, r.tl(), cv::Point(r.x + csz, r.y), color, 2);
        cv::line(frame, r.tl(), cv::Point(r.x, r.y + csz), color, 2);
        // Top-right
        cv::line(frame, cv::Point(r.x + r.width, r.y), cv::Point(r.x + r.width - csz, r.y), color, 2);
        cv::line(frame, cv::Point(r.x + r.width, r.y), cv::Point(r.x + r.width, r.y + csz), color, 2);
        // Bottom-left
        cv::line(frame, cv::Point(r.x, r.y + r.height), cv::Point(r.x + csz, r.y + r.height), color, 2);
        cv::line(frame, cv::Point(r.x, r.y + r.height), cv::Point(r.x, r.y + r.height - csz), color, 2);
        // Bottom-right
        cv::line(frame, r.br(), cv::Point(r.x + r.width - csz, r.y + r.height), color, 2);
        cv::line(frame, r.br(), cv::Point(r.x + r.width, r.y + r.height - csz), color, 2);

        // Label: "ID:CLASS CONF THREAT"
        std::ostringstream label;
        label << "T" << track.id << ":" << track.class_name
              << " " << std::fixed << std::setprecision(0) << (track.confidence * 100) << "%"
              << threat_str;
        draw_label(frame, label.str(), cv::Point(r.x, r.y - 5), color, 0.4, 1);

        // Speed/heading info below box
        if (track.speed > 2.0f) {
            std::ostringstream info;
            info << std::fixed << std::setprecision(0)
                 << track.speed << "px/f " << track.heading << "deg";
            draw_label(frame, info.str(), cv::Point(r.x, r.y + r.height + 15), hud_cyan_, 0.35, 1);
        }
    }
}

void MilitaryHUD::draw_minimap(cv::Mat& frame, const std::vector<Track>& tracks,
                                const std::vector<ThreatInfo>& threats) {
    int mw = 150, mh = 150;
    int mx = frame.cols - mw - 10, my = 10;

    // Semi-transparent background
    cv::Mat roi = frame(cv::Rect(mx, my, mw, mh));
    cv::Mat overlay(mh, mw, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::addWeighted(roi, 0.5, overlay, 0.5, 0, roi);

    // Border
    cv::rectangle(frame, cv::Rect(mx, my, mw, mh), hud_green_, 1);

    // Draw center marker (self)
    int cmx = mx + mw / 2, cmy = my + mh / 2;
    cv::drawMarker(frame, cv::Point(cmx, cmy), hud_cyan_,
                   cv::MARKER_DIAMOND, 8, 2);

    // Build threat map
    std::unordered_map<int, const ThreatInfo*> tmap;
    for (auto& t : threats) tmap[t.track_id] = &t;

    // Draw targets as dots
    for (auto& track : tracks) {
        // Map track center to minimap coords
        float nx = track.center.x / frame.cols;
        float ny = track.center.y / frame.rows;
        int px = mx + (int)(nx * mw);
        int py = my + (int)(ny * mh);

        cv::Scalar col = hud_green_;
        auto it = tmap.find(track.id);
        if (it != tmap.end()) col = it->second->color;

        cv::circle(frame, cv::Point(px, py), 3, col, cv::FILLED);
    }

    // Label
    draw_label(frame, "MINIMAP", cv::Point(mx + 2, my + 12), hud_green_, 0.35, 1);
}

void MilitaryHUD::draw_threat_bar(cv::Mat& frame, ThreatLevel overall) {
    int bw = 200, bh = 20;
    int bx = 10, by = frame.rows - bh - 10;

    // Background
    cv::rectangle(frame, cv::Rect(bx, by, bw, bh), cv::Scalar(40, 40, 40), cv::FILLED);
    cv::rectangle(frame, cv::Rect(bx, by, bw, bh), hud_green_, 1);

    // Fill based on level
    float fill = 0;
    cv::Scalar fill_color;
    switch (overall) {
        case ThreatLevel::LOW:      fill = 0.25f; fill_color = cv::Scalar(0, 200, 0); break;
        case ThreatLevel::MEDIUM:   fill = 0.5f;  fill_color = cv::Scalar(0, 255, 255); break;
        case ThreatLevel::HIGH:     fill = 0.75f; fill_color = cv::Scalar(0, 165, 255); break;
        case ThreatLevel::CRITICAL: fill = 1.0f;  fill_color = cv::Scalar(0, 0, 255); break;
    }

    int fw = (int)(fill * (bw - 4));
    cv::rectangle(frame, cv::Rect(bx + 2, by + 2, fw, bh - 4), fill_color, cv::FILLED);

    // Label
    std::string name = "THREAT: " + ThreatAssessor::level_name(overall);
    draw_label(frame, name, cv::Point(bx, by - 5), fill_color, 0.4, 1);
}

void MilitaryHUD::draw_trails(cv::Mat& frame, const std::vector<Track>& tracks) {
    for (auto& track : tracks) {
        if (track.trail.size() < 2) continue;

        for (size_t i = 1; i < track.trail.size(); i++) {
            float alpha = (float)i / track.trail.size();
            int intensity = (int)(alpha * 255);
            cv::Scalar color(0, intensity, 0);
            cv::line(frame,
                     cv::Point((int)track.trail[i - 1].x, (int)track.trail[i - 1].y),
                     cv::Point((int)track.trail[i].x, (int)track.trail[i].y),
                     color, 1);
        }
    }
}

void MilitaryHUD::draw_info_panel(cv::Mat& frame, float fps, int target_count,
                                   int frame_number) {
    int px = 10, py = 10;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_);
    int mins = elapsed.count() / 60;
    int secs = elapsed.count() % 60;

    // Semi-transparent panel
    cv::Mat roi = frame(cv::Rect(px, py, 180, 80));
    cv::Mat overlay(80, 180, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::addWeighted(roi, 0.5, overlay, 0.5, 0, roi);
    cv::rectangle(frame, cv::Rect(px, py, 180, 80), hud_green_, 1);

    std::ostringstream ss;
    ss << "TANK VISION AI v3";
    draw_label(frame, ss.str(), cv::Point(px + 5, py + 15), hud_green_, 0.4, 1);

    ss.str(""); ss << "FPS: " << std::fixed << std::setprecision(0) << fps;
    draw_label(frame, ss.str(), cv::Point(px + 5, py + 33), hud_cyan_, 0.4, 1);

    ss.str(""); ss << "TARGETS: " << target_count;
    draw_label(frame, ss.str(), cv::Point(px + 5, py + 51), hud_yellow_, 0.4, 1);

    ss.str(""); ss << std::setw(2) << std::setfill('0') << mins << ":"
                    << std::setw(2) << std::setfill('0') << secs
                    << " | F" << frame_number;
    draw_label(frame, ss.str(), cv::Point(px + 5, py + 69), hud_white_, 0.35, 1);
}

void MilitaryHUD::draw_compass(cv::Mat& frame) {
    int cx = frame.cols / 2, cy = 30;
    int w = 100;

    // Simple compass bar
    cv::line(frame, cv::Point(cx - w, cy), cv::Point(cx + w, cy), hud_green_, 1);
    cv::line(frame, cv::Point(cx, cy - 5), cv::Point(cx, cy + 5), hud_green_, 2);

    // Cardinal directions
    draw_label(frame, "N", cv::Point(cx - 4, cy - 8), hud_green_, 0.4, 1);
    draw_label(frame, "W", cv::Point(cx - w - 5, cy + 4), hud_green_, 0.3, 1);
    draw_label(frame, "E", cv::Point(cx + w - 3, cy + 4), hud_green_, 0.3, 1);
}

void MilitaryHUD::draw(cv::Mat& frame,
                        const std::vector<Track>& tracks,
                        const std::vector<ThreatInfo>& threats,
                        float fps, int frame_number) {
    ThreatLevel overall = ThreatLevel::LOW;
    for (auto& t : threats) {
        if (t.level > overall) overall = t.level;
    }

    draw_trails(frame, tracks);
    draw_detections(frame, tracks, threats);
    draw_crosshair(frame);
    draw_minimap(frame, tracks, threats);
    draw_threat_bar(frame, overall);
    draw_info_panel(frame, fps, (int)tracks.size(), frame_number);
    draw_compass(frame);

    // Scan lines effect (subtle)
    for (int y = 0; y < frame.rows; y += 3) {
        cv::Mat row = frame.row(y);
        row *= 0.95;
    }
}

} // namespace tv
