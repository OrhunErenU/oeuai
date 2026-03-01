#include "tracker.h"
#include <algorithm>
#include <cmath>

namespace tv {

ByteTracker::ByteTracker(float high_thresh, float low_thresh,
                           int max_lost, int max_trail)
    : high_thresh_(high_thresh), low_thresh_(low_thresh),
      max_lost_(max_lost), max_trail_(max_trail) {}

void ByteTracker::init_kalman(Track& track) {
    track.kf.init(8, 4, 0);
    cv::setIdentity(track.kf.transitionMatrix);
    track.kf.transitionMatrix.at<float>(0, 4) = 1;
    track.kf.transitionMatrix.at<float>(1, 5) = 1;
    track.kf.transitionMatrix.at<float>(2, 6) = 1;
    track.kf.transitionMatrix.at<float>(3, 7) = 1;
    cv::setIdentity(track.kf.measurementMatrix);
    cv::setIdentity(track.kf.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(track.kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    track.kf.statePost.at<float>(0) = track.center.x;
    track.kf.statePost.at<float>(1) = track.center.y;
    track.kf.statePost.at<float>(2) = track.bbox.width;
    track.kf.statePost.at<float>(3) = track.bbox.height;
    track.kf_initialized = true;
}

void ByteTracker::predict(Track& track) {
    if (!track.kf_initialized) return;
    cv::Mat pred = track.kf.predict();
    float cx = pred.at<float>(0), cy = pred.at<float>(1);
    float w = pred.at<float>(2), h = pred.at<float>(3);
    track.bbox = cv::Rect2f(cx - w / 2, cy - h / 2, w, h);
    track.center = cv::Point2f(cx, cy);
}

float ByteTracker::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    cv::Rect2f inter = a & b;
    float ia = inter.area();
    float ua = a.area() + b.area() - ia;
    return (ua > 0) ? ia / ua : 0.0f;
}

std::vector<std::pair<int, int>> ByteTracker::match(
    const std::vector<Track>& tracks,
    const std::vector<Detection>& dets,
    float iou_thresh) {
    if (tracks.empty() || dets.empty()) return {};

    struct Cand { float s; int ti, di; };
    std::vector<Cand> cands;
    for (int i = 0; i < (int)tracks.size(); i++)
        for (int j = 0; j < (int)dets.size(); j++) {
            float s = iou(tracks[i].bbox, dets[j].bbox);
            if (s > iou_thresh) cands.push_back({s, i, j});
        }
    std::sort(cands.begin(), cands.end(), [](auto& a, auto& b){ return a.s > b.s; });

    std::vector<bool> tm(tracks.size(), false), dm(dets.size(), false);
    std::vector<std::pair<int,int>> matches;
    for (auto& c : cands) {
        if (!tm[c.ti] && !dm[c.di]) {
            matches.push_back({c.ti, c.di});
            tm[c.ti] = true;
            dm[c.di] = true;
        }
    }
    return matches;
}

static void update_track_from_det(Track& t, const Detection& d, int max_trail) {
    cv::Point2f old_c = t.center;
    t.bbox = d.bbox;
    t.center = d.center;
    t.confidence = d.confidence;
    t.class_id = d.class_id;
    t.class_name = d.class_name;
    t.time_since_update = 0;
    t.age++;

    t.velocity = t.center - old_c;
    t.speed = std::sqrt(t.velocity.x * t.velocity.x + t.velocity.y * t.velocity.y);
    t.heading = std::atan2(t.velocity.x, -t.velocity.y) * 180.0f / (float)CV_PI;
    if (t.heading < 0) t.heading += 360.0f;
    t.is_approaching = (t.velocity.y > 1.0f);

    t.trail.push_back(t.center);
    if ((int)t.trail.size() > max_trail)
        t.trail.erase(t.trail.begin());

    if (t.kf_initialized) {
        cv::Mat m = (cv::Mat_<float>(4,1) << t.center.x, t.center.y, t.bbox.width, t.bbox.height);
        t.kf.correct(m);
    }
}

std::vector<Track> ByteTracker::update(const std::vector<Detection>& detections) {
    // Predict all tracks
    for (auto& t : active_tracks_) predict(t);
    for (auto& t : lost_tracks_) predict(t);

    // Split detections
    std::vector<Detection> hi, lo;
    for (auto& d : detections) {
        if (d.confidence >= high_thresh_) hi.push_back(d);
        else if (d.confidence >= low_thresh_) lo.push_back(d);
    }

    // First: active tracks vs high-conf dets
    auto m1 = match(active_tracks_, hi, 0.3f);
    std::vector<bool> at_matched(active_tracks_.size(), false);
    std::vector<bool> hd_matched(hi.size(), false);

    for (auto& [ti, di] : m1) {
        at_matched[ti] = true;
        hd_matched[di] = true;
        update_track_from_det(active_tracks_[ti], hi[di], max_trail_);
    }

    // Collect unmatched active tracks
    std::vector<Track*> unmatched_active;
    for (int i = 0; i < (int)active_tracks_.size(); i++)
        if (!at_matched[i]) unmatched_active.push_back(&active_tracks_[i]);

    // Second: unmatched active vs low-conf dets
    std::vector<Track> um_vec;
    for (auto* p : unmatched_active) um_vec.push_back(*p);
    auto m2 = match(um_vec, lo, 0.5f);
    std::vector<bool> um_matched(um_vec.size(), false);

    for (auto& [ti, di] : m2) {
        um_matched[ti] = true;
        // Find in active_tracks_ and update
        for (auto& at : active_tracks_) {
            if (at.id == um_vec[ti].id) {
                update_track_from_det(at, lo[di], max_trail_);
                break;
            }
        }
    }

    // Third: lost tracks vs unmatched high-conf dets
    std::vector<Detection> unmatched_hi;
    for (int i = 0; i < (int)hi.size(); i++)
        if (!hd_matched[i]) unmatched_hi.push_back(hi[i]);

    auto m3 = match(lost_tracks_, unmatched_hi, 0.3f);
    std::vector<bool> lt_matched(lost_tracks_.size(), false);
    std::vector<bool> uh_matched(unmatched_hi.size(), false);

    for (auto& [ti, di] : m3) {
        lt_matched[ti] = true;
        uh_matched[di] = true;
        update_track_from_det(lost_tracks_[ti], unmatched_hi[di], max_trail_);
        active_tracks_.push_back(lost_tracks_[ti]); // Recover
    }

    // Move unmatched active to lost
    for (int i = 0; i < (int)um_vec.size(); i++) {
        if (!um_matched[i]) {
            for (auto& at : active_tracks_) {
                if (at.id == um_vec[i].id) {
                    at.time_since_update++;
                    if (at.time_since_update <= max_lost_)
                        lost_tracks_.push_back(at);
                    break;
                }
            }
        }
    }

    // Create new tracks for remaining unmatched high-conf dets
    for (int i = 0; i < (int)unmatched_hi.size(); i++) {
        if (!uh_matched[i]) {
            Track t;
            t.id = next_id_++;
            t.class_id = unmatched_hi[i].class_id;
            t.class_name = unmatched_hi[i].class_name;
            t.bbox = unmatched_hi[i].bbox;
            t.center = unmatched_hi[i].center;
            t.confidence = unmatched_hi[i].confidence;
            t.velocity = cv::Point2f(0, 0);
            t.speed = 0; t.heading = 0;
            t.is_approaching = false;
            t.age = 1; t.time_since_update = 0;
            t.kf_initialized = false;
            t.trail.push_back(t.center);
            init_kalman(t);
            active_tracks_.push_back(t);
        }
    }

    // Clean up: remove stale active tracks
    active_tracks_.erase(
        std::remove_if(active_tracks_.begin(), active_tracks_.end(),
            [](const Track& t) { return t.time_since_update > 0; }),
        active_tracks_.end());

    // Clean up lost tracks
    lost_tracks_.erase(
        std::remove_if(lost_tracks_.begin(), lost_tracks_.end(),
            [&](const Track& t) { return t.time_since_update > max_lost_; }),
        lost_tracks_.end());

    // Remove duplicate IDs in lost
    std::vector<bool> lt_rm(lost_tracks_.size(), false);
    for (auto& at : active_tracks_)
        for (int i = 0; i < (int)lost_tracks_.size(); i++)
            if (lost_tracks_[i].id == at.id) lt_rm[i] = true;
    std::vector<Track> clean_lost;
    for (int i = 0; i < (int)lost_tracks_.size(); i++)
        if (!lt_rm[i]) clean_lost.push_back(lost_tracks_[i]);
    lost_tracks_ = std::move(clean_lost);

    return active_tracks_;
}

} // namespace tv
