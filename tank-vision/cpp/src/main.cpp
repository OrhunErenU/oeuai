#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "detector.h"
#include "tracker.h"
#include "threat.h"
#include "hud.h"
#include "depth.h"

void print_usage() {
    std::cout << "Tank Vision AI — C++ Inference Engine (ONNX Runtime)\n"
              << "Usage:\n"
              << "  tank_vision --onnx <model.onnx> --source <video|camera_id>\n"
              << "\nOptions:\n"
              << "  --onnx      ONNX model file (.onnx) [REQUIRED]\n"
              << "  --source    Video file path or camera index (0 for webcam)\n"
              << "  --conf      Confidence threshold (default: 0.35)\n"
              << "  --iou       IoU threshold for NMS (default: 0.45)\n"
              << "  --save      Output video path (optional)\n"
              << "  --no-hud    Disable HUD overlay\n"
              << "  --cpu       Force CPU inference\n"
              << std::endl;
}

int main(int argc, char** argv) {
    std::string onnx_path, source = "0";
    std::string save_path;
    float conf_thresh = 0.35f, iou_thresh = 0.45f;
    bool show_hud = true;
    bool use_gpu = true;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--onnx" && i + 1 < argc) onnx_path = argv[++i];
        else if (arg == "--source" && i + 1 < argc) source = argv[++i];
        else if (arg == "--conf" && i + 1 < argc) conf_thresh = std::stof(argv[++i]);
        else if (arg == "--iou" && i + 1 < argc) iou_thresh = std::stof(argv[++i]);
        else if (arg == "--save" && i + 1 < argc) save_path = argv[++i];
        else if (arg == "--no-hud") show_hud = false;
        else if (arg == "--cpu") use_gpu = false;
        else if (arg == "--help" || arg == "-h") { print_usage(); return 0; }
    }

    if (onnx_path.empty()) {
        std::cerr << "Error: Must provide --onnx <model.onnx>\n";
        print_usage();
        return 1;
    }

    std::cout << "========================================\n"
              << "  TANK VISION AI — C++ ONNX Runtime\n"
              << "========================================\n";

    // Initialize detector
    tv::Detector detector;
    if (!detector.load(onnx_path, use_gpu)) {
        std::cerr << "Failed to load model: " << onnx_path << std::endl;
        return 1;
    }

    // Initialize tracker
    tv::ByteTracker tracker(0.5f, 0.1f, 30, 100);

    // Initialize threat assessor
    tv::ThreatAssessor threat_assessor;

    // Initialize HUD
    tv::MilitaryHUD hud;

    // Open video source
    cv::VideoCapture cap;
    bool is_camera = false;
    try {
        int cam_id = std::stoi(source);
        cap.open(cam_id);
        is_camera = true;
    } catch (...) {
        cap.open(source);
    }

    if (!cap.isOpened()) {
        std::cerr << "Failed to open: " << source << std::endl;
        return 1;
    }

    int frame_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double video_fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Source: " << source << " (" << frame_w << "x" << frame_h
              << " @ " << video_fps << " fps)\n";

    // Video writer
    cv::VideoWriter writer;
    if (!save_path.empty()) {
        writer.open(save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    video_fps > 0 ? video_fps : 30.0, cv::Size(frame_w, frame_h));
    }

    // Main loop
    cv::Mat frame;
    int frame_num = 0;
    float fps = 0;
    auto fps_start = std::chrono::steady_clock::now();
    int fps_frames = 0;

    std::cout << "\nRunning... Press 'q' to quit, 's' for screenshot\n\n";

    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            if (!is_camera) {
                std::cout << "Video ended.\n";
                break;
            }
            continue;
        }
        frame_num++;

        auto t1 = std::chrono::high_resolution_clock::now();

        // Detect
        auto detections = detector.detect(frame, conf_thresh, iou_thresh);

        // Track
        auto tracks = tracker.update(detections);

        // Assess threats
        auto threats = threat_assessor.assess_all(tracks, frame_w, frame_h);

        auto t2 = std::chrono::high_resolution_clock::now();
        float inference_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();

        // FPS (rolling average)
        fps_frames++;
        auto fps_now = std::chrono::steady_clock::now();
        float fps_elapsed = std::chrono::duration<float>(fps_now - fps_start).count();
        if (fps_elapsed >= 1.0f) {
            fps = fps_frames / fps_elapsed;
            fps_frames = 0;
            fps_start = fps_now;
        }

        // Draw HUD
        if (show_hud) {
            hud.draw(frame, tracks, threats, fps, frame_num);
        }

        cv::imshow("Tank Vision AI", frame);
        if (writer.isOpened()) writer.write(frame);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;
        if (key == 's') {
            std::string screenshot = "screenshot_" + std::to_string(frame_num) + ".jpg";
            cv::imwrite(screenshot, frame);
            std::cout << "Screenshot: " << screenshot << std::endl;
        }

        if (frame_num % 100 == 0) {
            std::cout << "Frame " << frame_num
                      << " | FPS: " << std::fixed << std::setprecision(1) << fps
                      << " | Inference: " << std::setprecision(1) << inference_ms << "ms"
                      << " | Targets: " << tracks.size()
                      << " | Threat: " << tv::ThreatAssessor::level_name(
                             threat_assessor.overall_level(threats))
                      << std::endl;
        }
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "\nTotal frames: " << frame_num << std::endl;
    return 0;
}
