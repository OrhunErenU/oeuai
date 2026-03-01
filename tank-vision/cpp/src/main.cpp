#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "detector.h"
#include "tracker.h"
#include "threat.h"
#include "hud.h"
#include "depth.h"

void print_usage() {
    std::cout << "Tank Vision AI — C++ Inference Engine\n"
              << "Usage:\n"
              << "  tank_vision --engine <model.engine> --source <video|camera_id>\n"
              << "\nOptions:\n"
              << "  --engine    TensorRT engine file (.engine)\n"
              << "  --onnx      ONNX model file (builds engine if --engine not found)\n"
              << "  --source    Video file path or camera index (0 for webcam)\n"
              << "  --conf      Confidence threshold (default: 0.3)\n"
              << "  --iou       IoU threshold for NMS (default: 0.45)\n"
              << "  --depth     MiDaS depth model path (optional)\n"
              << "  --save      Output video path (optional)\n"
              << "  --no-hud    Disable HUD overlay\n"
              << std::endl;
}

int main(int argc, char** argv) {
    std::string engine_path, onnx_path, source = "0";
    std::string depth_model_path, save_path;
    float conf_thresh = 0.3f, iou_thresh = 0.45f;
    bool show_hud = true;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) engine_path = argv[++i];
        else if (arg == "--onnx" && i + 1 < argc) onnx_path = argv[++i];
        else if (arg == "--source" && i + 1 < argc) source = argv[++i];
        else if (arg == "--conf" && i + 1 < argc) conf_thresh = std::stof(argv[++i]);
        else if (arg == "--iou" && i + 1 < argc) iou_thresh = std::stof(argv[++i]);
        else if (arg == "--depth" && i + 1 < argc) depth_model_path = argv[++i];
        else if (arg == "--save" && i + 1 < argc) save_path = argv[++i];
        else if (arg == "--no-hud") show_hud = false;
        else if (arg == "--help" || arg == "-h") { print_usage(); return 0; }
    }

    if (engine_path.empty() && onnx_path.empty()) {
        std::cerr << "Error: Must provide --engine or --onnx\n";
        print_usage();
        return 1;
    }

    std::cout << "========================================\n"
              << "  TANK VISION AI — C++ Engine v3\n"
              << "========================================\n";

    // Initialize detector
    tv::Detector detector;
    if (!engine_path.empty()) {
        if (!detector.load_engine(engine_path)) {
            if (!onnx_path.empty()) {
                std::cout << "Engine not found, building from ONNX...\n";
                if (!detector.build_engine(onnx_path, engine_path, true)) {
                    std::cerr << "Failed to build engine\n";
                    return 1;
                }
            } else {
                std::cerr << "Failed to load engine\n";
                return 1;
            }
        }
    } else {
        engine_path = onnx_path + ".engine";
        if (!detector.build_engine(onnx_path, engine_path, true)) {
            std::cerr << "Failed to build engine from ONNX\n";
            return 1;
        }
    }

    // Initialize tracker
    tv::ByteTracker tracker(0.5f, 0.1f, 30, 100);

    // Initialize threat assessor
    tv::ThreatAssessor threat_assessor;

    // Initialize HUD
    tv::MilitaryHUD hud;

    // Initialize depth estimator (optional)
    tv::DepthEstimator depth_est;
    if (!depth_model_path.empty()) {
        depth_est.load(depth_model_path);
    }

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

        // FPS calculation (rolling average over 1 second)
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

        // Show
        cv::imshow("Tank Vision AI", frame);
        if (writer.isOpened()) writer.write(frame);

        // Key handling
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break; // q or ESC
        if (key == 's') {
            std::string screenshot = "screenshot_" + std::to_string(frame_num) + ".jpg";
            cv::imwrite(screenshot, frame);
            std::cout << "Screenshot saved: " << screenshot << std::endl;
        }

        // Print stats periodically
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

    std::cout << "\nTotal frames processed: " << frame_num << std::endl;
    return 0;
}
