#include <opencv2/opencv.hpp>
#include "../include/TrafficStream.h" // Links to your header in the include folder

int main() {
    // 1. INITIALIZATION
    cv::VideoCapture cap("traffic_sample.mp4");
    if(!cap.isOpened()) return -1;

    // Create a Stream Object (7.5m is the real road width for normalization)
    TrafficStream lane1(1, 7.5); 

    // 2. CALIBRATION PHASE (Static)
    // Here we would call your MouseCallback to set lane1.roi and lane1.divider
    // For now, we assume the user has clicked and points are stored.

    cv::Mat frame, grayFrame, foregroundMask;
    
    // 3. THE PERCEPTION & LOGIC LOOP
    while (cap.read(frame)) {
        
        // --- PARTNER'S SECTION: PERCEPTION (MOG2) ---
        // Your partner will insert their MOG2 processing here to create 'foregroundMask'
        // Example: partnerMOG->apply(frame, foregroundMask);

        // --- YOUR SECTION: LOGIC (GEOMETRIC NORMALIZATION) ---
        // We pass the partner's mask into your math engine
        if (!foregroundMask.empty()) {
            double pressure = lane1.calculatePressure(foregroundMask);
            
            // REPORT LOGIC: Convert Pressure to Timing
            int greenTime = lane1.getGreenTime(pressure);

            // Display results on screen
            std::string status = "Pressure: " + std::to_string(pressure * 100) + "%";
            cv::putText(frame, status, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("NASA-Grade Traffic Logic", frame);
        if (cv::waitKey(30) == 27) break; // Exit on ESC
    }

    return 0;
}