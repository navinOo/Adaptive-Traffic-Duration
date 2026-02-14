#ifndef TRAFFIC_STREAM_H
#define TRAFFIC_STREAM_H

#include <opencv2/opencv.hpp>
#include <vector>

class TrafficStream {
private:
    int id;
    double roadWidthMeters;
    
    // CALIBRATION DATA (Your Role)
    std::vector<cv::Point> roi; 
    cv::Point dividerAnchor;

public:
    // Constructor
    TrafficStream(int streamId, double width);

    // LOGIC: The Geometric Normalization Math
    // Partner calls this with their MOG2 Mask
    double calculatePressure(cv::Mat binaryMask);

    // LOGIC: Converts Pressure (0.0-1.0) to Green Time (Seconds)
    int getGreenTime(double pressure);

    // SETUP: For the Mouse Callback
    void setROI(std::vector<cv::Point> points);
};

#endif