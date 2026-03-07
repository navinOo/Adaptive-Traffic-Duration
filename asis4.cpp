#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <fstream>

const int MIN_GREEN = 15;
const int MAX_GREEN = 60;
const int YELLOW_TIME = 5;
const float BASELINE = 40.0f;

// ═══════════════════════════════════════════════════════════════
// LANE CLASS
// ═══════════════════════════════════════════════════════════════

class Lane {
public:
    std::string name;
    int phaseIndex;
    
    float width_meters;
    float depth_meters;
    std::vector<cv::Point2f> manifold;
    cv::Mat homography;
    cv::Mat inverseHomography;  // NEW: Store inverse explicitly
    
    int vehicleCount;
    int bikeCount;
    int carCount;
    int busCount;
    float totalPCU;
    float pcuPerMeterWidth;
    float queueDepth;
    cv::Point2f queueLineStart;  // NEW: Store pixel coordinates
    cv::Point2f queueLineEnd;    // NEW: Store pixel coordinates
    
    cv::Mat lastDetectionFrame;
    
    Lane(std::string n, int idx) 
        : name(n), phaseIndex(idx), vehicleCount(0), bikeCount(0),
          carCount(0), busCount(0), totalPCU(0.0f), pcuPerMeterWidth(0.0f), queueDepth(0.0f) {}
    
    float calculateDensity() {
        if (width_meters > 0.01f) {
            pcuPerMeterWidth = totalPCU / width_meters;
            return pcuPerMeterWidth;
        }
        return 0.0f;
    }
    
    void resetDetection() {
        vehicleCount = 0;
        bikeCount = 0;
        carCount = 0;
        busCount = 0;
        totalPCU = 0.0f;
        pcuPerMeterWidth = 0.0f;
        queueDepth = 0.0f;
    }
};

// ═══════════════════════════════════════════════════════════════
// VEHICLE DETECTOR
// ═══════════════════════════════════════════════════════════════

class VehicleDetector {
private:
    cv::dnn::Net yolo;
    std::mutex yoloMutex;
    static VehicleDetector* instance;
    
    cv::Mat letterbox(const cv::Mat& src, int target_size) {
        float scale = std::min((float)target_size / src.cols, 
                              (float)target_size / src.rows);
        cv::Mat resized, canvas(target_size, target_size, CV_8UC3, cv::Scalar(0,0,0));
        cv::resize(src, resized, cv::Size(src.cols * scale, src.rows * scale), 
                   0, 0, cv::INTER_CUBIC);
        resized.copyTo(canvas(cv::Rect(0, 0, resized.cols, resized.rows)));
        return canvas;
    }
    
    VehicleDetector(const std::string& modelPath) {
        std::cout << "\n[SYSTEM] Loading YOLO..." << std::endl;
        yolo = cv::dnn::readNetFromONNX(modelPath);
        std::cout << "[SYSTEM] ✓ Ready\n" << std::endl;
    }
    
public:
    static VehicleDetector* getInstance(const std::string& modelPath = "") {
        if (!instance) instance = new VehicleDetector(modelPath);
        return instance;
    }
    
    void detectAndClassify(const cv::Mat& frame, Lane& lane) {
    lane.resetDetection();
    
    if (frame.empty() || lane.manifold.empty()) return;
    
    cv::Mat vizFrame = frame.clone();
    
    cv::Rect roiRect = cv::boundingRect(lane.manifold);
    roiRect &= cv::Rect(0, 0, frame.cols, frame.rows);
    cv::Mat roiRaw = frame(roiRect);
    
    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    std::vector<cv::Point2f> footPoints;  // NEW: Store foot points
    
    std::cout << "[DETECTION] " << lane.name << "..." << std::endl;
    
    for (int y = 0; y < roiRaw.rows; y += 400) {
        int tileH = std::min(640, roiRaw.rows - y);
        cv::Mat tile = roiRaw(cv::Rect(0, y, roiRaw.cols, tileH));
        
        cv::Mat lbox = letterbox(tile, 640);
        cv::Mat blob = cv::dnn::blobFromImage(lbox, 1.0/255.0, 
                                               cv::Size(640, 640), 
                                               cv::Scalar(), true, false);
        
        std::lock_guard<std::mutex> lock(yoloMutex);
        yolo.setInput(blob);
        cv::Mat prob = yolo.forward();
        
        cv::Mat rawData(prob.size[1], prob.size[2], CV_32F, prob.ptr<float>());
        cv::Mat output = rawData.t();
        
        float tileScale = std::min(640.0f / tile.cols, 640.0f / tile.rows);
        
        for (int i = 0; i < output.rows; ++i) {
            cv::Mat row = output.row(i);
            cv::Mat scores = row.colRange(4, output.cols);
            cv::Point classIdPoint;
            double score;
            cv::minMaxLoc(scores, 0, &score, 0, &classIdPoint);
            
            if (score > 0.35) {
                float cx = (row.at<float>(0) / tileScale) + roiRect.x;
                float cy = (row.at<float>(1) / tileScale) + roiRect.y + y;
                float bw = (row.at<float>(2) / tileScale);
                float bh = (row.at<float>(3) / tileScale);
                
                cv::Point2f foot(cx, cy + bh/2);
                
                if (cv::pointPolygonTest(lane.manifold, foot, false) >= 0) {
                    bboxes.push_back(cv::Rect(cx - bw/2, cy - bh/2, bw, bh));
                    confidences.push_back((float)score);
                    classIds.push_back(classIdPoint.x);
                    footPoints.push_back(foot);
                }
            }
        }
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, confidences, 0.20, 0.40, indices);
    
    // FIND FARTHEST VEHICLE
    float maxDepth = 0.0f;
    cv::Point2f farthestFoot(0, 0);
    
    for (int idx : indices) {
        cv::Rect box = bboxes[idx];
        int classId = classIds[idx];
        cv::Point2f foot = footPoints[idx];
        
        // Transform to get depth in meters
        std::vector<cv::Point2f> footPts = {foot};
        std::vector<cv::Point2f> groundPts;
        cv::perspectiveTransform(footPts, groundPts, lane.homography);
        
        float depth = groundPts[0].y;  // Depth in meters
        
        if (depth > maxDepth) {
            maxDepth = depth;
            farthestFoot = foot;  // Store pixel coordinates!
        }
        
        cv::Scalar color;
        std::string label;
        
        lane.vehicleCount++;
        
        if (classId == 3) {
            lane.bikeCount++;
            lane.totalPCU += 0.5f;
            color = cv::Scalar(255, 255, 0);
            label = "BIKE";
        }
        else if (classId == 5 || classId == 7) {
            lane.busCount++;
            lane.totalPCU += 4.0f;
            color = cv::Scalar(0, 0, 255);
            label = "BUS";
        }
        else if (classId == 2) {
            lane.carCount++;
            lane.totalPCU += 2.0f;
            color = cv::Scalar(0, 255, 0);
            label = "CAR";
        }
        else {
            lane.vehicleCount--;  // Don't count this one
            continue;
        }
        
        // Draw bounding box
        cv::rectangle(vizFrame, box, color, 2);
        cv::putText(vizFrame, label, cv::Point(box.x, box.y - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
    
    lane.queueDepth = maxDepth;
    
    // ═══════════════════════════════════════════════════════════
    // DRAW RED LINE AT FARTHEST VEHICLE (FIXED!)
    // ═══════════════════════════════════════════════════════════
    
    if (maxDepth > 0.5f && farthestFoot.x > 0 && farthestFoot.y > 0) {
        std::cout << "[QUEUE] Farthest vehicle at " << (int)maxDepth << "m, pixel Y=" << (int)farthestFoot.y << std::endl;
        
        // Get the Y coordinate of the farthest vehicle
        float lineY = farthestFoot.y;
        
        // Find left and right edges of the lane at this Y coordinate
        // We'll scan the manifold polygon
        
        std::vector<float> xIntersections;
        
        for (size_t i = 0; i < lane.manifold.size(); i++) {
            cv::Point2f p1 = lane.manifold[i];
            cv::Point2f p2 = lane.manifold[(i + 1) % lane.manifold.size()];
            
            // Check if this edge crosses lineY
            float minY = std::min(p1.y, p2.y);
            float maxY = std::max(p1.y, p2.y);
            
            if (lineY >= minY && lineY <= maxY) {
                // Calculate X intersection using linear interpolation
                float t = (lineY - p1.y) / (p2.y - p1.y + 0.0001f);
                float x = p1.x + t * (p2.x - p1.x);
                xIntersections.push_back(x);
            }
        }
        
        // We should have at least 2 intersections (left and right edges)
        if (xIntersections.size() >= 2) {
            std::sort(xIntersections.begin(), xIntersections.end());
            
            float leftX = xIntersections[0];
            float rightX = xIntersections[xIntersections.size() - 1];
            
            cv::Point2f leftPt(leftX, lineY);
            cv::Point2f rightPt(rightX, lineY);
            
            // Draw THICK RED LINE
            cv::line(vizFrame, leftPt, rightPt, cv::Scalar(0, 0, 255), 6);
            
            // Add text label
            std::string queueText = "QUEUE: " + std::to_string((int)maxDepth) + "m";
            cv::Point2f textPos((leftX + rightX) / 2 - 60, lineY - 15);
            
            // Black background for text
            cv::Size textSize = cv::getTextSize(queueText, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, nullptr);
            cv::rectangle(vizFrame, 
                         cv::Point(textPos.x - 5, textPos.y - textSize.height - 5),
                         cv::Point(textPos.x + textSize.width + 5, textPos.y + 5),
                         cv::Scalar(0, 0, 0), -1);
            
            // Red text
            cv::putText(vizFrame, queueText, textPos,
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
            
            std::cout << "[RED LINE] Drawn from (" << (int)leftX << ", " << (int)lineY 
                      << ") to (" << (int)rightX << ", " << (int)lineY << ")" << std::endl;
        } else {
            std::cout << "[WARNING] Could not find lane edges at Y=" << (int)lineY << std::endl;
        }
    }
    
    // Draw ROI boundary (yellow)
    for (size_t i = 0; i < lane.manifold.size(); i++) {
        cv::line(vizFrame, lane.manifold[i], 
                lane.manifold[(i+1) % lane.manifold.size()],
                cv::Scalar(255, 255, 0), 2);
    }
    
    // Add stats overlay
    std::string stats = lane.name + ": " + std::to_string(lane.vehicleCount) + 
                       " veh, " + std::to_string((int)lane.totalPCU) + " PCU";
    cv::putText(vizFrame, stats, cv::Point(20, 40),
               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
    
    lane.lastDetectionFrame = vizFrame.clone();
    lane.calculateDensity();
    
    std::cout << "[RESULT] " << lane.name << ": " 
              << lane.vehicleCount << " veh, "
              << (int)lane.totalPCU << " PCU, "
              << "queue=" << (int)maxDepth << "m\n" << std::endl;
    
    
                cv::imshow("AI Detection - " + lane.name, vizFrame);
    cv::waitKey(1000);  // Show for 1 second
    }
};

VehicleDetector* VehicleDetector::instance = nullptr;

// ═══════════════════════════════════════════════════════════════
// CALIBRATION
// ═══════════════════════════════════════════════════════════════

struct CalibrationData {
    std::vector<cv::Point2f> dividerPoints;
    std::vector<cv::Point2f> shoulderPoints;
    float width_meters;
    float depth_meters;
};

bool fileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void saveCalibration(const std::string& filename, const std::vector<CalibrationData>& allCal) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "laneCount" << (int)allCal.size();
    for (size_t i = 0; i < allCal.size(); i++) {
        std::string prefix = "lane_" + std::to_string(i + 1) + "_";
        fs << prefix + "divider" << allCal[i].dividerPoints;
        fs << prefix + "shoulder" << allCal[i].shoulderPoints;
        fs << prefix + "width" << allCal[i].width_meters;
        fs << prefix + "depth" << allCal[i].depth_meters;
    }
    fs.release();
    std::cout << "[SYSTEM] ✓ Calibration saved\n" << std::endl;
}

std::vector<CalibrationData> loadCalibrationFile(const std::string& filename) {
    std::vector<CalibrationData> allCal;
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) return allCal;

    int count;
    fs["laneCount"] >> count;
    for (int i = 0; i < count; i++) {
        CalibrationData cal;
        std::string prefix = "lane_" + std::to_string(i + 1) + "_";
        fs[prefix + "divider"] >> cal.dividerPoints;
        fs[prefix + "shoulder"] >> cal.shoulderPoints;
        fs[prefix + "width"] >> cal.width_meters;
        fs[prefix + "depth"] >> cal.depth_meters;
        allCal.push_back(cal);
    }
    fs.release();
    return allCal;
}

CalibrationData calibrateLane(const std::string& laneName, cv::Mat& sampleFrame) {
    CalibrationData cal;
    int clickState = 0;
    
    std::cout << "\n[CALIBRATION] " << laneName << std::endl;
    
    std::string winName = "Calibration";
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    cv::resizeWindow(winName, 1200, 800);
    
    auto mouseCallback = [](int event, int x, int y, int, void* userdata) {
        if (event != cv::EVENT_LBUTTONDOWN) return;
        auto* data = (std::pair<CalibrationData*, int*>*)userdata;
        cv::Point2f p(x, y);
        if (*data->second == 0) {
            data->first->dividerPoints.push_back(p);
            std::cout << "  Divider: (" << x << ", " << y << ")" << std::endl;
        } else {
            data->first->shoulderPoints.push_back(p);
            std::cout << "  Shoulder: (" << x << ", " << y << ")" << std::endl;
        }
    };
    
    auto callbackData = std::make_pair(&cal, &clickState);
    cv::setMouseCallback(winName, mouseCallback, &callbackData);
    
    std::cout << "Click DIVIDER (bottom→top), SPACE" << std::endl;
    
    while (true) {
        cv::Mat display = sampleFrame.clone();
        
        std::string msg = (clickState == 0) ? "DIVIDER [SPACE]" : "SHOULDER [ENTER]";
        cv::putText(display, msg, cv::Point(20, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        for(size_t i = 0; i < cal.dividerPoints.size(); i++) {
            cv::circle(display, cal.dividerPoints[i], 5, cv::Scalar(255, 0, 0), -1);
            if(i > 0) cv::line(display, cal.dividerPoints[i-1], cal.dividerPoints[i], 
                              cv::Scalar(255, 0, 0), 2);
        }
        
        for(size_t i = 0; i < cal.shoulderPoints.size(); i++) {
            cv::circle(display, cal.shoulderPoints[i], 5, cv::Scalar(0, 0, 255), -1);
            if(i > 0) cv::line(display, cal.shoulderPoints[i-1], cal.shoulderPoints[i], 
                              cv::Scalar(0, 0, 255), 2);
        }
        
        cv::imshow(winName, display);
        char key = cv::waitKey(30);
        
        if (key == 32 && cal.dividerPoints.size() > 1) {
            clickState = 1;
            std::cout << "Click SHOULDER, ENTER" << std::endl;
        }
        if (key == 13 && cal.shoulderPoints.size() > 1) break;
        if (key == 27) exit(0);
    }
    
    std::cout << "\nWidth (m): ";
    std::cin >> cal.width_meters;
    std::cout << "[INPUT] " << cal.width_meters << "m" << std::endl;
    
    std::cout << "\nCone distance (m): ";
    float refDist;
    std::cin >> refDist;
    std::cout << "[INPUT] " << refDist << "m" << std::endl;
    
    std::cout << "\nClick cone, ENTER" << std::endl;
    cv::Point2f conePoint;
    bool clicked = false;
    
    auto coneCallback = [](int event, int x, int y, int, void* userdata) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            auto* data = (std::pair<cv::Point2f*, bool*>*)userdata;
            *data->first = cv::Point2f(x, y);
            *data->second = true;
            std::cout << "  Cone: (" << x << ", " << y << ")" << std::endl;
        }
    };
    
    auto coneData = std::make_pair(&conePoint, &clicked);
    cv::setMouseCallback(winName, coneCallback, &coneData);
    
    while (true) {
        cv::Mat display = sampleFrame.clone();
        cv::putText(display, "CONE [ENTER]", cv::Point(20, 30),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        if (clicked) cv::circle(display, conePoint, 10, cv::Scalar(0, 255, 0), 3);
        cv::imshow(winName, display);
        if (cv::waitKey(30) == 13 && clicked) break;
    }
    
    std::vector<cv::Point2f> srcCorners = {
        cal.dividerPoints.front(), cal.shoulderPoints.front(),
        cal.shoulderPoints.back(), cal.dividerPoints.back()
    };
    
    std::vector<cv::Point2f> dstTemp = {
        cv::Point2f(0, 1), cv::Point2f(cal.width_meters, 1),
        cv::Point2f(cal.width_meters, 0), cv::Point2f(0, 0)
    };
    
    cv::Mat tempH = cv::getPerspectiveTransform(srcCorners, dstTemp);
    std::vector<cv::Point2f> coneTrans;
    cv::perspectiveTransform(std::vector<cv::Point2f>{conePoint}, coneTrans, tempH);
    
    cal.depth_meters = refDist / std::max(coneTrans[0].y, 0.0001f);
    
    std::cout << "[CALCULATED] Depth: " << cal.depth_meters << "m\n" << std::endl;
    
    cv::destroyWindow(winName);
    return cal;
}

// ═══════════════════════════════════════════════════════════════
// GRID DISPLAY
// ═══════════════════════════════════════════════════════════════

class GridDisplayManager {
private:
    struct LaneDisplay {
        std::string name;
        cv::Mat frame;
        int countdown;
        std::string status;
        cv::Scalar color;
        bool isActive;
    };
    
    std::vector<LaneDisplay> lanes;
    std::mutex displayMutex;
    const int tileWidth = 640;
    const int tileHeight = 480;
    
public:
    GridDisplayManager(const std::vector<std::string>& names) {
        for (const auto& name : names) {
            LaneDisplay ld;
            ld.name = name;
            ld.countdown = 0;
            ld.status = "RED";
            ld.color = cv::Scalar(0, 0, 255);
            ld.isActive = false;
            ld.frame = cv::Mat(tileHeight, tileWidth, CV_8UC3, cv::Scalar(50, 50, 50));
            lanes.push_back(ld);
        }
        
        cv::namedWindow("ASIS", cv::WINDOW_NORMAL);
        cv::resizeWindow("ASIS", 1280, 960);
    }
    
    void updateFrame(int idx, const cv::Mat& frame) {
        std::lock_guard<std::mutex> lock(displayMutex);
        if (!frame.empty()) {
            cv::resize(frame, lanes[idx].frame, cv::Size(tileWidth, tileHeight));
        }
    }
    
    void setActive(int activeIdx) {
        std::lock_guard<std::mutex> lock(displayMutex);
        for (size_t i = 0; i < lanes.size(); i++) {
            lanes[i].isActive = (i == activeIdx);
        }
    }
    
    void updateCountdown(int idx, const std::string& status, int countdown) {
        std::lock_guard<std::mutex> lock(displayMutex);
        lanes[idx].status = status;
        lanes[idx].countdown = countdown;
        
        if (status == "GREEN") lanes[idx].color = cv::Scalar(0, 255, 0);
        else if (status == "YELLOW") lanes[idx].color = cv::Scalar(0, 255, 255);
        else lanes[idx].color = cv::Scalar(0, 0, 255);
    }
    
    void render() {
        std::lock_guard<std::mutex> lock(displayMutex);
        
        cv::Mat grid(tileHeight * 2, tileWidth * 2, CV_8UC3);
        
        for (size_t i = 0; i < lanes.size(); i++) {
            cv::Mat tile = lanes[i].frame.clone();
            
            cv::putText(tile, lanes[i].name, cv::Point(15, 35),
                       cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            
            std::string statusText = lanes[i].status + ": " + std::to_string(lanes[i].countdown) + "s";
            cv::putText(tile, statusText, cv::Point(15, 70),
                       cv::FONT_HERSHEY_SIMPLEX, 0.8, lanes[i].color, 2);
            
            std::string bigNum = std::to_string(lanes[i].countdown);
            cv::Size textSize = cv::getTextSize(bigNum, cv::FONT_HERSHEY_SIMPLEX, 3, 6, nullptr);
            cv::Point textPos((tile.cols - textSize.width)/2, 
                             (tile.rows + textSize.height)/2);
            cv::putText(tile, bigNum, textPos,
                       cv::FONT_HERSHEY_SIMPLEX, 3, lanes[i].color, 6);
            
            std::string state = lanes[i].isActive ? "[LIVE]" : "[PAUSED]";
            cv::putText(tile, state, cv::Point(tile.cols - 120, 35),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                       lanes[i].isActive ? cv::Scalar(0, 255, 0) : cv::Scalar(128, 128, 128), 2);
            
            int row = (i / 2) * tileHeight;
            int col = (i % 2) * tileWidth;
            tile.copyTo(grid(cv::Rect(col, row, tileWidth, tileHeight)));
        }
        
        cv::imshow("ASIS", grid);
        cv::waitKey(1);
    }
};

// ═══════════════════════════════════════════════════════════════
// CYCLE MANAGER
// ═══════════════════════════════════════════════════════════════

class CycleManager {
private:
    std::vector<Lane> lanes;
    VehicleDetector* detector;
    std::vector<cv::VideoCapture> cameras;
    GridDisplayManager displayMgr;
    std::vector<int> waitTimes;
    std::vector<int> currentGreenTimes;
    std::vector<bool> captured;
    bool isFirstCycle;
    
public:
    CycleManager(const std::string& modelPath, const std::vector<std::string>& names) 
        : displayMgr(names), isFirstCycle(true), currentGreenTimes(4, 40), captured(4, false) {
        
        detector = VehicleDetector::getInstance(modelPath);
        
        for (size_t i = 0; i < names.size(); i++) {
            lanes.push_back(Lane(names[i], i));
            waitTimes.push_back(0);
        }
    }
    
    void loadCalibration(int idx, const CalibrationData& cal) {
        lanes[idx].width_meters = cal.width_meters;
        lanes[idx].depth_meters = cal.depth_meters;
        
        std::vector<cv::Point2f> manifold;
        for(auto& p : cal.dividerPoints) manifold.push_back(p);
        for(int i = cal.shoulderPoints.size()-1; i >= 0; i--) {
            manifold.push_back(cal.shoulderPoints[i]);
        }
        lanes[idx].manifold = manifold;
        
        std::vector<cv::Point2f> src = {
            cal.dividerPoints.front(), cal.shoulderPoints.front(),
            cal.shoulderPoints.back(), cal.dividerPoints.back()
        };
        
        std::vector<cv::Point2f> dst = {
            cv::Point2f(0, cal.depth_meters),
            cv::Point2f(cal.width_meters, cal.depth_meters),
            cv::Point2f(cal.width_meters, 0),
            cv::Point2f(0, 0)
        };
        
        lanes[idx].homography = cv::getPerspectiveTransform(src, dst);
        lanes[idx].inverseHomography = cv::getPerspectiveTransform(dst, src);
    }
    
    void loadCameras(const std::vector<std::string>& paths) {
        for (const auto& path : paths) {
            cameras.push_back(cv::VideoCapture(path));
        }
    }
    
    void initializeWaitTimes() {
        std::cout << "\n[BOOTSTRAP]" << std::endl;
        waitTimes[0] = 0;
        waitTimes[1] = 45;
        waitTimes[2] = 90;
        waitTimes[3] = 135;
        
        for (size_t i = 0; i < lanes.size(); i++) {
            std::cout << "  " << lanes[i].name << ": " << waitTimes[i] << "s" << std::endl;
        }
    }
    
    void captureLaneAt40s(int laneIdx) {
        std::cout << "\n[40s CAPTURE] " << lanes[laneIdx].name << std::endl;
        
        cv::Mat frame;
        if (!cameras[laneIdx].read(frame)) {
            cameras[laneIdx].set(cv::CAP_PROP_POS_FRAMES, 0);
            cameras[laneIdx].read(frame);
        }
        
        if (!frame.empty()) {
            detector->detectAndClassify(frame, lanes[laneIdx]);
            
            if (!lanes[laneIdx].lastDetectionFrame.empty()) {
                displayMgr.updateFrame(laneIdx, lanes[laneIdx].lastDetectionFrame);
            }
            
            captured[laneIdx] = true;
        }
    }
    
    std::vector<int> calculateGreenTimes() {
        std::cout << "\n[CALCULATION]" << std::endl;
        
        std::vector<float> rawGreens;
        
        for (size_t i = 0; i < lanes.size(); i++) {
            float density = lanes[i].pcuPerMeterWidth;
            float queueFactor = lanes[i].queueDepth / std::max(lanes[i].depth_meters, 1.0f);
            float urgency = density + (queueFactor * 5.0f);
            
            float normUrgency;
            if (waitTimes[i] > 0) {
                float rate = urgency / waitTimes[i];
                normUrgency = rate * BASELINE;
            } else {
                normUrgency = urgency;
            }
            
            float pressure = normUrgency / (normUrgency + 10.0f);
            float rawGreen = MIN_GREEN + pressure * (MAX_GREEN - MIN_GREEN);
            rawGreens.push_back(rawGreen);
            
            std::cout << "  " << lanes[i].name << ": urgency=" << urgency
                      << " → " << rawGreen << "s" << std::endl;
        }
        
        float maxRaw = *std::max_element(rawGreens.begin(), rawGreens.end());
        if (maxRaw < 0.01f) maxRaw = BASELINE;
        
        float scale = (maxRaw > BASELINE) ? (BASELINE / maxRaw) : 
                     (maxRaw < BASELINE * 0.9f) ? (BASELINE / maxRaw) : 1.0f;
        
        std::vector<int> finalGreens;
        for (size_t i = 0; i < rawGreens.size(); i++) {
            int g = std::clamp((int)(rawGreens[i] * scale), MIN_GREEN, MAX_GREEN);
            finalGreens.push_back(g);
            std::cout << "  " << lanes[i].name << ": " << g << "s FINAL" << std::endl;
        }
        
        return finalGreens;
    }
    
    void executeCycle(int cycleNum) {
        std::cout << "\n╔═══ CYCLE " << cycleNum << " ═══╗" << std::endl;
        
        if (isFirstCycle) {
            initializeWaitTimes();
            isFirstCycle = false;
        }
        
        bool allCaptured = true;
        for (bool c : captured) {
            if (!c) {
                allCaptured = false;
                break;
            }
        }
        
        if (allCaptured) {
            currentGreenTimes = calculateGreenTimes();
            std::fill(captured.begin(), captured.end(), false);
        }
        
        for (size_t phaseIdx = 0; phaseIdx < lanes.size(); phaseIdx++) {
            executePhase(phaseIdx);
        }
    }
    
private:
    void executePhase(int activeIdx) {
        int greenDuration = currentGreenTimes[activeIdx];
        
        std::cout << "\n→ " << lanes[activeIdx].name << " GREEN (" << greenDuration << "s)" << std::endl;
        
        displayMgr.setActive(activeIdx);
        
        std::vector<int> remainingWaits(4, 0);
        for (size_t i = 0; i < 4; i++) {
            if (i == activeIdx) {
                remainingWaits[i] = 0;
            } else {
                int totalWait = 0;
                size_t currentPhase = activeIdx;
                
                while (currentPhase != i) {
                    currentPhase = (currentPhase + 1) % 4;
                    totalWait += currentGreenTimes[currentPhase] + YELLOW_TIME;
                }
                
                remainingWaits[i] = totalWait;
            }
        }
        
        // GREEN PHASE
        for (int t = greenDuration; t > 0; t--) {
            cv::Mat frame;
            if (cameras[activeIdx].read(frame)) {
                displayMgr.updateFrame(activeIdx, frame);
            } else {
                cameras[activeIdx].set(cv::CAP_PROP_POS_FRAMES, 0);
            }
            
            displayMgr.updateCountdown(activeIdx, "GREEN", t);
            
            for (size_t i = 0; i < 4; i++) {
                if (i != activeIdx) {
                    int displayWait = remainingWaits[i] + t;
                    displayMgr.updateCountdown(i, "RED", displayWait);
                    waitTimes[i]++;
                    
                    if (waitTimes[i] == 40 && !captured[i]) {
                        captureLaneAt40s(i);
                    }
                }
            }
            
            displayMgr.render();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        
        waitTimes[activeIdx] = 0;
        
        // YELLOW
        for (int t = YELLOW_TIME; t > 0; t--) {
            displayMgr.updateCountdown(activeIdx, "YELLOW", t);
            
            for (size_t i = 0; i < 4; i++) {
                if (i != activeIdx) {
                    int displayWait = remainingWaits[i] + t;
                    displayMgr.updateCountdown(i, "RED", displayWait);
                    waitTimes[i]++;
                    
                    if (waitTimes[i] == 40 && !captured[i]) {
                        captureLaneAt40s(i);
                    }
                }
            }
            
            displayMgr.render();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    }
};

// ═══════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════

int main() {
    std::cout << "╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║  ASIS v4.0 - PRODUCTION FINAL          ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝\n" << std::endl;

    std::vector<std::string> laneNames = {"NORTH", "EAST", "SOUTH", "WEST"};
    std::string videoPath = "assets/trafficVideo3.mp4";
    std::string calibFile = "calibration.yml";
    
    CycleManager manager("yolov8n.onnx", laneNames);
    std::vector<CalibrationData> allLaneData;

    if (fileExists(calibFile)) {
        std::cout << "[SYSTEM] Loading calibration..." << std::endl;
        allLaneData = loadCalibrationFile(calibFile);
        
        if (allLaneData.size() != laneNames.size()) {
            allLaneData.clear();
        } else {
            std::cout << "[SYSTEM] ✓ Loaded\n" << std::endl;
        }
    }

    if (allLaneData.empty()) {
        cv::VideoCapture calibCap(videoPath);
        if (!calibCap.isOpened()) {
            std::cerr << "[ERROR] Video not found" << std::endl;
            return -1;
        }

        cv::Mat sampleFrame;
        calibCap >> sampleFrame;
        
        for (size_t i = 0; i < laneNames.size(); i++) {
            allLaneData.push_back(calibrateLane(laneNames[i], sampleFrame));
        }
        calibCap.release();
        
        saveCalibration(calibFile, allLaneData);
    }

    for (size_t i = 0; i < allLaneData.size(); i++) {
        manager.loadCalibration(i, allLaneData[i]);
    }

    manager.loadCameras({videoPath, videoPath, videoPath, videoPath});
    
    std::cout << "\n[STARTING]\n" << std::endl;
    
    for (int cycle = 0; cycle < 10; cycle++) {
        manager.executeCycle(cycle);
    }
    
    std::cout << "\n✓ DONE!" << std::endl;
    cv::waitKey(0);
    
    return 0;
}