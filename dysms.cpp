#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>

using namespace cv;
using namespace std;

#define LANE_COUNT 2
#define MAX_WAIT_TIME 40
#define DENSITY_THRESHOLD 0.05
#define SMOOTH_ALPHA 0.2

//lane info
struct LaneData {
    Rect roi;
    float rawDensity = 0;
    float smoothedDensity = 0;
    int waitTime = 0;};

int main() {
    VideoCapture cap("traffic.mp4");
    if (!cap.isOpened()) {
        cout << "Error opening video" << endl;
        return -1;
    }
    Ptr<BackgroundSubtractor> bgSub = createBackgroundSubtractorMOG2();
    vector<LaneData> lanes(LANE_COUNT);
    int currentGreen = 0;
    Mat frame;
    while (cap.read(frame)) {

        //ROAD ROI 
        int road_y = frame.rows / 3;
        int road_height = frame.rows / 2;
        Rect roadROI(0, road_y, frame.cols, road_height);
        Mat road = frame(roadROI);

        //Background Subtraction
        Mat fgMask;
        bgSub->apply(road, fgMask);
        threshold(fgMask, fgMask, 200, 255, THRESH_BINARY);
        morphologyEx(fgMask, fgMask, MORPH_CLOSE,
                     getStructuringElement(MORPH_RECT, Size(5,5)));
        //Split into Lanes
        int laneHeight = road.rows / LANE_COUNT;
        for (int i = 0; i < LANE_COUNT; i++) {
            lanes[i].roi = Rect(0, i * laneHeight,
                                road.cols, laneHeight);

            Mat laneMask = fgMask(lanes[i].roi);

            //Depth Weighting
            int zoneHeight = laneMask.rows / 3;

            Rect farZone(0, 0, laneMask.cols, zoneHeight);
            Rect midZone(0, zoneHeight, laneMask.cols, zoneHeight);
            Rect nearZone(0, 2 * zoneHeight, laneMask.cols, zoneHeight);
            float farDensity =
                (float)countNonZero(laneMask(farZone)) /
                laneMask(farZone).total();
            float midDensity =
                (float)countNonZero(laneMask(midZone)) /
                laneMask(midZone).total();
            float nearDensity =
                (float)countNonZero(laneMask(nearZone)) /
                laneMask(nearZone).total();
            lanes[i].rawDensity =
                0.2 * farDensity +
                0.3 * midDensity +
                0.5 * nearDensity;
            
            //Smoothing 
            lanes[i].smoothedDensity =
                (1 - SMOOTH_ALPHA) * lanes[i].smoothedDensity +
                SMOOTH_ALPHA * lanes[i].rawDensity;
        }
        // Dynamic Signal
        int nextGreen = currentGreen;
        float maxDensity = lanes[0].smoothedDensity;
        for (int i = 1; i < LANE_COUNT; i++) {
            if (lanes[i].smoothedDensity > maxDensity) {
                maxDensity = lanes[i].smoothedDensity;
                nextGreen = i;
            }
        }
        // Switch only if difference is significant
        if (abs(lanes[nextGreen].smoothedDensity -
                lanes[currentGreen].smoothedDensity) > DENSITY_THRESHOLD) {
            currentGreen = nextGreen;
            lanes[currentGreen].waitTime = 0;
        }
        //Starvation Protection
        for (int i = 0; i < LANE_COUNT; i++) {
            if (i != currentGreen)
                lanes[i].waitTime++;
            else
                lanes[i].waitTime = 0;

            if (lanes[i].waitTime > MAX_WAIT_TIME) {
                currentGreen = i;
                lanes[i].waitTime = 0;
            }
        }
        //Emergency Override
        bool emergencyDetected = false; 

        if (emergencyDetected) {
            currentGreen = 0; 
        }

}
cap.release();
destroyAllWindows();
return 0;}
