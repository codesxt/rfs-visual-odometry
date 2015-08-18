/**
* This file is part of RFS-VISUAL-ODOMETRY.
*
* This implementation is based on the paper "Single Camera Visual
* Odometry Based on Random Finite Set Statistics" by Feizhu Zhang,
* Hauke Stähle, Andre Gaschler, Christian Buckl and Alois Knoll.
*
* RFS-VISUAL-ODOMETRY is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* RFS-VISUAL-ODOMETRY is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with RFS-VISUAL-ODOMETRY. If not, see <http://www.gnu.org/licenses/>.
*/

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#define MIN_NUM_FEAT 2000

using namespace cv;
using namespace std;

void featureDetection(Mat img, vector<Point2f>& points)	{
  vector<KeyPoint> keypoints;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  FAST(img, keypoints, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints, points, vector<int>());
}

void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{
  vector<float> err;
  Size winSize=Size(21,21);
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++){
		Point2f pt = points2.at(i- indexCorrection);
    if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0)){
			if((pt.x<0)||(pt.y<0))	{
				status.at(i) = 0;
			}
			points1.erase (points1.begin() + i - indexCorrection);
			points2.erase (points2.begin() + i - indexCorrection);
			indexCorrection++;
		}
	}
}

int main(int argc, char **argv){
	cout << "Visual Odometry" << endl;
	Mat frame, p_frame;
	Mat frame_c, p_frame_c;
	Mat R_f, t_f;
	vector<Point2f> keypoints_1, keypoints_2;
	double scale = 1.00;

	VideoCapture cap(0);
	if(!cap.isOpened())
		return -1;
	namedWindow("Camera", WINDOW_AUTOSIZE);

  Mat E, R, t, mask;
  double focal = 718.8560;
  cv::Point2d pp(frame.size().width/2, frame.size().height/2);

  vector<Point2f> prevFeatures;
  vector<Point2f> currFeatures;
	while(1){
		cap >> frame_c;
		if(!p_frame_c.empty()){
			//Convert images to grayscale for faster computation
			cvtColor(frame_c, frame, COLOR_BGR2GRAY);
			cvtColor(p_frame_c, p_frame, COLOR_BGR2GRAY);
			//Features of each image are stored in keypoints_i
			featureDetection(frame, keypoints_1);
			featureDetection(p_frame, keypoints_2);
			imshow("Camera", frame);
			//Features are tracked from p_frame to frame
			vector<uchar> status;
			featureTracking(p_frame,frame,keypoints_2,keypoints_1, status);
      prevFeatures = keypoints_2;
      currFeatures = keypoints_1;
			cout << "Tracked points: " << keypoints_2.size() << endl;
			if(keypoints_2.size() >= 5 && keypoints_1.size() >= 5){
				E = findEssentialMat(keypoints_1, keypoints_2, focal, pp, RANSAC, 0.999, 1.0, mask);
        recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);
				cout << "Essential Matrix: " << endl << E << endl;
        cout << "Rotation Matrix: " << endl << R << endl;
        cout << "Translation Matrix: " << endl << t << endl;
        R_f = R.clone();
        t_f = t.clone();

  			Mat prevPts(2,keypoints_2.size(), CV_64F);
  			Mat currPts(2,keypoints_1.size(), CV_64F);

  			for(int i=0;i<keypoints_2.size();i++)	{   //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
  	  		prevPts.at<double>(0,i) = keypoints_2.at(i).x;
  	  		prevPts.at<double>(1,i) = keypoints_2.at(i).y;

  	  		currPts.at<double>(0,i) = keypoints_2.at(i).x;
  	  		currPts.at<double>(1,i) = keypoints_2.at(i).y;
  	    }

  			scale = 1.00;
  	    if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1))) {
  	      t_f = t_f + scale*(R_f*t);
  	      R_f = R*R_f;
  	    }

        cout << "Final Rotation Matrix: " << endl << R_f << endl;
        cout << "Final Translation Matrix: " << endl << t_f << endl;
      }
		}
		p_frame_c = Mat(frame_c);
		if(waitKey(30) >= 0)
			break;
	}
	return 0;
}
