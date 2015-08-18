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

using namespace cv;
using namespace std;

int main(int argc, char **argv){
	cout << "RFS-VISUAL-ODOMETRY" << endl;
	VideoCapture cap(0);
	if(!cap.isOpened())
		return -1;
	Ptr<Feature2D> detector = ORB::create();
	namedWindow("Image", 1);
	Mat frame, p_frame;
	vector <KeyPoint> kp, p_kp;
	Mat desc, p_desc, imout;
	BFMatcher matcher;
	vector <DMatch> matches;
	while(1){
		cap >> frame;
		detector->detect(frame, kp);
		detector->compute(frame, kp, desc);
		if(!p_frame.empty()){
			matcher.match( desc, p_desc, matches );

			double tresholdDist = 0.25 * sqrt(double(frame.size().height*frame.size().height + frame.size().width*frame.size().width));

  		vector< DMatch > good;
      float dist_sum = 0;
      for (size_t i = 0; i < matches.size(); i++){
        dist_sum += matches[i].distance;
      }
      float th_dist = dist_sum / matches.size() * 0.6;
      for (size_t i = 0; i < matches.size(); i++){
        if(matches[i].distance < th_dist){
          good.push_back(matches[i]);
        }
      }
      cout << "Drawing " << good.size() << " matches" << endl;

			drawMatches(frame, kp, p_frame, p_kp, good, imout);
			imshow("Image", imout);
		}
		p_frame = Mat(frame);
		p_kp = kp;
		p_desc = Mat(desc);
		if(waitKey(30) >= 0)
			break;
	}
	return 0;
}
