#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/imgproc.hpp"


using namespace cv;
using namespace std;

typedef struct Image_feature 
{
	Mat descriptors;
	vector< KeyPoint > keypoint;
	Mat img;
}Image_feature;

Image_feature find_image_fature(Mat img)
{
	vector< KeyPoint > keypoint;
	Image_feature img_class;
	Mat img_g;
	Mat descriptors;

	cvtColor(img, img_g, COLOR_BGR2GRAY);

	SiftDescriptorExtractor detector;
	detector.detect(img_g, keypoint);
	SiftDescriptorExtractor extractor;
	extractor.compute(img_g, keypoint, descriptors);

	img_class.descriptors = descriptors;
	img_class.keypoint = keypoint;
	img_class.img = img;

	return img_class;
}

double find_matches_percent(Image_feature img1, Image_feature img2)
{
	
	vector < DMatch > good_matches;
	vector < DMatch > matches;

	//Flann
	FlannBasedMatcher Matcher;
	Matcher.match(img1.descriptors, img2.descriptors, matches);

	//BFMatch
	//BFMatcher matcher;
	//matcher.knnMatch(img1.descriptors, img2.descriptors, matches, 2); // Find two nearest matches

	//좋은 매칭 걸러내기 방법 1 -> Flann
	
	double dMaxDist = 0;
	double dMinDist = 100;
	double dDistance;


	for (int i = 0; i < img1.descriptors.rows; i++)
	{
		dDistance = matches[i].distance;

		if (dDistance < dMinDist) dMinDist = dDistance;
		if (dDistance > dMaxDist) dMaxDist = dDistance;

	}

	for (int i = 0; i < img1.descriptors.rows; i++)
	{
		if (matches[i].distance < 3 * dMinDist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	

	//좋은 매칭 걸러내기 방법 2 -> BFMatcher
	/*
	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = 0.7; // As in Lowe's paper; can be tuned
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
		if (i == matches.size() - 1) break;
	}
	*/

	vector<Point2f> img1_pt;
	vector<Point2f> img2_pt;

	for (int i = 0; i < good_matches.size(); i++)
	{
		img1_pt.push_back(img1.keypoint[good_matches[i].queryIdx].pt);
		img2_pt.push_back(img2.keypoint[good_matches[i].trainIdx].pt);
	}

	Mat mask;
	Mat HomoMatrix = findHomography(img2_pt, img1_pt, RANSAC, 3, mask);

	double outline_cnt = 0;
	double inline_cnt = 0;

	for (int i = 0; i < mask.rows; i++)
	{
		if (mask.at<bool>(i) == 0)
		{
			outline_cnt++;
		}
		else
		{
			inline_cnt++;
		}
	}

	double percentage = ((inline_cnt) / (inline_cnt + outline_cnt)) * 100;


	return percentage;
}

Mat panorama_stiching(Image_feature img1, Image_feature img2)
{
	
	vector < DMatch > good_matches;
	vector<vector < DMatch >> matches;
	
	//Flann
	//FlannBasedMatcher Matcher;
	//Matcher.match(img1.descriptors, img2.descriptors, matches);

	//BFMatch
	BFMatcher matcher;
	matcher.knnMatch(img1.descriptors, img2.descriptors, matches, 2); // Find two nearest matches

	//좋은 매칭 걸러내기 방법 1 -> Flann
	/*
	double dMaxDist = 0;
	double dMinDist = 100;
	double dDistance;


	for (int i = 0; i < img1.descriptors.rows; i++)
	{
		dDistance = matches[i].distance;

		if (dDistance < dMinDist) dMinDist = dDistance;
		if (dDistance > dMaxDist) dMaxDist = dDistance;

	}
	cout << "Max :" << dMaxDist << endl;
	cout << "Min :" << dMinDist << endl;

	for (int i = 0; i < img1_descriptors.rows; i++)
	{
		if (matches[i].distance < 3 * dMinDist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	*/

	//좋은 매칭 걸러내기 방법 2 -> BFMatcher

	for (int i = 0; i < matches.size(); ++i)
	{
		const float ratio = 0.7; // As in Lowe's paper; can be tuned
		if (matches[i][0].distance < ratio * matches[i][1].distance)
		{
			good_matches.push_back(matches[i][0]);
		}
		if (i == matches.size() - 1) break;
	}

	cout << "Good Match : " << good_matches.size() << endl;



	vector<Point2f> img1_pt;
	vector<Point2f> img2_pt;

	for (int i = 0; i < good_matches.size(); i++)
	{
		img1_pt.push_back(img1.keypoint[good_matches[i].queryIdx].pt);
		img2_pt.push_back(img2.keypoint[good_matches[i].trainIdx].pt);
	}

	Mat HomoMatrix = findHomography(img2_pt, img1_pt, RANSAC, 3);

	cout << HomoMatrix << endl;

	Mat matResult;
	Mat matPanorama;

	// 4개의 코너 구하기
	vector<Point2f> conerPt, conerPt_1;

	conerPt.push_back(Point2f(0, 0));
	conerPt.push_back(Point2f(img2.img.size().width, 0));
	conerPt.push_back(Point2f(0, img2.img.size().height));
	conerPt.push_back(Point2f(img2.img.size().width, img2.img.size().height));

	Mat P_Trans_conerPt;
	perspectiveTransform(Mat(conerPt), P_Trans_conerPt, HomoMatrix);

	// 이미지의 모서리 계산
	double min_x, min_y, max_x, max_y, bef_max_x;
	float min_x1, min_x2, min_y1, min_y2, max_x1, max_x2, max_y1, max_y2;

	min_x1 = min(P_Trans_conerPt.at<Point2f>(0).x, P_Trans_conerPt.at<Point2f>(1).x);
	min_x2 = min(P_Trans_conerPt.at<Point2f>(2).x, P_Trans_conerPt.at<Point2f>(3).x);
	min_y1 = min(P_Trans_conerPt.at<Point2f>(0).y, P_Trans_conerPt.at<Point2f>(1).y);
	min_y2 = min(P_Trans_conerPt.at<Point2f>(2).y, P_Trans_conerPt.at<Point2f>(3).y);
	max_x1 = max(P_Trans_conerPt.at<Point2f>(0).x, P_Trans_conerPt.at<Point2f>(1).x);
	max_x2 = max(P_Trans_conerPt.at<Point2f>(2).x, P_Trans_conerPt.at<Point2f>(3).x);
	max_y1 = max(P_Trans_conerPt.at<Point2f>(0).y, P_Trans_conerPt.at<Point2f>(1).y);
	max_y2 = max(P_Trans_conerPt.at<Point2f>(2).y, P_Trans_conerPt.at<Point2f>(3).y);
	min_x = min(min_x1, min_x2);
	min_y = min(min_y1, min_y2);
	max_x = max(max_x1, max_x2);
	max_y = max(max_y1, max_y2);

	// Transformation matrix
	Mat Htr = Mat::eye(3, 3, CV_64F);
	if (min_x < 0) 
	{
		max_x = img1.img.size().width - min_x;
		Htr.at<double>(0, 2) = -min_x;
	}
	else
	{
		if (max_x < img1.img.size().width) max_x = img1.img.size().width;
	}
	
	if (min_y < 0)
	{
		max_y = img1.img.size().height - min_y;
		Htr.at<double>(1, 2) = -min_y;
	}
	else
	{
		if (max_y < img1.img.size().height) max_y = img1.img.size().height;
	}

	// 파노라마 만들기
	matPanorama = Mat(Size(max_x, max_y), CV_32F);
	warpPerspective(img1.img, matPanorama, Htr, matPanorama.size(), INTER_CUBIC, BORDER_CONSTANT, 0);
	warpPerspective(img2.img, matPanorama, (Htr*HomoMatrix), matPanorama.size(), INTER_CUBIC, BORDER_TRANSPARENT, 0);


	return matPanorama;

}


int main()
{

	vector<Mat> panorama;
	vector<Image_feature> Image_array;
	Image_feature Image_feature;


	String folderpath = "D:/ForTA/imagestitching/imagestitching/test";
	vector<String> filenames;
	glob(folderpath, filenames);

	cout << "\n------- file load ---------\n" << endl;

	for (size_t i = 0; i < filenames.size(); i++)
	{
		panorama.push_back(imread(filenames[i], IMREAD_COLOR));
		cout << filenames[i] << "  load" << endl;
	}

	cout << "\n------- find image Feature (SIFT) ---------\n" << endl;
	for (int i = 0; i < filenames.size() ; i++)
	{
		Image_feature = find_image_fature(panorama[i]);
		Image_array.push_back(Image_feature);
		cout <<  filenames[i] << " finish "<< endl;
	}


	cout << "\n------- find Center Image ---------\n" << endl;

	vector<int> image_match_count;
	int match_count = 0;
	int bef_match_count = 0;
	int max_match = 0;

	for (int i = 0; i < filenames.size(); i++)
	{
		for (int j = 0; j < filenames.size(); j++)
		{
			if (i != j) 
			{
			if ((find_matches_percent(Image_array[i], Image_array[j])) >= 10)	match_count++;
			}
		}
		if (max_match < match_count) max_match = i;

		cout << filenames[i] << " be matching " << match_count << " images" << endl;

		match_count = 0;

	}

	cout << "\n------- image Matching ---------\n" << endl;

	Mat Panorama = imread(filenames[max_match], IMREAD_COLOR);

	for (int i = 0; i < filenames.size(); i++) 
	{
		Image_feature = find_image_fature(Panorama);

		for (int j = 0 ; j < filenames.size(); j++)
		{
			int match = 0;
			
			if (j== max_match) break;

			match = find_matches_percent(Image_feature, Image_array[j]);

			if (match >= 10)
			{
				Panorama = panorama_stiching(Image_feature, Image_array[j]);
				Image_array.erase(Image_array.begin()+j);
				break;
			}
		}
		cv::namedWindow("test", CV_WINDOW_FREERATIO);
		cv::imshow("test", Panorama);
		cv::waitKey(20);
	}
	cv::namedWindow("result", CV_WINDOW_FREERATIO);
	cv::imshow("result", Panorama);
	cv::waitKey(0);

	return 0;
}



