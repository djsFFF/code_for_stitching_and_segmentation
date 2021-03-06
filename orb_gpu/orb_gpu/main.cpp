#include <iostream>
#include <fstream>
#include <string>
#include <Python.h>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cuda;
using namespace cv::detail;

float match_conf = 0.65f;
//ORB参数
Size grid_size = Size(3, 1);
int nfeatures = 1500;
float scaleFactor = 1.3f;
//默认为5，修改为1
int nlevels = 1;

void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);
void CalcCorners(const Mat& H, const Mat& src);

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;
four_corners_t corners;

int stitching()
{
	GpuMat image_gpu_1, image_gpu_2, gray_image_gpu_1, gray_image_gpu_2;
	GpuMat keypoints_gpu_1, keypoints_gpu_2, descriptors_gpu_1, descriptors_gpu_2, descriptors_gpu_1_32F, descriptors_gpu_2_32F;

	vector<Mat> images;

	//600*800
	images.push_back(imread("image\\monitor7-1.jpg"));
	images.push_back(imread("image\\monitor7-2.jpg"));


	////1280*720
	//images.push_back(imread("image\\image1.jpg"));
	//images.push_back(imread("image\\image2.jpg"));
	//images.push_back(imread("image\\image1.jpg"));
	//images.push_back(imread("image\\image2.jpg"));
	//images.push_back(imread("image\\image1.jpg"));
	//images.push_back(imread("image\\image2.jpg"));
	//images.push_back(imread("image\\image1.jpg"));
	//images.push_back(imread("image\\image2.jpg"));
	//images.push_back(imread("image\\image1.jpg"));
	//images.push_back(imread("image\\image2.jpg"));

	for (int n = 0; n < images.size(); n += 2)
	{
		int num = n / 2 + 1;
		UMat descriptors_1, descriptors_2;
		vector<KeyPoint> keypoints_1, keypoints_2;
		vector<DMatch> all_matches, good_matches;

		cout << "开始拼接第" << num << "张" << endl;
		int64 app_start_time = getTickCount();

		//寻找特征点
		cout << "特征点提取" << endl;
		int64 t = getTickCount();
		Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(nfeatures * (99 + grid_size.area()) / 100 / grid_size.area(), scaleFactor, nlevels, false);
		//GPU灰度图转化
		image_gpu_1.upload(images[n]);
		image_gpu_2.upload(images[n + 1]);
		cv::cuda::cvtColor(image_gpu_1, gray_image_gpu_1, COLOR_BGR2GRAY);
		cv::cuda::cvtColor(image_gpu_2, gray_image_gpu_2, COLOR_BGR2GRAY);
		//检测特征点及描述
		orb->detectAndComputeAsync(gray_image_gpu_1, cuda::GpuMat(), keypoints_gpu_1, descriptors_gpu_1);
		orb->convert(keypoints_gpu_1, keypoints_1);
		descriptors_gpu_1.convertTo(descriptors_gpu_1_32F, CV_32F);

		orb->detectAndComputeAsync(gray_image_gpu_2, cuda::GpuMat(), keypoints_gpu_2, descriptors_gpu_2);
		orb->convert(keypoints_gpu_2, keypoints_2);
		descriptors_gpu_2.convertTo(descriptors_gpu_2_32F, CV_32F);

		//释放内存
		image_gpu_1.release();
		image_gpu_2.release();
		gray_image_gpu_1.release();
		gray_image_gpu_2.release();
		keypoints_gpu_1.release();
		keypoints_gpu_2.release();
		descriptors_gpu_1.release();
		descriptors_gpu_2.release();
		orb.release();
		cout << "特征点提取耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;

		//特征点匹配
		cout << "特征点匹配" << endl;
		t = getTickCount();
		Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L1);
		matcher->match(descriptors_gpu_1_32F, descriptors_gpu_2_32F, all_matches);
		int sz = all_matches.size();
		double max_dist = 0, min_dist = 100;

		for (int i = 0; i < sz; i++)
		{
			double dist = all_matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}
		for (int i = 0; i < sz; i++)
		{
			if (all_matches[i].distance < 0.6 * max_dist)
			{
				good_matches.push_back(all_matches[i]);
			}
		}
		descriptors_gpu_1_32F.release();
		descriptors_gpu_2_32F.release();
		matcher.release();
		cout << "特征点匹配耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;

		//存储特征点匹配图像
		Mat match_image;
		drawMatches(images[n], keypoints_1, images[n+1], keypoints_2, good_matches, match_image);
		string match_name = "match_image" + to_string(num) + ".jpg";
		imwrite(match_name, match_image);

		//图像配准
		cout << "图像配准" << endl;
		t = getTickCount();
		Mat src_points(1, static_cast<int>(good_matches.size()), CV_32FC2);
		Mat dst_points(1, static_cast<int>(good_matches.size()), CV_32FC2);
		for (size_t i = 0; i < good_matches.size(); i++)
		{
			const DMatch& m = good_matches[i];

			Point2f p = keypoints_2[m.trainIdx].pt;
			src_points.at<Point2f>(0, static_cast<int>(i)) = p;

			p = keypoints_1[m.queryIdx].pt;
			dst_points.at<Point2f>(0, static_cast<int>(i)) = p;
		}


		Mat H = findHomography(src_points, dst_points, RANSAC);

		//计算配准图的四个顶点坐标
		CalcCorners(H, images[n + 1]);

		//图像配准  
		Mat imageTransform1, imageTransform2;
		cv::warpPerspective(images[n + 1], imageTransform2, H, Size(MAX(corners.right_top.x, corners.right_bottom.x), images[n].rows));
		//imshow("直接经过透视矩阵变换", imageTransform2);
		//waitKey();
		//存储配准图像
		//imwrite("trans_pic.jpg", imageTransform2);
		cout << "图像配准耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;


		//图像拼接
		cout << "图像拼接" << endl;
		t = getTickCount();
		//创建拼接后的图,需提前计算图的大小
		int dst_width = imageTransform2.cols;
		int dst_height = images[n].rows;

		Mat dst(dst_height, dst_width, CV_8UC3);
		dst.setTo(0);

		imageTransform2.copyTo(dst(Rect(0, 0, imageTransform2.cols, imageTransform2.rows)));
		images[n].copyTo(dst(Rect(0, 0, images[n].cols, images[n].rows)));
		OptimizeSeam(images[n], imageTransform2, dst);
		//存储拼接完成的图像
		string file_name = "dst" + to_string(num) + ".jpg";
		imwrite(file_name, dst);
		cout << "图像拼接耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;
		cout << "总耗时: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec\n\n" << endl;
	}
	return 1;
}

int main(void)
{
	stitching();
	return 0;
}

void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

	V1 = H * V2;
	//左上角(0,0,1)
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

}

//优化两图的连接处，使得拼接自然
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界  

	double processWidth = img1.cols - start;//重叠区域的宽度  
	int rows = dst.rows;
	int cols = img1.cols; //注意，是列数*通道数
	double alpha = 1;//img1中像素的权重  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
				alpha = (processWidth - (j - start)) / processWidth;
			}

			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}
	}

}