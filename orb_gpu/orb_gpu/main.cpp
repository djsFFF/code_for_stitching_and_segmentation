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
//ORB����
Size grid_size = Size(3, 1);
int nfeatures = 1500;
float scaleFactor = 1.3f;
//Ĭ��Ϊ5���޸�Ϊ1
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

		cout << "��ʼƴ�ӵ�" << num << "��" << endl;
		int64 app_start_time = getTickCount();

		//Ѱ��������
		cout << "��������ȡ" << endl;
		int64 t = getTickCount();
		Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(nfeatures * (99 + grid_size.area()) / 100 / grid_size.area(), scaleFactor, nlevels, false);
		//GPU�Ҷ�ͼת��
		image_gpu_1.upload(images[n]);
		image_gpu_2.upload(images[n + 1]);
		cv::cuda::cvtColor(image_gpu_1, gray_image_gpu_1, COLOR_BGR2GRAY);
		cv::cuda::cvtColor(image_gpu_2, gray_image_gpu_2, COLOR_BGR2GRAY);
		//��������㼰����
		orb->detectAndComputeAsync(gray_image_gpu_1, cuda::GpuMat(), keypoints_gpu_1, descriptors_gpu_1);
		orb->convert(keypoints_gpu_1, keypoints_1);
		descriptors_gpu_1.convertTo(descriptors_gpu_1_32F, CV_32F);

		orb->detectAndComputeAsync(gray_image_gpu_2, cuda::GpuMat(), keypoints_gpu_2, descriptors_gpu_2);
		orb->convert(keypoints_gpu_2, keypoints_2);
		descriptors_gpu_2.convertTo(descriptors_gpu_2_32F, CV_32F);

		//�ͷ��ڴ�
		image_gpu_1.release();
		image_gpu_2.release();
		gray_image_gpu_1.release();
		gray_image_gpu_2.release();
		keypoints_gpu_1.release();
		keypoints_gpu_2.release();
		descriptors_gpu_1.release();
		descriptors_gpu_2.release();
		orb.release();
		cout << "��������ȡ��ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;

		//������ƥ��
		cout << "������ƥ��" << endl;
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
		cout << "������ƥ���ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;

		//�洢������ƥ��ͼ��
		Mat match_image;
		drawMatches(images[n], keypoints_1, images[n+1], keypoints_2, good_matches, match_image);
		string match_name = "match_image" + to_string(num) + ".jpg";
		imwrite(match_name, match_image);

		//ͼ����׼
		cout << "ͼ����׼" << endl;
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

		//������׼ͼ���ĸ���������
		CalcCorners(H, images[n + 1]);

		//ͼ����׼  
		Mat imageTransform1, imageTransform2;
		cv::warpPerspective(images[n + 1], imageTransform2, H, Size(MAX(corners.right_top.x, corners.right_bottom.x), images[n].rows));
		//imshow("ֱ�Ӿ���͸�Ӿ���任", imageTransform2);
		//waitKey();
		//�洢��׼ͼ��
		//imwrite("trans_pic.jpg", imageTransform2);
		cout << "ͼ����׼��ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;


		//ͼ��ƴ��
		cout << "ͼ��ƴ��" << endl;
		t = getTickCount();
		//����ƴ�Ӻ��ͼ,����ǰ����ͼ�Ĵ�С
		int dst_width = imageTransform2.cols;
		int dst_height = images[n].rows;

		Mat dst(dst_height, dst_width, CV_8UC3);
		dst.setTo(0);

		imageTransform2.copyTo(dst(Rect(0, 0, imageTransform2.cols, imageTransform2.rows)));
		images[n].copyTo(dst(Rect(0, 0, images[n].cols, images[n].rows)));
		OptimizeSeam(images[n], imageTransform2, dst);
		//�洢ƴ����ɵ�ͼ��
		string file_name = "dst" + to_string(num) + ".jpg";
		imwrite(file_name, dst);
		cout << "ͼ��ƴ�Ӻ�ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;
		cout << "�ܺ�ʱ: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec\n\n" << endl;
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
	double v2[] = { 0, 0, 1 };//���Ͻ�
	double v1[3];//�任�������ֵ
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //������
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //������

	V1 = H * V2;
	//���Ͻ�(0,0,1)
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//���½�(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //������
	V1 = Mat(3, 1, CV_64FC1, v1);  //������
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//���Ͻ�(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //������
	V1 = Mat(3, 1, CV_64FC1, v1);  //������
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//���½�(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //������
	V1 = Mat(3, 1, CV_64FC1, v1);  //������
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

}

//�Ż���ͼ�����Ӵ���ʹ��ƴ����Ȼ
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);//��ʼλ�ã����ص��������߽�  

	double processWidth = img1.cols - start;//�ص�����Ŀ��  
	int rows = dst.rows;
	int cols = img1.cols; //ע�⣬������*ͨ����
	double alpha = 1;//img1�����ص�Ȩ��  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //��ȡ��i�е��׵�ַ
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//�������ͼ��trans�������صĺڵ㣬����ȫ����img1�е�����
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1�����ص�Ȩ�أ��뵱ǰ�������ص�������߽�ľ�������ȣ�ʵ��֤�������ַ���ȷʵ��  
				alpha = (processWidth - (j - start)) / processWidth;
			}

			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}
	}

}