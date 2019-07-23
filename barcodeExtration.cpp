#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc//imgproc.hpp>
#include <vector>
#include <opencv2/opencv.hpp>  
#include <assert.h>
#include <time.h>
#include <string>
#include <io.h>
#include <fstream>
#include <iostream>
  
using namespace cv;  
using namespace std;

bool get_filelist_from_dir(string path,vector<string>& files)
{
    long hFile = 0;
    struct _finddata_t fileinfo;
    files.clear();
    if((hFile = _findfirst(path.c_str(),&fileinfo)) != -1)
    {
        do
        {
            if(!(fileinfo.attrib & _A_SUBDIR))
                files.push_back(fileinfo.name);
        }while(_findnext(hFile,&fileinfo) == 0);
        _findclose(hFile);
        return true;
    }
    else
        return false;
}

Mat lineFilter_v(Mat src, int filterLength_lb, int filterLength_hb)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat::zeros(height,width,CV_8UC1);
	for(int i = 0; i < width; ++i)
	{
		int j = 0;
		int cnt = 0;
		while(j < height)
		{
			if(static_cast<int>(src.at<Vec3b>(j,i)[0]) < 128)
			{
				j++;
			}
			else
			{
				int start = j;
				j++;
				cnt++;
				while(j < height && static_cast<int>(src.at<Vec3b>(j,i)[0]) >= 128)
				{
					j++;
					cnt++;
				}
				if(cnt > filterLength_lb && cnt < filterLength_hb)
				{
					for(int k = 0; k < cnt; ++k )
					{
						dst.at<uchar>(start+k,i) = 255;
					}
				}
				cnt = 0;
			}
		}
	}

	for(int i = 0; i < height; ++i)
	{
		int cnt = countNonZero(dst.rowRange(i,i+1));
		if(cnt < 30)
			dst.rowRange(i,i+1) = 0;
	}
	return dst;
}

Mat lineFilter_h(Mat src, int filterLength_lb, int filterLength_hb)
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat::zeros(height,width,CV_8UC1);
	for(int i = 0; i < height; ++i)
	{
		int j = 0;
		int cnt = 0;
		while(j < width)
		{
			if(static_cast<int>(src.at<Vec3b>(i,j)[0]) < 128)
			{
				j++;
			}
			else
			{
				int start = j;
				j++;
				cnt++;
				while(j < width && static_cast<int>(src.at<Vec3b>(i,j)[0]) >= 128)
				{
					j++;
					cnt++;
				}
				if(cnt > filterLength_lb && cnt < filterLength_hb)
				{
					for(int k = 0; k < cnt; ++k )
					{
						dst.at<uchar>(i,start+k) = 255;
					}
				}
				cnt = 0;
			}
		}
	}

	for(int i = 0; i < width; ++i)
	{
		int cnt = countNonZero(dst.colRange(i,i+1));
		if(cnt < 30)
			dst.colRange(i,i+1) = 0;
	}
	return dst;
}

Mat extraction(Mat src)
{
	Mat grad_x,grad_y;
    Mat abs_grad_x, abs_grad_y;

    Mat srcNormalized;
    int normalizedRow = 660;
    int normalizedCol = 510;
    resize(src,srcNormalized,Size(normalizedCol,normalizedRow));
   

    Sobel(srcNormalized,grad_x,CV_8U,1,0,3,1,1,BORDER_DEFAULT);
    convertScaleAbs(grad_x,abs_grad_x);
      
    Sobel(srcNormalized,grad_y,CV_8U,0,1,3,1,1,BORDER_DEFAULT);
    convertScaleAbs(grad_y,abs_grad_y);
     
    Mat gradient = abs(abs_grad_x - abs_grad_y);

    
    int filterLength_lb = 15;
    int filterLength_hb = 100;
    Mat filteredImg_v = lineFilter_v(gradient,filterLength_lb,filterLength_hb);
    Mat filteredImg_h = lineFilter_h(gradient,filterLength_lb,filterLength_hb);
	
    
    Mat closing_v,closing_h;
    Mat element = getStructuringElement(MORPH_RECT,Size(20,15));
    morphologyEx(filteredImg_v,closing_v,MORPH_CLOSE,element);
    Mat element2 = getStructuringElement(MORPH_RECT,Size(15,20));
    morphologyEx(filteredImg_h,closing_h,MORPH_CLOSE,element);

    
    vector<vector<Point>> contours1;
    findContours(closing_v,contours1,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> contours2;
    findContours(closing_h,contours2,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE);

    //竖直方向：
    if(!contours1.empty())
	{
		int MaxArea = INT_MIN;
		int MaxAreaIndex = 0;
		for(int i = 0; i < contours1.size(); ++i)
		{
			int area = contourArea(contours1[i]);
			if(area > MaxArea)
			{
				MaxArea = area;
				MaxAreaIndex = i;
			}
		}
		int rowStart = contours1[MaxAreaIndex][0].y;;
		int rowEnd = contours1[MaxAreaIndex][0].y;
		int colStart = contours1[MaxAreaIndex][0].x;
		int colEnd = contours1[MaxAreaIndex][0].x;

		for(int i = 1; i < contours1[MaxAreaIndex].size(); ++i)
		{
			if(rowStart > contours1[MaxAreaIndex][i].y)
				rowStart = contours1[MaxAreaIndex][i].y;
			if(rowEnd < contours1[MaxAreaIndex][i].y)
				rowEnd = contours1[MaxAreaIndex][i].y;
			if(colStart > contours1[MaxAreaIndex][i].x)
				colStart = contours1[MaxAreaIndex][i].x;
			if(colEnd < contours1[MaxAreaIndex][i].x)
				colEnd = contours1[MaxAreaIndex][i].x;
		}
	
		if(colStart - 20 > 0)
			colStart = colStart - 20;
		else
			colStart = 0;
		if(colEnd + 20 < srcNormalized.cols)
			colEnd = colEnd + 20;
		else
			colEnd = srcNormalized.cols;
		if(rowStart - 20 > 0)
			rowStart = rowStart - 20;
		else
			rowStart = 0;
		if(rowEnd + 20 < srcNormalized.rows)
			rowEnd = rowEnd + 20;
		else
			rowEnd = src.rows;

		rectangle(srcNormalized,cvPoint(colStart,rowStart),cvPoint(colEnd,rowEnd),Scalar(0,255,0),3);
	}
	
	
    //水平方向：
    if(!contours2.empty())
	{
		int MaxArea2 = INT_MIN;
		int MaxAreaIndex2 = 0;
		for(int i = 0; i < contours2.size(); ++i)
		{
			int area = contourArea(contours2[i]);
			if(area > MaxArea2)
			{
				MaxArea2 = area;
				MaxAreaIndex2 = i;
			}
		}
		int rowStart2 = contours2[MaxAreaIndex2][0].y;;
		int rowEnd2 = contours2[MaxAreaIndex2][0].y;
		int colStart2 = contours2[MaxAreaIndex2][0].x;
		int colEnd2 = contours2[MaxAreaIndex2][0].x;

		for(int i = 1; i < contours2[MaxAreaIndex2].size(); ++i)
		{
			if(rowStart2 > contours2[MaxAreaIndex2][i].y)
				rowStart2 = contours2[MaxAreaIndex2][i].y;
			if(rowEnd2 < contours2[MaxAreaIndex2][i].y)
				rowEnd2 = contours2[MaxAreaIndex2][i].y;
			if(colStart2 > contours2[MaxAreaIndex2][i].x)
				colStart2 = contours2[MaxAreaIndex2][i].x;
			if(colEnd2 < contours2[MaxAreaIndex2][i].x)
				colEnd2 = contours2[MaxAreaIndex2][i].x;
		}
	
		if(colStart2 - 20 > 0)
			colStart2 = colStart2 - 20;
		else
			colStart2 = 0;
		if(colEnd2 + 20 < srcNormalized.cols)
			colEnd2 = colEnd2 + 20;
		else
			colEnd2 = srcNormalized.cols;
		if(rowStart2 - 20 > 0)
			rowStart2 = rowStart2 - 20;
		else
			rowStart2 = 0;
		if(rowEnd2 + 20 < srcNormalized.rows)
			rowEnd2 = rowEnd2 + 20;
		else
			rowEnd2 = srcNormalized.rows;
	
		
		rectangle(srcNormalized,cvPoint(colStart2,rowStart2),cvPoint(colEnd2,rowEnd2),Scalar(0,255,0),3);
	}
    return srcNormalized;
}

int main()
{
    clock_t t_start,t_end;
    t_start = clock();
	string file_path = "F:\\ebayWork\\normal image\\";
    string search_path = file_path + "*.jpg";
	vector<string> file_list;

	if(!get_filelist_from_dir(search_path,file_list)) 
        cout<<"open file error!"<<endl;

	for (int i = 0; i < file_list.size(); ++i)
	{
		string image_path = file_path + file_list[i];
		Mat src = imread(image_path);
		Mat result = extraction(src);
		imshow(image_path,result);
	}
    t_end = clock();
    cout<<"程序所用时间："<<static_cast<double>(t_end - t_start)<<"ms";
	
    waitKey(0);
    return 0;
}