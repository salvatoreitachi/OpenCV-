
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;


void sort_box(vector<Rect>& boxes)
{
    int size = boxes.size();
    for (int i = 0; i < size - 1; ++i) 
    {
        for (int j = i; j < size; ++j) 
        {
            if (boxes[j].y < boxes[i].y) 
            {
                Rect tmp = boxes[i];
                boxes[i] = boxes[j];
                boxes[j] = tmp;
            }
        }
    }
}
Mat get_template(Mat& binary, vector<Rect>& rects)
{
    return binary(rects[0]);
}
void detect_defects(Mat& binary, vector<Rect>& rects, Mat& tpl, vector<Rect>& defects)
{
    int height = tpl.rows;
    int width = tpl.cols;
    int index = 1;
    int size = rects.size();
    // ����ȱʧ
    for (int i = 0; i < size; ++i) 
    {
        Mat roi = binary(rects[i]);
        resize(roi, roi, tpl.size());
        Mat mask;
        subtract(tpl, roi, mask);
        Mat se = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, se);
        threshold(mask, mask, 0, 255, THRESH_BINARY);
        int count = 0;
        for (int row = 0; row < height; ++row) 
        {
            for (int col = 0; col < width; ++col) 
            {
                int pv = mask.at<uchar>(row, col);
                if (pv == 255) 
                {
                    ++count;
                }
            }
        }
        if (count > 0) {
            defects.push_back(rects[i]);
        }
    }
}

int main()
{
	Mat image;
	image = imread("E:\\opencv_tutorial-master\\opencv_tutorial-master\\data\\images\\ce_01.jpg"); // Read the file

	if (image.empty()) // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	namedWindow("input", WINDOW_AUTOSIZE); // Create a window for display.
	imshow("input", image); // Show our image inside it.

	//ͼ���ֵ��
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	imshow("binary", binary);

	//����ṹԪ��
	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));//MO_RECT���Σ�size��С��pointê��
	morphologyEx(binary, binary, MORPH_OPEN, se);
	imshow("open-binary", binary);

	//��������
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;//����㼶 ����4άint����
	vector<Rect>rects;
	findContours(binary, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);//������߽�����꣬��¼��contours��
    int height = image.rows;
    for (size_t t = 0; t < contours.size(); t++) {
        Rect rect = boundingRect(contours[t]);
        double area = contourArea(contours[t]);
        if (rect.height > (height / 2)) {
            continue;
        }
        if (area < 150) {
            continue;
        }
        rects.push_back(rect);
        // ����Ե���Ŵ�ȱ��
        drawContours(binary, contours, t, Scalar(0), 2, 8);
    }

    // ����Ӿ��ο�����
    sort_box(rects);

    // ��ȡģ��
    Mat tpl = get_template(binary, rects);

    for (int i = 0; i < rects.size(); ++i) {
        putText(image, format("num:%d", (i + 1)), Point(rects[i].x - 70, rects[i].y + 20),
            FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0), 1);
    }

    // ��Ⲣ�������
    vector<Rect> defects;
    detect_defects(binary, rects, tpl, defects);
    for (int i = 0; i < defects.size(); ++i) 
    {
        rectangle(image, defects[i], Scalar(0, 0, 255));
        putText(image, "bad", Point(defects[i].x, defects[i].y),
            FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 1);
    }

    imshow("result", image);
	
	waitKey(0);
    std::cout << "succeed.\n";
	return 0;// Wait for a keystroke in the window
}
