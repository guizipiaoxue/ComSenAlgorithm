#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    cout << "SVM implementation with OpenCV starting..." << endl;
    
    // 测试OpenCV是否正常工作
    cout << "OpenCV version: " << CV_VERSION << endl;
    
    // 创建一个简单的图像
    Mat image = Mat::zeros(300, 400, CV_8UC3);
    
    // 在图像上绘制一些内容
    circle(image, Point(200, 150), 50, Scalar(0, 255, 0), -1);
    rectangle(image, Point(100, 100), Point(300, 200), Scalar(255, 0, 0), 2);
    putText(image, "OpenCV + SVM", Point(120, 130), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    
    // 显示图像信息
    cout << "Created image with size: " << image.cols << "x" << image.rows << endl;
    cout << "Image channels: " << image.channels() << endl;
    
    // 简单的SVM数据准备示例
    cout << "\nPreparing SVM training data..." << endl;
    
    // 创建训练数据 (2个特征)
    Mat trainingData = (Mat_<float>(6, 2) << 
        1, 2,
        2, 3,
        3, 3,
        2, 1,
        3, 2,
        1, 1);
    
    // 创建标签
    Mat labels = (Mat_<int>(6, 1) << 1, 1, 1, -1, -1, -1);
    
    cout << "Training data shape: " << trainingData.rows << "x" << trainingData.cols << endl;
    cout << "Labels shape: " << labels.rows << "x" << labels.cols << endl;
    
    // 创建SVM对象
    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    
    // 训练SVM
    cout << "Training SVM..." << endl;
    svm->train(trainingData, ml::ROW_SAMPLE, labels);
    cout << "SVM training completed!" << endl;
    
    // 测试预测
    Mat testData = (Mat_<float>(2, 2) << 1.5, 2.5, 2.5, 1.5);
    Mat results;
    svm->predict(testData, results);
    
    cout << "\nPrediction results:" << endl;
    for (int i = 0; i < results.rows; i++) {
        cout << "Test point " << i+1 << ": " << results.at<float>(i) << endl;
    }
    
    cout << "\nOpenCV and SVM integration test completed successfully!" << endl;
    
    return 0;
}
