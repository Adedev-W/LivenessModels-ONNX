#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

class OnnxInference {
private:
    static constexpr int IMAGE_SIZE = 224;
    static constexpr float RESCALE = 1.0f / 255.0f;
    static constexpr float MEAN[3] = {0.5f, 0.5f, 0.5f};
    static constexpr float STD[3] = {0.5f, 0.5f, 0.5f};
    
public:
    static std::vector<float> preprocessImage(const std::string& imagePath) {
        // Load image
        cv::Mat image = cv::imread(imagePath);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        
        // Resize
        cv::resize(image, image, cv::Size(IMAGE_SIZE, IMAGE_SIZE));
        
        // Convert to float and apply preprocessing
        std::vector<float> imageArray(3 * IMAGE_SIZE * IMAGE_SIZE);
        
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < IMAGE_SIZE; h++) {
                for (int w = 0; w < IMAGE_SIZE; w++) {
                    int chwIndex = c * IMAGE_SIZE * IMAGE_SIZE + h * IMAGE_SIZE + w;
                    int hwcIndex = h * IMAGE_SIZE * 3 + w * 3 + c;
                    
                    float pixel = static_cast<float>(image.data[hwcIndex]) * RESCALE;
                    imageArray[chwIndex] = (pixel - MEAN[c]) / STD[c];
                }
            }
        }
        
        return imageArray;
    }
    
    static std::vector<float> runInference(const std::string& modelPath, 
                                         const std::string& imagePath) {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXInference");
        Ort::SessionOptions sessionOptions;
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
        
        // Preprocess image
        std::vector<float> inputData = preprocessImage(imagePath);
        
        // Create input tensor
        std::vector<int64_t> inputShape = {1, 3, IMAGE_SIZE, IMAGE_SIZE};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputData.data(), inputData.size(), 
            inputShape.data(), inputShape.size());
        
        // Get input/output names
        std::vector<const char*> inputNames = {session.GetInputName(0, allocator)};
        std::vector<const char*> outputNames = {session.GetOutputName(0, allocator)};
        
        // Run inference
        std::vector<Ort::Value> outputs = session.Run(
            Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1,
            outputNames.data(), 1);
        
        // Extract output
        float* outputData = outputs[0].GetTensorMutableData<float>();
        size_t outputSize = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        return std::vector<float>(outputData, outputData + outputSize);
    }
};


int main() {
    try {
        std::string modelPath = "model.onnx";
        std::string imagePath = "test_image.jpg";
        
        std::vector<float> result = OnnxInference::runInference(modelPath, imagePath);
        
        std::cout << "Output size: " << result.size() << std::endl;
        std::cout << "First 5 values: ";
        for (int i = 0; i < std::min(5, (int)result.size()); i++) {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
