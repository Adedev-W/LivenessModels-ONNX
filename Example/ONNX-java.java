```java
import ai.onnxruntime.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.FloatBuffer;
import javax.imageio.ImageIO;
import java.awt.Image;

public class OnnxInference {
    private static final int IMAGE_SIZE = 224;
    private static final float RESCALE = 1f / 255f;
    private static final float[] MEAN = {0.5f, 0.5f, 0.5f};
    private static final float[] STD = {0.5f, 0.5f, 0.5f};

    public static float[] preprocessImage(String imagePath) throws Exception {
        BufferedImage originalImage = ImageIO.read(new File(imagePath));
        
        // Resize image
        Image scaledImage = originalImage.getScaledInstance(IMAGE_SIZE, IMAGE_SIZE, Image.SCALE_SMOOTH);
        BufferedImage resizedImage = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_INT_RGB);
        resizedImage.getGraphics().drawImage(scaledImage, 0, 0, null);
        
        // Convert to CHW format
        float[] imageArray = new float[3 * IMAGE_SIZE * IMAGE_SIZE];
        
        for (int y = 0; y < IMAGE_SIZE; y++) {
            for (int x = 0; x < IMAGE_SIZE; x++) {
                int rgb = resizedImage.getRGB(x, y);
                
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                
                // CHW indices
                int rIndex = 0 * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x;
                int gIndex = 1 * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x;
                int bIndex = 2 * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x;
                
                // Apply preprocessing
                imageArray[rIndex] = ((r * RESCALE) - MEAN[0]) / STD[0];
                imageArray[gIndex] = ((g * RESCALE) - MEAN[1]) / STD[1];
                imageArray[bIndex] = ((b * RESCALE) - MEAN[2]) / STD[2];
            }
        }
        
        return imageArray;
    }

    public static float[] runInference(String modelPath, String imagePath) throws Exception {
        // Load model
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession session = env.createSession(modelPath);
        
        // Preprocess image
        float[] inputData = preprocessImage(imagePath);
        
        // Create tensor
        long[] shape = {1, 3, IMAGE_SIZE, IMAGE_SIZE};
        OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);
        
        // Get input name
        String inputName = session.getInputNames().iterator().next();
        
        // Run inference
        OrtSession.Result result = session.run(Map.of(inputName, tensor));
        
        // Get output
        float[][] output = (float[][]) result.get(0).getValue();
        
        // Clean up
        tensor.close();
        result.close();
        session.close();
        
        return output[0];
    }

    public static void main(String[] args) {
        try {
            String modelPath = "model.onnx";
            String imagePath = "test_image.jpg";
            
            float[] result = runInference(modelPath, imagePath);
            System.out.println("Output length: " + result.length);
            System.out.println("First 5 values: " + 
                Arrays.toString(Arrays.copyOf(result, 5)));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
