using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

public class OnnxInference
{
    private const int IMAGE_SIZE = 224;
    private const float RESCALE = 1f / 255f;
    private static readonly float[] MEAN = { 0.5f, 0.5f, 0.5f };
    private static readonly float[] STD = { 0.5f, 0.5f, 0.5f };

    public static float[] PreprocessImage(string imagePath)
    {
        using var image = Image.Load<Rgb24>(imagePath);
        
        // Resize
        image.Mutate(x => x.Resize(IMAGE_SIZE, IMAGE_SIZE));
        
        // Convert to array
        var imageArray = new float[3 * IMAGE_SIZE * IMAGE_SIZE];
        
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < IMAGE_SIZE; y++)
            {
                var pixelRow = accessor.GetRowSpan(y);
                for (int x = 0; x < IMAGE_SIZE; x++)
                {
                    var pixel = pixelRow[x];
                    
                    // CHW format
                    int rIndex = 0 * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x;
                    int gIndex = 1 * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x;
                    int bIndex = 2 * IMAGE_SIZE * IMAGE_SIZE + y * IMAGE_SIZE + x;
                    
                    // Apply preprocessing
                    imageArray[rIndex] = ((pixel.R * RESCALE) - MEAN[0]) / STD[0];
                    imageArray[gIndex] = ((pixel.G * RESCALE) - MEAN[1]) / STD[1];
                    imageArray[bIndex] = ((pixel.B * RESCALE) - MEAN[2]) / STD[2];
                }
            }
        });
        
        return imageArray;
    }

    public static float[] RunInference(string modelPath, string imagePath)
    {
        using var session = new InferenceSession(modelPath);
        
        // Preprocess image
        var inputData = PreprocessImage(imagePath);
        
        // Create tensor
        var tensor = new DenseTensor<float>(inputData, new[] { 1, 3, IMAGE_SIZE, IMAGE_SIZE });
        
        // Get input name
        var inputName = session.InputMetadata.Keys.First();
        
        // Run inference
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, tensor) };
        
        using var results = session.Run(inputs);
        var output = results.First().AsTensor<float>().ToArray();
        
        return output;
    }

    public static void Main()
    {
        string modelPath = "model.onnx";
        string imagePath = "test_image.jpg";
        
        var result = RunInference(modelPath, imagePath);
        Console.WriteLine($"Output length: {result.Length}");
        Console.WriteLine($"First 5 values: {string.Join(", ", result.Take(5))}");
    }
}
