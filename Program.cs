using System.IO.Compression;
using System.Net;
using System.Security.Cryptography;
using System.Text;

namespace csharp_neural_net;
public class Program
{
    public static void Main(string[] args)
    {
        var XTrainCompressed = Fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz");
        var YTrainCompressed = Fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz");
        var XTestCompressed = Fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz");
        var YTestCompressed = Fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz");

        var XTrain = ReshapeData(Decompress(XTrainCompressed).Skip(0x10).ToArray());
        var YTrain = Decompress(YTrainCompressed).Skip(8).Select(x => (double)x).ToArray();
        var XTest = ReshapeData(Decompress(XTestCompressed).Skip(0x10).ToArray());
        var YTest = Decompress(YTestCompressed).Skip(8).Select(x => (double)x).ToArray();

        // normalize
        XTrain = MatrixHelper.MultiplyValue(1d / 255, XTrain);


        var neuralNet = new NeuralNet(XTrain, YTrain);
        neuralNet.Run(5000, 300, 0.1d);

        Console.WriteLine("Calculating accuracy on test data...");

        XTest = MatrixHelper.TransposeMatrix(MatrixHelper.MultiplyValue(1d / 255, XTest));
        var predictions = neuralNet.GetPredictions(XTest);

        var correct = 0;

        for (var i = 0; i < predictions.GetLength(1); i++)
        {
            int maxIndex = 0;
            for (var j = 1; j < predictions.GetLength(0); j++)
            {
                if (predictions[j, i] > predictions[maxIndex, i])
                {
                    maxIndex = j;
                }
            }

            if (maxIndex == YTest[i])
            {
                correct += 1;
            }
        }

        var accuracy = (double)correct / YTest.Length;
        Console.WriteLine($"Test Accuracy: {accuracy}");
    }

    private static byte[] Fetch(string url)
    {
        string filePath = Path.Combine(Path.GetTempPath(), GetMd5Hash(url));

        if (File.Exists(filePath))
        {
            return File.ReadAllBytes(filePath);
        }
        else
        {
            byte[] data;
            using (var client = new WebClient())
            {
                data = client.DownloadData(url);
            }

            File.WriteAllBytes(filePath, data);
            return data;
        }
    }

    private static string GetMd5Hash(string input)
    {
        using (var md5 = MD5.Create())
        {
            var data = md5.ComputeHash(Encoding.UTF8.GetBytes(input));
            return BitConverter.ToString(data).Replace("-", "").ToLower();
        }
    }

    private static byte[] Decompress(byte[] data)
    {
        using (MemoryStream compressedStream = new MemoryStream(data))
        using (MemoryStream decompressedStream = new MemoryStream())
        using (GZipStream gzipStream = new GZipStream(compressedStream, CompressionMode.Decompress))
        {
            gzipStream.CopyTo(decompressedStream);
            return decompressedStream.ToArray();
        }
    }

    private static double[,] ReshapeData(byte[] data)
    {
        int numRows = data.Length / 784;
        int numCols = 784;

        var reshapedData = new double[numRows, numCols];

        for (var i = 0; i < data.Length; i++)
        {
            int row = i / numCols;
            int col = i % numCols;
            reshapedData[row, col] = (double)data[i];
        }

        return reshapedData;
    }

    private static void PrintNumber(double[,] numData, double[] labelData, int index)
    {
        for (var i = 0; i < numData.GetLength(1); i += 1)
        {
            if (i != 0 && i % 28 == 0)
            {
                Console.WriteLine();
            }

            Console.Write(numData[index, i] > 0d ? 1 : 0);
        }

        Console.WriteLine();

        Console.Write($"Label: {labelData[index]}");
    }

}
