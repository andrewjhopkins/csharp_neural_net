using System.IO.Compression;
using System.Net;
using System.Security.Cryptography;
using System.Text;

namespace csharp_neural_net;
public class Program
{
    public static void Main(string[] args)
    {
        byte[] XTrainCompressed = Fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz");
        byte[] YTrainCompressed = Fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz");
        byte[] XTestCompressed = Fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz");
        byte[] YTestCompressed = Fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz");

        var XTrain = ReshapeData(Decompress(XTrainCompressed).Skip(0x10).ToArray());
        var YTrain = Decompress(YTrainCompressed).Skip(8).Select(x => (double)x).ToArray();
        var XTest = ReshapeData(Decompress(XTestCompressed).Skip(0x10).ToArray());
        var YTest = Decompress(YTestCompressed).Skip(8).Select(x => (double)x).ToArray();

        var samples = 10;

        var XTrainBatch = MatrixHelper.Multiply((double)1 / 255, TransposeMatrix(GetBatch(XTrain, samples, 0)));
        var YTrainBatch = YTrain.Take(samples).ToArray();

/*
        for (var i = 0; i < XTrainBatch.GetLength(0); i++)
        {
            Console.WriteLine();
            for (var j = 0; j < XTrainBatch.GetLength(1); j++)
            {
                Console.Write(XTrainBatch[i, j]);
            }
        }
        */


/*
        Console.WriteLine();
        Console.WriteLine(YTrainBatch[0]);
        */

        var M = XTrainBatch.GetLength(1);

        var neuralNet = new NeuralNet(XTrainBatch, YTrainBatch, 0.1f);
        neuralNet.InitParams(M);
        neuralNet.GradientDescent();

        /*
        var example1 = new double[2,3] { {1f, 2f, 3f}, {4f, 5f, 6f} };
        var example2 = new double[3,2] { {10f, 11f}, {20f, 21f}, {30f, 31f} };

        var example3 = neuralNet.DotProduct(example1, example2);

        for (var i = 0; i < example3.GetLength(0); i++)
        {
            Console.WriteLine();
            for (var j = 0; j < example3.GetLength(1); j++)
            {
                Console.Write(example3[i,j]);
                Console.Write(" ");
            }
        }
        */
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
            using (WebClient client = new WebClient())
            {
                data = client.DownloadData(url);
            }

            File.WriteAllBytes(filePath, data);
            return data;
        }
    }

    private static string GetMd5Hash(string input)
    {
        using (MD5 md5 = MD5.Create())
        {
            byte[] data = md5.ComputeHash(Encoding.UTF8.GetBytes(input));
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

    private static double[,] GetBatch(double[,] data, int batch, int startRow)
    { 
        var batchData = new double[batch, data.GetLength(1)];

        for (var i = 0; i < batch; i++)
        {
            for (var j = 0; j < data.GetLength(1); j++)
            {
                batchData[i, j] = data[startRow + i, j];
            }
        }

        return batchData;
    }

    private static double[,] TransposeMatrix(double[,] matrix)
    { 
        var transposedMatrix = new double[matrix.GetLength(1), matrix.GetLength(0)];

        for (var i = 0; i < matrix.GetLength(0); i++)
        {
            for (var j = 0; j < matrix.GetLength(1); j++)
            {
                transposedMatrix[j, i] = matrix[i, j];
            }
        }

        return transposedMatrix;
    }
}
