﻿using System.IO.Compression;
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
        var YTrain = Decompress(YTrainCompressed).Skip(8).Select(x => (float) x ).ToArray();
        var XTest = ReshapeData(Decompress(XTestCompressed).Skip(0x10).ToArray());
        var YTest = Decompress(YTestCompressed).Skip(8).Select(x => (float) x ).ToArray();

        // PrintNumber(XTrain, YTrain, 4);

        var neuralNet = new NeuralNet(XTrain, YTrain, 0.1f);
        neuralNet.InitParams();
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

    private static float[,] ReshapeData(byte[] data)
    {
        int numRows = data.Length / 784;
        int numCols = 784;

        var reshapedData = new float[numRows, numCols];

        for (int i = 0; i < data.Length; i++)
        {
            int row = i / numCols;
            int col = i % numCols;
            reshapedData[row, col] = (float)data[i];
        }

        return reshapedData;
    }

    private static void PrintNumber(float[,] numData, float[] labelData, int index)
    {
        for (var i = 0; i < numData.GetLength(1); i += 1)
        {
            if (i != 0 && i % 28 == 0)
            {
                Console.WriteLine();
            }

            Console.Write(numData[index, i] > 0 ? 1 : 0);
        }

        Console.WriteLine();

        Console.Write($"Label: {labelData[index]}");
    }
}
