namespace csharp_neural_net;
public class NeuralNet
{
    public double[,] X { get; set; }
    public double[] Y { get; set; }
    public double SampleSize { get; set; }
    public double[,] Weights1 { get; set; }
    public double[,] Bias1 { get; set; }
    public double[,] Weights2 { get; set; }
    public double[,] Bias2 { get; set; }
    public double[,] Predictions { get; set; }

    public NeuralNet(double[,] x, double[] y)
    {
        X = x;
        Y = y;
    }

    public void Run(int iterations, int batchSize, double learningRate)
    {
        InitParams(batchSize);

        for (var run = 0; run < iterations; run++)
        {
            var randIndex = new Random().Next(0, X.GetLength(0) - batchSize);

            var xBatch = MatrixHelper.TransposeMatrix(MatrixHelper.GetBatch(X, batchSize, randIndex));
            var yBatch = Y.Skip(randIndex).Take(batchSize).ToArray();

            GradientDescent(xBatch, yBatch, learningRate);

            if (run % 10 == 0)
            {
                var accuracy = GetAccuracy(xBatch, yBatch, Predictions);
                Console.WriteLine($"Accuracy: {accuracy}, Iteration: {run}");
            }
        }
    }
    private void InitParams(double sampleSize)
    {
        SampleSize = sampleSize;
        Weights1 = MatrixHelper.GetRandomMatrix(10, 784);
        Bias1 = MatrixHelper.GetRandomMatrix(10, 1);
        Weights2 = MatrixHelper.GetRandomMatrix(10, 10);
        Bias2 = MatrixHelper.GetRandomMatrix(10, 1);
    }

    private void GradientDescent(double[,] x, double[] y, double learningRate)
    {
        // forward pass
        var linearResult1 = Linear(Weights1, x, Bias1);
        var activiationResult1 = MatrixHelper.ReLU(linearResult1);
        var linearResult2 = Linear(Weights2, activiationResult1, Bias2);
        var activationResult2 = MatrixHelper.SoftMax(linearResult2);
        Predictions = activationResult2;

        // backward pass
        var oneHotY = MatrixHelper.TransposeMatrix(MatrixHelper.OneHotEncode(y));
        var derivativeActivation2 = MatrixHelper.Subtract(activationResult2, oneHotY);
        var derivativeWeights2 = MatrixHelper.MultiplyValue(1 / SampleSize, MatrixHelper.DotProduct(derivativeActivation2, MatrixHelper.TransposeMatrix(activiationResult1)));

        var derivativeBias2 = MatrixHelper.Sum(derivativeActivation2) / SampleSize;

        var derivativeActivation1 = MatrixHelper.MultiplyValues(MatrixHelper.DotProduct(MatrixHelper.TransposeMatrix(Weights2), derivativeActivation2), MatrixHelper.ReLUDerivative(activiationResult1));

        var derivativeWeights1 = MatrixHelper.MultiplyValue(1 / SampleSize, MatrixHelper.DotProduct(derivativeActivation1, MatrixHelper.TransposeMatrix(x)));

        var derivativeBias1 = MatrixHelper.Sum(derivativeActivation1) / SampleSize;

        // update weights
        Weights1 = MatrixHelper.Subtract(Weights1, MatrixHelper.MultiplyValue(learningRate, derivativeWeights1));
        Weights2 = MatrixHelper.Subtract(Weights2, MatrixHelper.MultiplyValue(learningRate, derivativeWeights2));
        Bias1 = MatrixHelper.Subtract(learningRate * derivativeBias1, Bias1);
        Bias2 = MatrixHelper.Subtract(learningRate * derivativeBias2, Bias2);
    }

    public double[,] GetPredictions(double[,] x)
    { 
        var linearResult1 = Linear(Weights1, x, Bias1);
        var activiationResult1 = MatrixHelper.ReLU(linearResult1);
        var linearResult2 = Linear(Weights2, activiationResult1, Bias2);
        var activationResult2 = MatrixHelper.SoftMax(linearResult2);
        return activationResult2;
    }

    public double GetAccuracy(double[,] x, double[] y, double[,] predictions = null)
    {
        if (predictions == null)
        {
            predictions = GetPredictions(x);
        }

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

            if (maxIndex == y[i])
            {
                correct += 1;
            }
        }

        return (double)correct / y.Length;
    }

    private double[,] Linear(double[,] weights, double[,] x, double[,] bias)
    {
        var mx = MatrixHelper.DotProduct(weights, x);

        for (var i = 0; i < mx.GetLength(0); i++)
        {
            for (var j = 0; j < mx.GetLength(1); j++)
            {
                mx[i, j] += bias[i, 0];
            }
        }

        return mx;
    }

    private void PrintMatrix(double[,] matrix)
    {
        for (var i = 0; i < matrix.GetLength(0); i++)
        {
            Console.WriteLine();
            for (var j = 0; j< matrix.GetLength(1); j++)
            {
                Console.Write($"{matrix[i,j]} ");
            }
        }
        Console.WriteLine();
    }
}
