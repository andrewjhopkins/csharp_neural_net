namespace csharp_neural_net;

public class NeuralNet
{
    public double[,] X { get; set; }
    public double[] Y { get; set; }
    public double SampleSize { get; set; }
    public double[,] Weights1 { get; set; }
    public double[,] Weights2 { get; set; }
    public double[,] Bias1 { get; set; }
    public double[,] Bias2 { get; set; }

    public NeuralNet(double[,] x, double[] y)
    {
        X = x;
        Y = y;
    }

    public void Run(int batchSize, int iterations, double learningRate)
    {
        InitParams(batchSize);

        for (var run = 0; run < iterations; run++)
        {
            var randIndex = new Random().Next(0, X.GetLength(0) - batchSize);

            var x = MatrixHelper.TransposeMatrix(MatrixHelper.GetBatch(X, batchSize, randIndex));
            var y = Y.Skip(randIndex).Take(batchSize).ToArray();

            var prediction = GradientDescent(x, y, learningRate);

            if (run % 10 == 0)
            {
                var accuracy = GetAccuracy(x, y, prediction);
                Console.WriteLine($"Accuracy: {accuracy}, Iteration: {run}");
            }
        }
    }

    public double GetAccuracy(double[,] x, double[] y, double[,] predictions = null)
    {
        if (predictions == null)
        {
            (_, predictions) = ForwardPass(x);
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

    private void InitParams(double sampleSize)
    {
        SampleSize = sampleSize;
        Weights1 = MatrixHelper.GetRandomMatrix(10, 784);
        Bias1 = MatrixHelper.GetRandomMatrix(10, 1);
        Weights2 = MatrixHelper.GetRandomMatrix(10, 10);
        Bias2 = MatrixHelper.GetRandomMatrix(10, 1);
    }

    private double[,] GradientDescent(double[,] x, double[] y, double learningRate)
    {
        var (activationResult1, activationResult2) = ForwardPass(x);
        var (derivativeWeights1, derivativeWeights2, derivativeBias1, derivativeBias2) = BackwardPass(x, y, activationResult1, activationResult2);
        UpdateWeights(learningRate, derivativeWeights1, derivativeWeights2, derivativeBias1, derivativeBias2);
        return activationResult2;
    }

    private (double[,], double[,]) ForwardPass(double[,] x)
    { 
        var linearResult1 = Linear(Weights1, x, Bias1);
        var activationResult1 = MatrixHelper.ReLU(linearResult1);
        var linearResult2 = Linear(Weights2, activationResult1, Bias2);
        var activationResult2 = MatrixHelper.SoftMax(linearResult2);
        return (activationResult1, activationResult2);
    }

    private (double[,], double[,], double, double) BackwardPass(double[,] x, double[] y, double[,] activationResult1, double[,] activationResult2)
    { 
        var oneHotY = MatrixHelper.TransposeMatrix(MatrixHelper.OneHotEncode(y, 10));
        var derivativeActivation2 = MatrixHelper.Subtract(activationResult2, oneHotY);
        var derivativeWeights2 = MatrixHelper.MultiplyValue(1 / SampleSize, MatrixHelper.DotProduct(derivativeActivation2, MatrixHelper.TransposeMatrix(activationResult1)));
        var derivativeBias2 = MatrixHelper.Sum(derivativeActivation2) / SampleSize;
        var derivativeActivation1 = MatrixHelper.MultiplyValues(MatrixHelper.DotProduct(MatrixHelper.TransposeMatrix(Weights2), derivativeActivation2), MatrixHelper.ReLUDerivative(activationResult1));
        var derivativeWeights1 = MatrixHelper.MultiplyValue(1 / SampleSize, MatrixHelper.DotProduct(derivativeActivation1, MatrixHelper.TransposeMatrix(x)));
        var derivativeBias1 = MatrixHelper.Sum(derivativeActivation1) / SampleSize;

        return (derivativeWeights1, derivativeWeights2, derivativeBias1, derivativeBias2);
    }

    private void UpdateWeights(double learningRate, double[,] derivativeWeights1, double[,] derivativeWeights2, double derivativeBias1, double derivativeBias2)
    { 
        Weights1 = MatrixHelper.Subtract(Weights1, MatrixHelper.MultiplyValue(learningRate, derivativeWeights1));
        Weights2 = MatrixHelper.Subtract(Weights2, MatrixHelper.MultiplyValue(learningRate, derivativeWeights2));
        Bias1 = MatrixHelper.Subtract(learningRate * derivativeBias1, Bias1);
        Bias2 = MatrixHelper.Subtract(learningRate * derivativeBias2, Bias2);
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
}
