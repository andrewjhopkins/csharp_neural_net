namespace csharp_neural_net;
public class NeuralNet
{
    public double[,] X { get; set; }
    public double[] Y { get; set; }
    public double SampleSize { get; set; }
    public double LearningRate { get; set; }
    public double[,] Weights1 { get; set; }
    public double[,] Bias1 { get; set; }
    public double[,] Weights2 { get; set; }
    public double[,] Bias2 { get; set; }
    public double[,] Predictions { get; set; }

    public NeuralNet(double[,] x, double[] y, double learningRate)
    {
        X = x;
        Y = y;
        LearningRate = learningRate;
    }

    public void Run(double sampleSize, int epochs)
    {
        InitParams(sampleSize);

        for (var run = 0; run < epochs; run++)
        {
            GradientDescent();
            if (run % 10 == 0)
            {
                var correct = 0;

                for (var i = 0; i < Predictions.GetLength(1); i++)
                {
                    int maxIndex = 0;
                    for (var j = 1; j < Predictions.GetLength(0); j++)
                    {
                        if (Predictions[j, i] > Predictions[maxIndex, i])
                        {
                            maxIndex = j;
                        }
                    }

                    if (maxIndex == Y[i])
                    {
                        correct += 1;
                    }
                }

                var accuracy = (double)correct / Y.Length;
                Console.WriteLine($"Accuracy: {accuracy}");
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

    private void GradientDescent()
    {
        // forward pass
        var linearResult1 = Linear(Weights1, X, Bias1);

        var activiationResult1 = MatrixHelper.ReLU(linearResult1);

        var linearResult2 = Linear(Weights2, activiationResult1, Bias2);

        var activationResult2 = MatrixHelper.SoftMax(linearResult2);
        Predictions = activationResult2;

        // backward pass
        var oneHotY = MatrixHelper.TransposeMatrix(MatrixHelper.OneHotEncode(Y));
        var derivativeActivation2 = MatrixHelper.Subtract(activationResult2, oneHotY);
        var derivativeWeights2 = MatrixHelper.MultiplyValue(1 / SampleSize, MatrixHelper.DotProduct(derivativeActivation2, MatrixHelper.TransposeMatrix(activiationResult1)));

        var derivativeBias2 = MatrixHelper.Sum(derivativeActivation2) / SampleSize;

        var derivativeActivation1 = MatrixHelper.MultiplyValues(MatrixHelper.DotProduct(MatrixHelper.TransposeMatrix(Weights2), derivativeActivation2), MatrixHelper.ReLUDerivative(activiationResult1));

        var derivativeWeights1 = MatrixHelper.MultiplyValue(1 / SampleSize, MatrixHelper.DotProduct(derivativeActivation1, MatrixHelper.TransposeMatrix(X)));

        var derivativeBias1 = MatrixHelper.Sum(derivativeActivation1) / SampleSize;

        // update weights
        Weights1 = MatrixHelper.Subtract(Weights1, MatrixHelper.MultiplyValue(LearningRate, derivativeWeights1));
        Weights2 = MatrixHelper.Subtract(Weights2, MatrixHelper.MultiplyValue(LearningRate, derivativeWeights2));

        Bias1 = MatrixHelper.Subtract(LearningRate * derivativeBias1, Bias1);
        Bias2 = MatrixHelper.Subtract(LearningRate * derivativeBias2, Bias2);
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
