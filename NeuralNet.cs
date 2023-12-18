namespace csharp_neural_net;
public class NeuralNet
{
    public double[,] X { get; set; }
    public double[] Y { get; set; }
    public double M { get; set; }
    public double LearningRate { get; set; }
    public double[,] Weights1 { get; set; }
    public double[,] Bias1 { get; set; }
    public double[,] Weights2 { get; set; }
    public double[,] Bias2 { get; set; }

    public NeuralNet(double[,] x, double[] y, double learningRate)
    {
        X = x;
        Y = y;
        LearningRate = learningRate;
    }

    public void InitParams(double m)
    {
        M = m;
        Weights1 = MatrixHelper.GetRandomMatrix(10, 784);
        Bias1 = MatrixHelper.GetRandomMatrix(10, 1);
        Weights2 = MatrixHelper.GetRandomMatrix(10, 10);
        Bias2 = MatrixHelper.GetRandomMatrix(10, 1);
    }

    public void Run(int epochs)
    {
        for (var i = 0; i < epochs; i++)
        {
            GradientDescent();
        }
    }

    public void GradientDescent()
    {
        // forward pass
        var linearResult1 = Linear(Weights1, X, Bias1);
        var activiationResult1 = ReLU(linearResult1);
        var linearResult2 = Linear(Weights2, activiationResult1, Bias2);
        var activationResult2 = SoftMax(linearResult2);

        // backward pass
        var oneHotY = MatrixHelper.TransposeMatrix(MatrixHelper.OneHotEncode(Y));
        var derivativeActivation2 = MatrixHelper.Subtract(activationResult2, oneHotY);
        var derivativeWeights2 = MatrixHelper.Multiply(1 / M, DotProduct(derivativeActivation2, MatrixHelper.TransposeMatrix(activiationResult1)));

        var derivativeActivation2Sum = 0d;

        for (var i = 0; i < derivativeActivation2.GetLength(0); i++)
        {
            for (var j = 0; j < derivativeActivation2.GetLength(1); j++)
            {
                derivativeActivation2Sum += derivativeActivation2[i, j];
            }
        }

        var derivativeBias2 = derivativeActivation2Sum / M;

        var derivativeActivation1 = MatrixHelper.Multiply(DotProduct(MatrixHelper.TransposeMatrix(Weights2), derivativeActivation2), ReLUDerivative(activiationResult1));

        var derivativeWeights1 = MatrixHelper.Multiply(1 / M, DotProduct(derivativeActivation1, MatrixHelper.TransposeMatrix(X)));

        var derivativeActivation1Sum = 0d;

        for (var i = 0; i < derivativeActivation1.GetLength(0); i++)
        { 
            for (var j = 0; j < derivativeActivation1.GetLength(1); j++)
            {
                derivativeActivation1Sum += derivativeActivation1[i, j];
            }
        }

        var derivativeBias1 = derivativeActivation1Sum / M;

        // update weights
        Weights1 = MatrixHelper.Subtract(Weights1, MatrixHelper.Multiply(LearningRate, derivativeWeights1));
        Weights2 = MatrixHelper.Subtract(Weights2, MatrixHelper.Multiply(LearningRate, derivativeWeights2));

        Bias1 = MatrixHelper.Subtract(LearningRate * derivativeBias1, Bias1);
        Bias2 = MatrixHelper.Subtract(LearningRate * derivativeBias2, Bias2);
    }

    private double[,] ReLU(double[,] matrix)
    { 
        for (var i = 0; i < matrix.GetLength(0); i++)
        {
            for (var j = 0; j < matrix.GetLength(1); j++)
            {
                matrix[i, j] = Math.Max(0, matrix[i, j]);
            }
        }

        return matrix;
    }

    private double[,] ReLUDerivative(double[,] matrix)
    { 
        for (var i = 0; i < matrix.GetLength(0); i++)
        {
            for (var j = 0; j < matrix.GetLength(1); j++)
            {
                matrix[i, j] = matrix[i, j] > 0 ? 1 : 0;
            }
        }

        return matrix;
    }

    private double[,] SoftMax(double[,] matrix)
    { 
        for (var col = 0; col < matrix.GetLength(1); col++)
        {
            var sum = 0d;
            for (var row = 0; row < matrix.GetLength(0); row++)
            {
                sum += (double)Math.Exp(matrix[row,col]);
            }

            for (var row = 0; row < matrix.GetLength(0); row++)
            {
                matrix[row,col] = (double)Math.Exp(matrix[row,col]) / sum;
            }
        }

        return matrix;
    }

    private double[,] Linear(double[,] weights, double[,] x, double[,] bias)
    {
        var mx = DotProduct(weights, x);

        for (var i = 0; i < mx.GetLength(0); i++)
        {
            for (var j = 0; j < mx.GetLength(1); j++)
            {
                mx[i, j] += bias[i, 0];
            }
        }

        return mx;
    }

    public double[,] DotProduct(double[,] a, double[,] b)
    {
        if (a.GetLength(1) != b.GetLength(0))
        {
            throw new Exception("Matrices can not be multipled");
        }

        double[,] result = new double[a.GetLength(0), b.GetLength(1)];

        for (int i = 0; i < a.GetLength(0); i++)
        {
            for (int j = 0; j < b.GetLength(1); j++)
            {
                double sum = 0;
                for (int k = 0; k < a.GetLength(1); k++)
                {
                    sum += a[i, k] * b[k, j];
                }

                result[i, j] = sum;
            }   
        }

        return result;
    }

    private void PrintMatrix(double[,] matrix)
    {
        Console.WriteLine($"{matrix.GetLength(0)}, {matrix.GetLength(1)}");

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
