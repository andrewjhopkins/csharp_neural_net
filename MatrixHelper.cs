namespace csharp_neural_net;

public static class MatrixHelper
{
    public static double[,] GetRandomMatrix(int rows, int cols)
    { 
        Random random = new Random();

        double[,] matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        { 
            for (int j = 0; j < cols; j++)
            {
                // values between -0.5 and 0.5
                // matrix[i, j] = (double)(random.NextDouble() - 0.5);
                matrix[i, j] = 0.5f;
            }   
        }

        return matrix;
    }

    public static double[,] Multiply(double multiplier, double[,] matrix)
    {
        for (var i = 0; i < matrix.GetLength(0); i++)
        {
            for (var j = 0; j < matrix.GetLength(1); j++)
            {
                matrix[i,j] *= multiplier;
            }
        }

        return matrix;
    }

    public static double[,] Subtract(double[,] a, double[,] b)
    {
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
        {
            throw new Exception("Matrices can not be subtracted");
        }

        double[,] matrix = new double[a.GetLength(0), a.GetLength(1)];

        for (var i = 0; i < a.GetLength(0); i++)
        {
            for (var j = 0; j < a.GetLength(0); j++)
            {
                matrix[i,j] = a[i,j] - b[i,j];
            }
        }

        return matrix;
    }

    public static double[,] OneHotEncode(double[] y)
    { 
        var oneHotMatrix = new double[y.Length, (int)y.Max() + 1];

        for (var i = 0; i < y.Length; i++)
        {
            oneHotMatrix[i, (int)y[i]] = 1;
        }

        return oneHotMatrix;
    }
}
