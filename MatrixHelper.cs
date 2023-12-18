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

    public static double[,] Multiply(double[,] matrix1, double[,] matrix2)
    {
        if (matrix1.GetLength(0) != matrix2.GetLength(0) || matrix1.GetLength(1) != matrix2.GetLength(1))
        {
            throw new Exception("Can not be multiplied");
        }

        var result = new double[matrix1.GetLength(0), matrix2.GetLength(1)];

        for (var i = 0; i < matrix1.GetLength(0); i++)
        {
            for (var j = 0; j < matrix1.GetLength(1); j++)
            {
                result[i,j] = matrix1[i,j] * matrix2[i,j];
            }
        }

        return result;
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

    public static double[,] TransposeMatrix(double[,] matrix)
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
