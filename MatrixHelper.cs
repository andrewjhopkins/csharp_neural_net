namespace csharp_neural_net;

public static class MatrixHelper
{
    public static double[,] GetRandomMatrix(int rows, int cols)
    { 
        var random = new Random();

        double[,] matrix = new double[rows, cols];
        for (int row = 0; row < rows; row++)
        { 
            for (int col = 0; col < cols; col++)
            {
                // values between -0.5 and 0.5
                matrix[row, col] = random.NextDouble() - 0.5;
            }   
        }

        return matrix;
    }

    public static double[,] MultiplyValue(double value, double[,] matrix)
    {
        for (var row = 0; row < matrix.GetLength(0); row++)
        {
            for (var col = 0; col < matrix.GetLength(1); col++)
            {
                matrix[row, col] *= value;
            }
        }

        return matrix;
    }

    public static double[,] MultiplyValues(double[,] matrix1, double[,] matrix2)
    {
        if (matrix1.GetLength(0) != matrix2.GetLength(0) || matrix1.GetLength(1) != matrix2.GetLength(1))
        {
            throw new Exception("Can not be multiplied");
        }

        var result = new double[matrix1.GetLength(0), matrix2.GetLength(1)];

        for (var row = 0; row < matrix1.GetLength(0); row++)
        {
            for (var col = 0; col < matrix1.GetLength(1); col++)
            {
                result[row, col] = matrix1[row, col] * matrix2[row, col];
            }
        }

        return result;
    }

    public static double[,] Subtract(double value, double[,] matrix)
    {
        for (var row = 0; row < matrix.GetLength(0); row++)
        {
            for (var col = 0; col < matrix.GetLength(1); col++)
            {
                matrix[row, col] -= value;
            }
        }

        return matrix;
    }

    public static double[,] Subtract(double[,] matrix1, double[,] matrix2)
    {
        if (matrix1.GetLength(0) != matrix2.GetLength(0) || matrix1.GetLength(1) != matrix2.GetLength(1))
        {
            throw new Exception("Matrices can not be subtracted");
        }

        double[,] matrix = new double[matrix1.GetLength(0), matrix1.GetLength(1)];

        for (var row = 0; row < matrix1.GetLength(0); row++)
        {
            for (var col = 0; col < matrix1.GetLength(1); col++)
            {
                matrix[row, col] = matrix1[row, col] - matrix2[row, col];
            }
        }

        return matrix;
    }
    public static double Sum(double[,] matrix)
    {
        var sum = 0d;

        for (var row = 0; row < matrix.GetLength(0); row++)
        { 
            for (var col = 0; col < matrix.GetLength(1); col++)
            {
                sum += matrix[row, col];
            }
        }

        return sum;
    }

    public static double[,] OneHotEncode(double[] y, int size)
    { 
        var oneHotMatrix = new double[y.Length, size];

        for (var i = 0; i < y.Length; i++)
        {
            oneHotMatrix[i, (int)y[i]] = 1;
        }

        return oneHotMatrix;
    }

    public static double[,] TransposeMatrix(double[,] matrix)
    { 
        var transposedMatrix = new double[matrix.GetLength(1), matrix.GetLength(0)];

        for (var row = 0; row < matrix.GetLength(0); row++)
        {
            for (var col = 0; col < matrix.GetLength(1); col++)
            {
                transposedMatrix[col, row] = matrix[row, col];
            }
        }

        return transposedMatrix;
    }

    public static double[,] GetBatch(double[,] data, int batch, int startRow)
    { 
        var batchData = new double[batch, data.GetLength(1)];

        for (var row = 0; row < batch; row++)
        {
            for (var col = 0; col < data.GetLength(1); col++)
            {
                batchData[row, col] = data[startRow + row, col];
            }
        }

        return batchData;
    }

    public static double[,] DotProduct(double[,] a, double[,] b)
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

    public static double[,] SoftMax(double[,] matrix)
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

    public static double[,] ReLU(double[,] matrix)
    { 
        for (var row = 0; row < matrix.GetLength(0); row++)
        {
            for (var col = 0; col < matrix.GetLength(1); col++)
            {
                matrix[row, col] = Math.Max(0, matrix[row, col]);
            }
        }

        return matrix;
    }

    public static double[,] ReLUDerivative(double[,] matrix)
    { 
        for (var row = 0; row < matrix.GetLength(0); row++)
        {
            for (var col = 0; col < matrix.GetLength(1); col++)
            {
                matrix[row, col] = matrix[row, col] > 0 ? 1 : 0;
            }
        }

        return matrix;
    }
}
