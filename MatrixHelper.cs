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

        for (var i = 0; i < batch; i++)
        {
            for (var j = 0; j < data.GetLength(1); j++)
            {
                batchData[i, j] = data[startRow + i, j];
            }
        }

        return batchData;
    }
}
