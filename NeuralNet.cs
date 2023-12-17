namespace csharp_neural_net
{
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

        public void GradientDescent()
        { 
            // Console.WriteLine(X);
            // PrintMatrix(X);

            // forward pass
            var linearResult1 = Linear(Weights1, X, Bias1);

            // Console.WriteLine("linearResult1");
            // PrintMatrix(linearResult1);

            var activiationResult1 = ReLU(linearResult1);

            var linearResult2 = Linear(Weights2, activiationResult1, Bias2);

            Console.WriteLine("linearResult2");
            PrintMatrix(linearResult2);

            var activationResult2 = SoftMax(linearResult2);

            Console.WriteLine("activationResult2");
            PrintMatrix(linearResult2);

            return;
            
            // backward pass
            var oneHotY = MatrixHelper.OneHotEncode(Y);
            var derivativeActivation2 = MatrixHelper.Subtract(activationResult2, oneHotY);

            var derivativeWeights2 = MatrixHelper.Multiply(1/M, DotProduct(derivativeActivation2, activiationResult1));
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
            for (var i = 0; i < matrix.GetLength(1); i++)
            {
                var sum = 0d;
                for (var j = 0; j < matrix.GetLength(0); j++)
                {
                    sum += (double)matrix[j, i];
                }

                Console.WriteLine($"sum: {sum}");
                Console.WriteLine($"sum: {Math.Exp(sum)}");


                for (var j = 0; j < matrix.GetLength(1); j++)
                {
                    matrix[i,j] = (double)Math.Exp(matrix[i, j]) / sum;
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
}
