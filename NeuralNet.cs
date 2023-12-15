namespace csharp_neural_net
{
    public class NeuralNet
    {
        public float[,] X { get; set; }
        public float[] Y { get; set; }
        public float LearningRate { get; set; }
        public float[,] Weights1 { get; set; }
        public float[,] Bias1 { get; set; }

        public float[,] Weights2 { get; set; }
        public float[,] Bias2 { get; set; }

        public NeuralNet(float[,] x, float[] y, float learningRate)
        {
            X = x;
            Y = y;
            LearningRate = learningRate;
        }

        public void InitParams()
        {
            Weights1 = GetRandomMatrix(10, 784);
            Bias1 = GetRandomMatrix(10, 1);
            Weights2 = GetRandomMatrix(10, 10);
            Bias2 = GetRandomMatrix(10, 1);
        }

        public void GradientDescent()
        { 
            // forward pass
            var linearResult1 = Linear(Weights1, X, Bias1);
            var activiationResult1 = ReLU(linearResult1);
            var linearResult2 = Linear(Weights2, activiationResult1, Bias2);
            var activiationResult2 = SoftMax(linearResult2);

            // backward pass
            var oneHotY = OneHotEncode(Y);
        }

        private float[,] GetRandomMatrix(int rows, int cols)
        { 
            Random random = new Random();

            float[,] matrix = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            { 
                for (int j = 0; j < cols; j++)
                {
                    // values between -0.5 and 0.5
                    matrix[i, j] = (float)(random.NextDouble() - 0.5);
                }   
            }

            return matrix;
        }

        private float[,] OneHotEncode(float[] y)
        { 
            var oneHotMatrix = new float[y.Length, (int)y.Max() + 1];

            for (var i = 0; i < y.Length; i++)
            {
                oneHotMatrix[i, (int)y[i]] = 1;
            }

            return oneHotMatrix;
        }

        private float[,] ReLU(float[,] matrix)
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

        private float[,] ReLUDerivative(float[,] matrix)
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

        private float[,] SoftMax(float[,] matrix)
        { 
            for (var i = 0; i < matrix.GetLength(0); i++)
            {
                var sum = 0f;
                for (var j = 0; j < matrix.GetLength(1); j++)
                {
                    sum += matrix[i, j];
                }

                for (var j = 0; j < matrix.GetLength(1); j++)
                {
                    matrix[i,j] = (float)Math.Exp(matrix[i,j]) / sum;
                }
            }

            return matrix;
        }

        private float[,] Linear(float[,] weights, float[,] x, float[,] bias)
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

        private float[,] DotProduct(float[,] a, float[,] b)
        {
            float[,] result = new float[a.GetLength(0), b.GetLength(1)];

            for (int i = 0; i < a.GetLength(0); i++)
            {
                for (int j = 0; j < b.GetLength(1); j++)
                {
                    float sum = 0;
                    for (int k = 0; k < a.GetLength(1); k++)
                    {
                        sum += a[i, k] * b[k, j];
                    }

                    result[i, j] = sum;
                }   
            }

            return result;
        }
    }
}
