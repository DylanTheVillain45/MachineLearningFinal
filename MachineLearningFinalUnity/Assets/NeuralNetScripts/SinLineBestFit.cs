public class SinFit
{
    public float[,] W1, W2, W3, b1, b2, b3;
    public int layer1;
    public int layer2;

    public SinFit(int layer1, int layer2)
    {
        this.layer1 = layer1;
        this.layer2 = layer2;
    }

    public void SetWeightsAndBiases()
    {
        W1 = NeuralNetHelpers.SetRandWeights(100, layer1);
        W2 = NeuralNetHelpers.SetRandWeights(layer1, layer2);
        W3 = NeuralNetHelpers.SetRandWeights(layer2, 1);
        b1 = NeuralNetHelpers.SetZeroBias(layer1);
        b2 = NeuralNetHelpers.SetZeroBias(layer2);
        b3 = NeuralNetHelpers.SetZeroBias(1);
    }

    // public void Train(float[] X, float[] Y, int epochs, float lr)
    // {
    //     if (X.Length != Y.Length)
    //     {
    //         return;
    //     }
    //     int N = X.Length;

    //     float[] data = new float[1];

    //     for (int e = 0; e < epochs; e++)
    //     {
    //         for (int idx = 0; idx < N; idx++)
    //         {
    //             data[0] = X[idx];
    //             float yTrue = Y[idx];
    //             float[] Z1, A2, Z2, A2, Z3, A3;

    //             (Z1, A2, Z2, A2, Z3, A3) = ForwardPath(X);

    //             (float[])

    //         }
    //     }
    // }

    (float[], float[], float[], float[], float[], float[]) ForwardPath(float[] data)
    {
        float[] Z1 = NeuralNetHelpers.MatMulAndBias(data, W1, b1);
        float[] A1 = NeuralNetHelpers.ReLU(Z1);
        float[] Z2 = NeuralNetHelpers.MatMulAndBias(A1, W2, b2);
        float[] A2 = NeuralNetHelpers.ReLU(Z2);
        float[] Z3 = NeuralNetHelpers.MatMulAndBias(A2, W3, b3);
        float[] A3 = Z3;

        return (Z1, A2, Z2, A2, Z3, A3);
    }
}