using System;

public static class NeuralNetHelpers
{
    public static float[] MatMulAndBias(float[] X, float[,] W, float[,] B)
    {
        int XLen = X.Length;
        int WRows = W.GetLength(1);
        int WCols = W.GetLength(0);
        int BRows = B.GetLength(0);
        int BCols = B.GetLength(1);

        if (XLen != WRows || WCols != BRows || BCols != 1)
        {
            DigitClassifierManager.instance.RaiseError("Matrix sizes don't match");
        }

        float[] result = new float[WCols];

        for (int i = 0; i < WCols; i++)
        {
            float sum = 0;
            for (int j = 0; j < XLen; j++)
            {
                sum += X[j] * W[i, j];
            }
            result[i] += sum + B[i, 0];
        }

        return result;
    }

    public static float[,] TransposeVector(float[] vector)
    {
        float[,] matrix = new float[vector.Length, 1];

        for (int i = 0; i < vector.Length; i++)
        {
            matrix[i, 0] = vector[i];
        }

        return matrix;
    }

    public static float[] ReLU(float[] Z)
    {
        float[] A = new float[Z.Length];

        for (int i = 0; i < Z.Length; i++)
        {
            float v = Z[i];
            A[i] = (v > 0f) ? v : 0f;
        }

        return A;
    }

    public static float[] ReLUDeriv(float[] Z)
    {
        float[] dZ = new float[Z.Length];

        for (int i = 0; i < Z.Length; i++)
        {
            float v = Z[i];
            dZ[i] = (v > 0f) ? 1f : 0f;
        }

        return dZ;
    }

    public static float[] SoftMax(float[] Z)
    {
        float maxVal = float.MinValue;
        for (int i = 0; i < Z.Length; i++)
            if (Z[i] > maxVal) maxVal = Z[i];

        float sum = 0f;
        float[] exps = new float[Z.Length];
        for (int i = 0; i < Z.Length; i++)
        {
            exps[i] = (float)Math.Exp(Z[i] - maxVal);
            sum += exps[i];
        }

        float[] outp = new float[Z.Length];
        for (int i = 0; i < Z.Length; i++)
            outp[i] = exps[i] / sum;

        return outp;
    }

    public static int ArgMax(float[] A)
    {
        int highestNum = 0;
        float highestVal = 0;
        for (int i = 0; i < 10; i++)
        {
            if (A[i] > highestVal)
            {
                highestNum = i;
                highestVal = A[i];
            }
        }
        return highestNum;
    }

    public static float[,] SetRandWeights(int input, int layerHeight)
    {
        Random random = new Random();
        float[,] randWeights = new float[layerHeight, input];

        for (int i = 0; i < layerHeight; i++)
        {
            for (int j = 0; j < input; j++)
            {
                randWeights[i, j] = (float)random.NextDouble();
            }
        }

        return randWeights;
    }

    public static float[,] SetZeroBias(int layerHeight)
    {
        return new float[layerHeight, 1];
    }
}