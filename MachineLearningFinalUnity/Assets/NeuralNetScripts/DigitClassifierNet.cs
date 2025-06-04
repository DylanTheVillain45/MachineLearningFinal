using System.IO;

public class DigitClassifier
{
    public float[,] W1, W2, W3, b1, b2, b3;

    public DigitClassifier() {
        SetWeightsAndBiases();
    }

    public void SetWeightsAndBiases()
    {
        W1 = ReadCSV("../DigitClassifierPython/Weights/W1.csv");
        W2 = ReadCSV("../DigitClassifierPython/Weights/W2.csv");
        W3 = ReadCSV("../DigitClassifierPython/Weights/W3.csv");
        b1 = ReadCSV("../DigitClassifierPython/Weights/b1.csv");
        b2 = ReadCSV("../DigitClassifierPython/Weights/b2.csv");
        b3 = ReadCSV("../DigitClassifierPython/Weights/b3.csv");

    }

    public void DebugOneRow()
    {
        float[,] mnist = ReadCSV("../DigitClassifierPython/mnist_test.csv");
        int idx = 0;

        float[] data = new float[784];
        for (int j = 0; j < 784; j++)
        {
            data[j] = mnist[idx, j + 1] / 255f;
        }

        float[] Z1_cs = NeuralNetHelpers.MatMulAndBias(data, W1, b1);
        float[] A1_cs = NeuralNetHelpers.ReLU(Z1_cs);
        float[] Z2_cs = NeuralNetHelpers.MatMulAndBias(A1_cs, W2, b2);
        float[] A2_cs = NeuralNetHelpers.ReLU(Z2_cs);
        float[] Z3_cs = NeuralNetHelpers.MatMulAndBias(A2_cs, W3, b3);
        float[] A3_cs = NeuralNetHelpers.SoftMax(Z3_cs);

        for (int i = 0; i < 5; i++)
        {
            DigitClassifierManager.instance.Message($"C# Z1_cs[{i}] = {Z1_cs[i]:F6}");
        }
        for (int i = 0; i < 5; i++)
        {
            DigitClassifierManager.instance.Message($"C# A1_cs[{i}] = {A1_cs[i]:F6}");
        }
        for (int i = 0; i < 5; i++)
        {
            DigitClassifierManager.instance.Message($"C# Z2_cs[{i}] = {Z2_cs[i]:F6}");
        }
        for (int i = 0; i < 5; i++)
        {
            DigitClassifierManager.instance.Message($"C# A2_cs[{i}] = {A2_cs[i]:F6}");
        }
        for (int i = 0; i < 5; i++)
        {
            DigitClassifierManager.instance.Message($"C# Z3_cs[{i}] = {Z3_cs[i]:F6}");
        }
        for (int i = 0; i < 5; i++)
        {
            DigitClassifierManager.instance.Message($"C# A3_cs[{i}] = {A3_cs[i]:F6}");
        }
        DigitClassifierManager.instance.Message(NeuralNetHelpers.ArgMax(A3_cs).ToString());
    }

    public void Debug(float[] Z1, float[] A1, float[] Z2, float[] A2, float[] Z3, float[] A3)
    {
        for (int i = 0; i < 5; i++)
        {
            DigitClassifierManager.instance.Message($"C# Z1_cs[{i}] = {Z1[i]:F6}");
        }
        for (int i = 0; i < 5; i++)
        {
            DigitClassifierManager.instance.Message($"C# A1_cs[{i}] = {A1[i]:F6}");
        }
        for (int i = 0; i < 5; i++)
        {
            DigitClassifierManager.instance.Message($"C# Z2_cs[{i}] = {Z2[i]:F6}");
        }
        for (int i = 0; i < 5; i++)
        {
            DigitClassifierManager.instance.Message($"C# A2_cs[{i}] = {A2[i]:F6}");
        }
        for (int i = 0; i < 5; i++)
        {
            DigitClassifierManager.instance.Message($"C# Z3_cs[{i}] = {Z3[i]:F6}");
        }
        for (int i = 0; i < 5; i++)
        {
            DigitClassifierManager.instance.Message($"C# A3_cs[{i}] = {A3[i]:F6}");
        }
        DigitClassifierManager.instance.Message(NeuralNetHelpers.ArgMax(A3).ToString());
        
    }

    public float[,] ReadCSV(string path)
    {
        var lines = System.IO.File.ReadAllLines(path);
        int rows = lines.Length;
        int cols = lines[0].Split(',').Length;

        float[,] result = new float[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            var values = lines[i].Split(',');
            for (int j = 0; j < cols; j++)
            {
                result[i, j] = float.Parse(values[j]);
            }
        }

        return result;
    }
    public float[] CalculatePercentages(float[] data)
    {
        float[] row = ForwardPath(data);

        return row;
    }

    float[] ForwardPath(float[] data)
    {
        float[] Z1 = NeuralNetHelpers.MatMulAndBias(data, W1, b1);
        float[] A1 = NeuralNetHelpers.ReLU(Z1);
        float[] Z2 = NeuralNetHelpers.MatMulAndBias(A1, W2, b2);
        float[] A2 = NeuralNetHelpers.ReLU(Z2);
        float[] Z3 = NeuralNetHelpers.MatMulAndBias(A2, W3, b3);
        float[] A3 = NeuralNetHelpers.SoftMax(Z3);

        Debug(Z1, A1, Z2, A2, Z3, A3);

        return A3;
    }
}