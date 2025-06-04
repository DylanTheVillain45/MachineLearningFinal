using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System;
public class DigitClassifierManager : MonoBehaviour
{
    public static DigitClassifierManager instance;
    void Awake()
    {
        if (instance != null)
        {
            Destroy(this.gameObject);
        }
        else
        {
            instance = this;
        }
    }

    public GameObject TilePref;
    public Transform board;

    public int sizeX;
    public int sizeY;
    public float tileScale;

    public GameObject[] textVals = new GameObject[10];
    DigitClassifier Net;

    public Dictionary<(int, int), GameObject> TileDictionary = new Dictionary<(int, int), GameObject>();
    public GameObject[,] TileArray;

    void SetBoard()
    {
        TileArray = new GameObject[sizeY, sizeX];
        for (int i = 0; i < sizeX; i++)
        {
            for (int j = 0; j < sizeY; j++)
            {
                GameObject tileObj = Instantiate(TilePref, board);
                TileArray[i, j] = tileObj;
                tileObj.name = $"Tile - {i} {j}";
                tileObj.transform.localScale = new Vector3(tileScale, tileScale, 1);
                tileObj.transform.position = new Vector2((j - sizeX / 2) * tileScale, (-i + sizeY / 2) * tileScale);
                TileManager tileManager = tileObj.GetComponent<TileManager>();
                (tileManager.X, tileManager.Y) = (j, i);

                tileManager.alphaVal = 0;
                TileDictionary.Add((i, j), tileObj);
            }
        }
    }

    public void ResetBoard()
    {
        foreach (var kvp in TileDictionary)
        {
            SpriteRenderer tileSprite = kvp.Value.GetComponent<SpriteRenderer>();
            TileManager tileManager = kvp.Value.GetComponent<TileManager>();

            tileSprite.color = Color.black;
            tileManager.alphaVal = 0;
        }

        foreach (GameObject textVal in textVals)
        {
            textVal.GetComponent<Text>().text = $"0%";
        }
    }

    float[] ParseData() {
        float[] data = new float[784];
        for (int i = 0; i < 28; i++)
        {
            for (int j = 0; j < 28; j++)
            { 
                data[i * 28 + j] = TileArray[i, j].GetComponent<TileManager>().alphaVal;   
            }
        }
        // Debug.Log($"{data[0]}, {data[27]}, {data[783 - 27]}, {data[783]}");
        return data;
    }

    public void Random()
    {
        ResetBoard();
        float[,] mnist = Net.ReadCSV("../DigitClassifierPython/mnist_test.csv");

        int totalSamples = mnist.GetLength(0);
        int featuresPerSample = mnist.GetLength(1);

        if (featuresPerSample != 785)
        {
            RaiseError("Mnist should have 785 samples");
            return;
        }

        System.Random rng = new System.Random();
        int idx = rng.Next(totalSamples);

        int label = (int)mnist[idx, 0];
        Message($"NUM {label} IDX {idx}");

        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                int pixelIndex = y * 28 + x;
                float pixelVal = mnist[idx, pixelIndex + 1] / 255f;

                GameObject tileObj = TileArray[y, x];
                TileManager tile = tileObj.GetComponent<TileManager>();
                tile.Highlight(tileObj, pixelVal);
            }
        }

    }

    public void Run()
    {
        float[] data = ParseData();
        float[] percentages = Net.CalculatePercentages(data);

        for (int i = 0; i < textVals.Length; i++)
        {
            textVals[i].GetComponent<Text>().text = $"{Math.Round(percentages[i] * 100, 1)}%";
        }
    }

    void Start()
    {
        Net = new DigitClassifier();
        SetBoard();
    }

    public void Message(string error)
    {
        Debug.Log(error);
    }

    public void RaiseError(string error)
    {
        Debug.LogError(error);
    }
}
