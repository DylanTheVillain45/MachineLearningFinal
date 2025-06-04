using UnityEngine;

public class SineWaveManager : MonoBehaviour
{
    public static SineWaveManager instance;
    public GameObject sineLine;
    public GameObject sinPredict;
    private LineRenderer sinLineR;
    private LineRenderer sinPredictR;


    public int points = 100;
    public float xStart = -15f;
    public float xEnd = 10f;
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

    void Start()
    {
        sinLineR = sineLine.GetComponent<LineRenderer>();
        sinPredictR = sinPredict.GetComponent<LineRenderer>();
        DrawSineLine();
    }

    void DrawSineLine()
    {
        sinLineR.positionCount = 100;
        sinLineR.widthMultiplier = 0.2f;

        float step = (xEnd - xStart) / (points - 1);
        for (int i = 0; i < points; i++)
        {
            float x = xStart + i * step;
            float y = Mathf.Sin(x) * 5;
            sinLineR.SetPosition(i, new Vector3(x, y, 0));

        }
    }

}
