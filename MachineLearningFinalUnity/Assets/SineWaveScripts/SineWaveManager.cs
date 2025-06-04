using UnityEngine;

public class SineWaveManager : MonoBehaviour
{
    public static SineWaveManager instance;

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
}
