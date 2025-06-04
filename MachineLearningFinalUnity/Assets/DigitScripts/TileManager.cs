using UnityEngine;
using System.Collections.Generic;
using System;
public class TileManager : MonoBehaviour
{
    public int X, Y;
    public float alphaVal;

    void OnMouseOver()
    {
        if (Input.GetMouseButton(0))
        {
            Highlight(this.gameObject, 1);
            HighlightAdjacents();
        }
    }

    void HighlightAdjacents()
    {
        List<(GameObject, float)> adjacentTiles = new List<(GameObject, float)>();
        int radius = 1;

        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                if (dx == 0 && dy == 0) continue; // Skip the center tile (self)

                int newY = Y + dy;
                int newX = X + dx;

                if (newY >= 0 && newY < DigitClassifierManager.instance.sizeY &&
                    newX >= 0 && newX < DigitClassifierManager.instance.sizeX)
                {
                    adjacentTiles.Add((DigitClassifierManager.instance.TileDictionary[(newY, newX)], (float)Math.Sqrt(dx * dx + dy * dy)));
                }
            }
        }

        foreach ((GameObject tile, float dist) in adjacentTiles)
        {
            if (tile.GetComponent<TileManager>().alphaVal < 1)
            {
                float val = UnityEngine.Random.Range(0.8f, 1f) / (dist * dist * dist * 4);
                float newAlpha = tile.GetComponent<TileManager>().alphaVal + val > 1 ? 1 : tile.GetComponent<TileManager>().alphaVal + val;
                Highlight(tile, newAlpha);
            }
        }
    }

    public void Highlight(GameObject tile, float intensity)
    {
        tile.GetComponent<TileManager>().alphaVal = intensity;
        Color grayColor = new Color(intensity, intensity, intensity);
        tile.GetComponent<SpriteRenderer>().color = grayColor;
    }
}
