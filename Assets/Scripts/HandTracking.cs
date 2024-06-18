using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HandTracking : MonoBehaviour
{
    public UDPReceive udpReceive;
    public GameObject[] handPointsLeft;
    public GameObject[] handPointsRight;

    // Update is called once per frame
    void Update()
    {
        string data = udpReceive.data;

        if (data.Length > 0)
        {
            data = data.Remove(0, 1); // [ on the start
            data = data.Remove(data.Length - 1, 1); // ] on the end
            string[] points = data.Split(',');

            //string side = points[0];
            //points = points[1..points.Length];
            
            for (int i = 0; i < 21; i++)
            {
                // x1, y1, z1, x2, y,2, z2...
                float x = float.Parse(points[i * 3]) / 100;
                float y = float.Parse(points[i * 3 + 1]) / 100;
                float z = float.Parse(points[i * 3 + 2]) / 100;
                
                //if (side == "Left")
                handPointsLeft[i].transform.localPosition = new Vector3(x, y, z);
                //else 
                //    handPointsRight[i].transform.localPosition = new Vector3(x, y, z);
            }
        }
    }
}
