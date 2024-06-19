using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Globalization;

public class HandTracking : MonoBehaviour
{
    public UDPReceive udpReceive;
    public GameObject[] handPointsLeft;
    public GameObject[] handPointsRight;
    public GameObject[] footPointsLeft;
    public GameObject[] footPointsRight;
    public GameObject footLeft;
    public GameObject footRight;

    // Update is called once per frame
    void Update()
    {
        string data = udpReceive.data;

        if (data.Length > 0)
        {
            // [foot_L, foot_R, hand_L, hand_R]
            data = RemoveBrackets(data);
            
            string[] points = data.Split(',');


            Debug.Log(data);
            // 544, 269, 0, 566, 333, -24, 567, 402, -31, 559, 459, -36, 554, 503, -40, 487, 420, -2, 450, 469, -10, 423, 494, -20, 398, 517, -29, 458, 398, 0, 413, 445, -6, 383, 472, -19, 357, 495, -30, 439, 369, -2, 394, 408, -10, 368, 434, -21, 347, 457, -29, 425, 333, -7, 383, 354, -16, 356, 371, -21, 334, 390, -25


            // FOOT LEFT
            int indexStart = 0;
            int indexEnd = data.IndexOf(']') + 1;

            string dataFootLeft = data[indexStart..indexEnd];
            dataFootLeft = RemoveBrackets(dataFootLeft);

            if (dataFootLeft.Length > 0)
            {
                string[] pointsFootLeft = dataFootLeft.Split(',');
                ParseFootKeypoints("left", pointsFootLeft);
            }

            
            data = data[indexEnd..data.Length];


            // FOOT RIGHT
            indexStart = data.IndexOf('[');
            indexEnd = data.IndexOf(']') + 1;

            string dataFootRight = data[indexStart..indexEnd];
            dataFootRight = RemoveBrackets(dataFootRight);

            if (dataFootRight.Length > 0)
            {
                string[] pointsFootRight = dataFootRight.Split(',');
                ParseFootKeypoints("right", pointsFootRight);
            }


            data = data[indexEnd..data.Length];


            // HAND LEFT
            indexStart = data.IndexOf('[');
            indexEnd = data.IndexOf(']') + 1;

            string dataHandLeft = data[indexStart..indexEnd];
            dataHandLeft = RemoveBrackets(dataHandLeft);

            if (dataHandLeft.Length > 0)
            {
                string[] pointsHandLeft = dataHandLeft.Split(',');
                ParseHandKeypoints("left", pointsHandLeft);
            }


            data = data[indexEnd..data.Length];


            // HAND RIGHT
            indexStart = data.IndexOf('[');
            indexEnd = data.IndexOf(']') + 1;

            string dataHandRight = data[indexStart..indexEnd];
            dataHandRight = RemoveBrackets(dataHandRight);

            if (dataHandRight.Length > 0)
            {
                string[] pointsHandRight = dataHandRight.Split(',');
                ParseHandKeypoints("right", pointsHandRight);
            }
        }
    }

    string RemoveBrackets(string data)
    {
        string dataUpdated = data.Remove(0, 1); // [ on the start
        dataUpdated = dataUpdated.Remove(dataUpdated.Length - 1, 1); // ] on the end
        return dataUpdated;
    }

    Vector3 GetFootPosition(string[] data, int i)
    {
        float x = float.Parse(data[i * 3], CultureInfo.InvariantCulture) / 100 - 2;
        float y = float.Parse(data[i * 3 + 1], CultureInfo.InvariantCulture) - 2;
        float z = float.Parse(data[i * 3 + 2], CultureInfo.InvariantCulture) / 100;
        
        return new Vector3(x, y, z);
    }

    Vector3 GetHandPosition(string[] data, int i)
    {
        float x = float.Parse(data[i * 3], CultureInfo.InvariantCulture) / 100 - 2;
        float y = float.Parse(data[i * 3 + 1], CultureInfo.InvariantCulture) / 100;
        float z = float.Parse(data[i * 3 + 2], CultureInfo.InvariantCulture) / 100;
        
        return new Vector3(x, y, z);
    }

    void ParseFootKeypoints(string side, string[] data)
    {
        if (side == "left")
        {
            for (int i = 0; i < 2; i++)
            {
                footPointsLeft[i].transform.localPosition = GetFootPosition(data, i);
                footLeft.transform.position = footPointsLeft[i].transform.localPosition;
            }
        }

        if (side == "left")
        {
            for (int i = 0; i < 2; i++)
            {
                footPointsRight[i].transform.localPosition = GetFootPosition(data, i);
                footRight.transform.position = footPointsRight[i].transform.localPosition;
            }
        }

    }

    void ParseHandKeypoints(string side, string[] data)
    {
        // x1, y1, z1, x2, y2, z2...

        if (side == "left")
        {
            for (int i = 0; i < 21; i++)
                handPointsLeft[i].transform.localPosition = GetHandPosition(data, i);
        }

        if (side == "right")
        {
            for (int i = 0; i < 21; i++)
                handPointsRight[i].transform.localPosition = GetHandPosition(data, i);
        }
    }
}
