using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/along_position")]



public class along_position : Randomizer
{
    // object variables initialisation
    public GameObjectParameter prefabs_mat1; // objects folder
    public GameObjectParameter prefabs_mat2; // track folder
    private GameObject currentInstance1; // objects
    private GameObject currentInstance2; // track
    // public FloatParameter xShift;
    public FloatParameter objectScale;
    public Vector3Parameter trackScale;
    public float spawnCollisionCheckRadius=0.5f;   
    // public FloatParameter trackScaleX;
    // public FloatParameter trackScaleY;
    // public FloatParameter trackScaleZ;
    public FloatParameter object_x;
    public FloatParameter object_y;
    public Camera mainCamera;
    public float camera_radius=10f;
    public float sphere_limit=7.5f;
    public Material material1;
    public Material material2;

    protected override void OnIterationStart()
    {   
        // // function to create a random point along a line
        // static float RandomZ(float start, float end)
        // {
        //     // Generate a random distance along the line segment
        //     float distance = Random.Range(0f, 1f);
        //     // Calculate the point along the line segment at the generated distance
        //     float point_z = start + distance * (end - start);
        //     return point_z;
        // }
        // function to create a random point along a line
        static Vector3 RandomPointOnLine(float x, float y, float z)
        {
            Vector3 point = new Vector3(x,y,z);
            return point;
        }

        // object variables code
        // tracks 
        currentInstance1 = GameObject.Instantiate(prefabs_mat1.Sample()); // track
        currentInstance1.transform.localScale = Vector3.one * objectScale.Sample();

        currentInstance2 = GameObject.Instantiate(prefabs_mat2.Sample()); // track
        currentInstance2.transform.position = new Vector3(0,0,0);
        currentInstance2.transform.localScale = trackScale.Sample(); 
        
        // float track_length = currentInstance2.transform.localScale.z;
        // float startPoint = -track_length/2;
        // float endPoint = track_length/2;
        float randomZ = Random.Range(-2,-5);
        Vector3 randomPoint = RandomPointOnLine(object_x.Sample(),object_y.Sample(),randomZ);
        currentInstance1.transform.position = randomPoint;
        // Debug.Log(startPoint);
        // Debug.Log(endPoint);
        // Debug.Log(randomZ);
        // Debug.Log(randomPoint);

        int objectCount = Random.Range(2,6);
        for (int loop=0;loop<objectCount;loop++)
        {
            // float startPoint1 = -track_length/2-1;
            // float endPoint1 = track_length/2-1;
            float random_step = Random.Range(1.5f,2);
            randomZ += random_step;
            Vector3 spawnPoint = RandomPointOnLine(object_x.Sample(),object_y.Sample(),randomZ);
            GameObject.Instantiate(currentInstance1,spawnPoint,Random.rotation);
            // if (!Physics.CheckSphere(spawnPoint,spawnCollisionCheckRadius))
            // {
            //     GameObject.Instantiate(currentInstance1,spawnPoint,Random.rotation);
            // }
        }

        // camera randomisation code
        mainCamera.transform.position = Random.onUnitSphere * camera_radius;
        // formula for sphere x2+y2+z2=r2, r=10: x2+y2+z2=100
        // set a limit, e.g. limit=7.5
        // while (Mathf.Abs(mainCamera.transform.position.x)>sphere_limit | Mathf.Abs(mainCamera.transform.position.y)>sphere_limit | Mathf.Abs(mainCamera.transform.position.z)>sphere_limit)
        while (Mathf.Abs(mainCamera.transform.position.y)<sphere_limit | mainCamera.transform.position.y<0)
        {
            mainCamera.transform.position = Random.onUnitSphere * camera_radius;
        }
        // mainCamera.transform.LookAt(currentInstance1.transform); //look at gameobject position
        mainCamera.transform.LookAt(new Vector3(0,0,0)); // can set a manual point to look at

        // material randomisation code
        material1.color = Random.ColorHSV();
        material2.color = Random.ColorHSV();
    }

    protected override void OnIterationEnd()
    {
        GameObject[] GameObjects = (GameObject.FindObjectsOfType<GameObject>() as GameObject[]);
        for (int i = 0; i < GameObjects.Length; i++)
        {
            if (GameObjects[i].name==currentInstance1.name || GameObjects[i].name==currentInstance1.name+"(Clone)" || GameObjects[i].name==currentInstance2.name || GameObjects[i].name==currentInstance2.name+"(Clone)")
            {
                GameObject.Destroy(GameObjects[i]);
            }
        }
    }

}

// using System;
// using System.IO;
// using System.Collections;
// using System.Collections.Generic;
// using Random=UnityEngine.Random;
// using UnityEngine;
// using UnityEngine.Perception.Randomization.Parameters;
// using UnityEngine.Perception.Randomization.Randomizers;

// [Serializable]
// [AddRandomizerMenu("Perception/along_position")]



// public class along_position : Randomizer
// {
//     // object variables initialisation
//     public GameObjectParameter prefabs_mat1; // objects folder
//     public GameObjectParameter prefabs_mat2; // track folder
//     private GameObject currentInstance1; // objects
//     private GameObject currentInstance2; // track
//     // public FloatParameter xShift;
//     public FloatParameter objectScale;
//     public Vector3Parameter trackScale;
//     public int objectCount;
//     public float spawnCollisionCheckRadius=0.5f;   
//     // public FloatParameter trackScaleX;
//     // public FloatParameter trackScaleY;
//     // public FloatParameter trackScaleZ;
//     public FloatParameter object_x;
//     public FloatParameter object_y;
//     public Camera mainCamera;
//     public float camera_radius=10f;
//     public float sphere_limit=7.5f;
//     public Material material1;
//     public Material material2;

//     protected override void OnIterationStart()
//     {   
//         // function to create a random point along a line
//         static float RandomZ(float start, float end)
//         {
//             // Generate a random distance along the line segment
//             float distance = Random.Range(0f, 1f);
//             // Calculate the point along the line segment at the generated distance
//             float point_z = start + distance * (end - start);
//             return point_z;
//         }
//         // function to create a random point along a line
//         static Vector3 RandomPointOnLine(float x, float y, float z)
//         {
//             Vector3 point = new Vector3(x,y,z);
//             return point;
//         }

//         // object variables code
//         // tracks 
//         currentInstance1 = GameObject.Instantiate(prefabs_mat1.Sample()); // track
//         currentInstance1.transform.localScale = Vector3.one * objectScale.Sample();

//         currentInstance2 = GameObject.Instantiate(prefabs_mat2.Sample()); // track
//         currentInstance2.transform.position = new Vector3(0,0,0);
//         currentInstance2.transform.localScale = trackScale.Sample(); 
        
//         float track_length = currentInstance2.transform.localScale.z;
//         float startPoint = -track_length/2;
//         float endPoint = track_length/2;
//         float randomZ = RandomZ(startPoint, endPoint);
//         Vector3 randomPoint = RandomPointOnLine(object_x.Sample(),object_y.Sample(),randomZ);
//         currentInstance1.transform.position = randomPoint;
//         Debug.Log(startPoint);
//         Debug.Log(endPoint);
//         Debug.Log(randomZ);
//         Debug.Log(randomPoint);

        
//         for (int loop=0;loop<objectCount;loop++)
//         {
//             float startPoint1 = -track_length/2-1;
//             float endPoint1 = track_length/2-1;
//             float randomZ1 = RandomZ(startPoint1, endPoint1);
//             Vector3 spawnPoint = RandomPointOnLine(object_x.Sample(),object_y.Sample(),randomZ1);
//             GameObject.Instantiate(currentInstance1,spawnPoint,Random.rotation);
//             // if (!Physics.CheckSphere(spawnPoint,spawnCollisionCheckRadius))
//             // {
//             //     GameObject.Instantiate(currentInstance1,spawnPoint,Random.rotation);
//             // }
//         }

//         // camera randomisation code
//         mainCamera.transform.position = Random.onUnitSphere * camera_radius;
//         // formula for sphere x2+y2+z2=r2, r=10: x2+y2+z2=100
//         // set a limit, e.g. limit=7.5
//         // while (Mathf.Abs(mainCamera.transform.position.x)>sphere_limit | Mathf.Abs(mainCamera.transform.position.y)>sphere_limit | Mathf.Abs(mainCamera.transform.position.z)>sphere_limit)
//         while (Mathf.Abs(mainCamera.transform.position.y)<sphere_limit | mainCamera.transform.position.y<0)
//         {
//             mainCamera.transform.position = Random.onUnitSphere * camera_radius;
//         }
//         // mainCamera.transform.LookAt(currentInstance1.transform); //look at gameobject position
//         mainCamera.transform.LookAt(new Vector3(0,0,0)); // can set a manual point to look at

//         // material randomisation code
//         material1.color = Random.ColorHSV();
//         material2.color = Random.ColorHSV();
//     }

//     protected override void OnIterationEnd()
//     {
//         GameObject[] GameObjects = (GameObject.FindObjectsOfType<GameObject>() as GameObject[]);
//         for (int i = 0; i < GameObjects.Length; i++)
//         {
//             if (GameObjects[i].name==currentInstance1.name || GameObjects[i].name==currentInstance1.name+"(Clone)" || GameObjects[i].name==currentInstance2.name || GameObjects[i].name==currentInstance2.name+"(Clone)")
//             {
//                 GameObject.Destroy(GameObjects[i]);
//             }
//         }
//     }

// }

        // // function to create a random point along a line
        // static Vector3 RandomPointOnLine(Vector3 start, Vector3 end)
        // {
        //     // Generate a random distance along the line segment
        //     float distance = Random.Range(0f, 1f);
        //     // Calculate the point along the line segment at the generated distance
        //     Vector3 point = start + distance * (end - start);
        //     return point;
        // }

// using System;
// using System.IO;
// using System.Collections;
// using System.Collections.Generic;
// using Random=UnityEngine.Random;
// using UnityEngine;
// using UnityEngine.Perception.Randomization.Parameters;
// using UnityEngine.Perception.Randomization.Randomizers;

// [Serializable]
// [AddRandomizerMenu("Perception/along_position")]
// public class along_position : Randomizer
// {
//     // output colour filepaths
//     string filePath1 = "/Users/haidiazaman/Desktop/FYP/dataset_v4/colors_txt_files/colors_along_position1.txt";
//     string filePath2 = "/Users/haidiazaman/Desktop/FYP/dataset_v4/colors_txt_files/colors_along_position2.txt";

//     // object variables initialisation
//     public GameObjectParameter prefabs_mat1; // track folder
//     public GameObjectParameter prefabs_mat2;
//     private GameObject currentInstance1; // track
//     private GameObject second_object;
//     private GameObject currentInstance2;
//     private GameObject currentInstance3;
//     private GameObject currentInstance4;
//     private GameObject currentInstance5;
//     public FloatParameter xShift;
//     public FloatParameter trackScaleX;
//     public FloatParameter trackScaleY;
//     public FloatParameter trackScaleZ;
//     private float cheat_height = 0.1f;
//     private float x11,y11,z11;
//     private float x21,y21,z21;
//     private float x31,y31,z31;
//     private float x41,y41,z41;
//     private float x51,y51,z51;

//     // camera variables initialisation
//     public Camera mainCamera;
//     public Vector3Parameter cameraPosition;
//     public Vector3Parameter cameraRot;
//     // material variables initialisation
//     public Material material1;
//     public Material material2;
//     private Color[] colorList = { 
//         Color.red, 
//         Color.green, 
//         Color.blue, 
//         Color.magenta, //purple
//         new Color(0.5f,0.5f,1,1), //light blue
//         new Color(1.0f,0.64f,0.0f,1), //orange
//         new Color(0.75f,0.75f,0.75f,1), //dark grey
//         Color.yellow
//         };
//     private Dictionary<string, string> colour_identifier =  new Dictionary<string, string>()
//         {
//             {"RGBA(1.000, 0.000, 0.000, 1.000)","red"},
//             {"RGBA(0.000, 1.000, 0.000, 1.000)","green"},
//             {"RGBA(0.000, 0.000, 1.000, 1.000)","dark blue"},
//             {"RGBA(1.000, 0.000, 1.000, 1.000)","purple"},
//             {"RGBA(0.500, 0.500, 1.000, 1.000)","light blue"},
//             {"RGBA(1.000, 0.640, 0.000, 1.000)","orange"},
//             {"RGBA(0.750, 0.750, 0.750, 1.000)","dark grey"},
//             {"RGBA(1.000, 0.922, 0.016, 1.000)","yellow"}
//         }; 

//     protected override void OnIterationStart()
//     {   
//         // object variables code
//         // track 
//         currentInstance2 = GameObject.Instantiate(prefabs_mat2.Sample()); // track
//         currentInstance2.transform.position = new Vector3(0,0.25f,0);
//         currentInstance2.transform.localScale = new Vector3(trackScaleX.Sample(),trackScaleY.Sample(),trackScaleZ.Sample());

//         // objects positioned along
//         second_object = GameObject.Instantiate(prefabs_mat1.Sample());
//         float largeObjectScale = Random.Range(0.5f,0.6f);
//         float smallObjectScale = Random.Range(1,1.1f);
//         if (second_object.name=="Arch(Clone)" || second_object.name=="Pipe(Clone)")
//         {
//             second_object.transform.position = new Vector3(-7,0.9f+cheat_height,0);
//             second_object.transform.rotation = Quaternion.Euler(90,0,0);    
//             second_object.transform.localScale = Vector3.one * largeObjectScale;
//         }    
//         else if (second_object.name=="Torus(Clone)")
//         {
//             second_object.transform.position = new Vector3(-7,0.8f+cheat_height,0);
//             second_object.transform.rotation = Quaternion.Euler(0,0,0);    
//             second_object.transform.localScale = Vector3.one * smallObjectScale;
//         }      
//         else
//         {
//             second_object.transform.position = new Vector3(-7,1f+cheat_height,0);
//             second_object.transform.rotation = Quaternion.Euler(0,0,90);    
//             second_object.transform.localScale = Vector3.one * smallObjectScale;
//         }  

//         currentInstance1 = GameObject.Instantiate(second_object);
//         currentInstance3 = GameObject.Instantiate(second_object);
//         currentInstance4 = GameObject.Instantiate(second_object);
//         currentInstance5 = GameObject.Instantiate(second_object);

//         x11=second_object.transform.position.x;
//         y11=second_object.transform.position.y;
//         z11=second_object.transform.position.z;

//         x21=currentInstance1.transform.position.x;
//         y21=currentInstance1.transform.position.y;
//         z21=currentInstance1.transform.position.z;

//         x31=currentInstance3.transform.position.x;
//         y31=currentInstance3.transform.position.y;
//         z31=currentInstance3.transform.position.z;

//         x41=currentInstance4.transform.position.x;
//         y41=currentInstance4.transform.position.y;
//         z41=currentInstance4.transform.position.z;

//         x51=currentInstance5.transform.position.x;
//         y51=currentInstance5.transform.position.y;
//         z51=currentInstance5.transform.position.z;

//         float xShift_ = xShift.Sample();
//         second_object.transform.position = new Vector3(x11+xShift_,y11,z11);
//         currentInstance1.transform.position = new Vector3(x21+2*xShift_,y21,z21);
//         currentInstance3.transform.position = new Vector3(x31+3*xShift_,y31,z31);
//         currentInstance4.transform.position = new Vector3(x41+4*xShift_,y41,z41);
//         currentInstance5.transform.position = new Vector3(x51+5*xShift_,y51,z51);



//         // camera randomisation code
//         // mainCamera.transform.position = new Vector3(6.4f,8.9f,-8.45f);
//         // mainCamera.transform.rotation = Quaternion.Euler(45,-45,0);
//         mainCamera.transform.position = cameraPosition.Sample();
//         mainCamera.transform.rotation = Quaternion.Euler(cameraRot.Sample());

//         // material randomisation code
//         int randomNumber1 = Random.Range(0,8); //initialise at every iteration so new number generated
//         material1.color = colorList[randomNumber1];
//         int randomNumber2 = Random.Range(0,8); //initialise at every iteration so new number generated
//         while (randomNumber2==randomNumber1)
//         {
//             randomNumber2 = Random.Range(0,8);
//         }
//         material2.color = colorList[randomNumber2];
//         // format object colour and object type
//         string object_colour1 = colour_identifier[material1.color.ToString()];
//         string object_type1 = currentInstance1.name;
//         object_type1 = object_type1.Substring(0,object_type1.Length-7); // remove the "(Clone) " which is 7 characters
//         string object_colour_type1 = object_colour1 + " " + object_type1;
//         string object_colour2 = colour_identifier[material2.color.ToString()];
//         string object_type2 = currentInstance2.name;
//         object_type2 = object_type2.Substring(0,object_type2.Length-7); // remove the "(Clone) " which is 7 characters
//         string object_colour_type2 = object_colour2 + " " + object_type2;
//         // write colour and object type to txt file
//         StreamWriter writer1 = new StreamWriter(filePath1, true);
//         writer1.WriteLine(object_colour_type1);
//         writer1.Close();
//         StreamWriter writer2 = new StreamWriter(filePath2, true);
//         writer2.WriteLine(object_colour_type2);
//         writer2.Close();
//     }

//     protected override void OnIterationEnd()
//     {
//         GameObject.Destroy(currentInstance1);
//         GameObject.Destroy(currentInstance2);
//         GameObject.Destroy(currentInstance3);
//         GameObject.Destroy(currentInstance4);
//         GameObject.Destroy(currentInstance5);
//         GameObject.Destroy(second_object);
//     }

// }