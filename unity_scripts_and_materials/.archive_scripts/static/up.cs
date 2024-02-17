using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/up")]
public class up : Randomizer
{
    // objects variables initialisation
    public GameObjectParameter prefabs_mat1;
    private GameObject currentInstance1;
    private float timeCounter = 0f;
    private float speed;
    // camera variables initialisation
    public Camera mainCamera;
    public float sphere_limit=7.5f;
    public Material material1;

    protected override void OnIterationStart()
    {   
        timeCounter = 0f;
        speed = Random.Range(3,4);
        Debug.Log(speed);
        // objects randomisation code
        float randomX = Random.Range(-2,2);
        float randomZ = Random.Range(-2,2);
        Vector3 startPos = new Vector3(randomX,-2,randomZ);
        currentInstance1 = GameObject.Instantiate(prefabs_mat1.Sample(),startPos,Random.rotation);
        // camera randomisation code
        mainCamera.transform.position = Random.onUnitSphere * 10;
        // formula for sphere x2+y2+z2=r2, r=10: x2+y2+z2=100
        // set a limit, e.g. limit=7.5
        // while (Mathf.Abs(mainCamera.transform.position.x)>sphere_limit | Mathf.Abs(mainCamera.transform.position.y)>sphere_limit | Mathf.Abs(mainCamera.transform.position.z)>sphere_limit)
        while (mainCamera.transform.position.y<-sphere_limit | mainCamera.transform.position.y>sphere_limit)
        {
            mainCamera.transform.position = Random.onUnitSphere * 10;
        }
        // mainCamera.transform.LookAt(currentInstance1.transform); //look at gameobject position
        mainCamera.transform.LookAt(Vector3.zero); // can set a manual point to look at
        // mainCamera.transform.up = -mainCamera.transform.forward;
        // material randomisation code
        material1.color = Random.ColorHSV();
    }

    protected override void OnUpdate()
    {
        timeCounter+=Time.deltaTime*speed;
        float x1 = currentInstance1.transform.position.x;
        float y1 = -2 + timeCounter;
        float z1 = currentInstance1.transform.position.z;
        currentInstance1.transform.position = new Vector3(x1,y1,z1);
    }

    protected override void OnIterationEnd()
    {
        GameObject.Destroy(currentInstance1);
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
// [AddRandomizerMenu("Perception/up")]
// public class up : Randomizer
// {
//     // output colour filepaths
//     string filePath1 = "C:/Users/haidi/OneDrive/Desktop/dataset_v3/colors_txt_files/colors_up1.txt";
//     string filePath2 = "C:/Users/haidi/OneDrive/Desktop/dataset_v3/colors_txt_files/colors_up2.txt";

//     // objects variables initialisation
//     public FloatParameter largeObjectScale; // arch,pipe,
//     public FloatParameter smallObjectScale; // capsule,cone,cube,cylinder,prism,sphere,torus
//     public Vector3Parameter objectRotation;
//     public GameObjectParameter prefabs_mat1;
//     public GameObjectParameter prefabs_mat2;
//     private GameObject currentInstance1;
//     private GameObject currentInstance2;
//     private float timeCounter = 0f;
//     private float speed=3f;

//     // camera variables initialisation
//     public Camera mainCamera;
//     public FloatParameter cameraDistance;

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
//         timeCounter = 0f;
//         // objects randomisation code
//         currentInstance1 = GameObject.Instantiate(prefabs_mat1.Sample());
//         if (currentInstance1.name=="Arch(Clone)" || currentInstance1.name=="Pipe(Clone)")
//         {
//             currentInstance1.transform.position = new Vector3(0,0,5);
//             currentInstance1.transform.rotation = Quaternion.Euler(objectRotation.Sample());
//             currentInstance1.transform.localScale = Vector3.one * largeObjectScale.Sample();        
//         }            
//         else
//         {
//             currentInstance1.transform.position = new Vector3(0,0,5);
//             currentInstance1.transform.rotation = Quaternion.Euler(objectRotation.Sample());
//             currentInstance1.transform.localScale = Vector3.one * smallObjectScale.Sample();
//         }

//         currentInstance2 = GameObject.Instantiate(prefabs_mat2.Sample());
//         currentInstance2.transform.position = new Vector3(0,100,0);

//         // camera randomisation code
//         var distance1 = cameraDistance.Sample();
//         var distance2 = cameraDistance.Sample();
//         mainCamera.transform.position = new Vector3(distance1,10,distance2);
//         mainCamera.transform.rotation = Quaternion.Euler(90,0,180);

//         // material randomisation code
//         int randomNumber1 = Random.Range(0,8); //initialise at every iteration so new number generated
//         material1.color = colorList[randomNumber1];
//         int randomNumber2 = Random.Range(0,8); //initialise at every iteration so new number generated
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

//     protected override void OnUpdate()
//     {
//         timeCounter+=Time.deltaTime*speed;
//         float x1 = 0;
//         float y1 = 0;
//         float z1 = 5 - timeCounter;
//         currentInstance1.transform.position = new Vector3(x1,y1,z1);
//     }

//     protected override void OnIterationEnd()
//     {
//         GameObject.Destroy(currentInstance1);
//         GameObject.Destroy(currentInstance2);
//     }
// }