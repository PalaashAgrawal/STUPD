using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/along")]
public class along : Randomizer
{
    public GameObjectParameter prefabs_mat1;
    public GameObjectParameter prefabs_mat2;
    private GameObject currentInstance1;
    private GameObject currentInstance2;
    public Vector3Parameter trackScale;
    private float timeCounter = 0f;
    private float speed;
    // camera variables initialisation
    public Camera mainCamera;
    public float sphere_limit=3f;
    // material variables initialisation
    public Material material1;
    public Material material2;
    
    protected override void OnIterationStart()
    {   
        timeCounter = 0f;
        speed = Random.Range(2,3);
        // object variables code
        float randomHeight = Random.Range(1.5f,2.5f);
        currentInstance1 = GameObject.Instantiate(prefabs_mat1.Sample(),new Vector3(-3.5f,randomHeight,0),Random.rotation);
        currentInstance2 = GameObject.Instantiate(prefabs_mat2.Sample());
        currentInstance2.transform.position = Vector3.zero;
        currentInstance2.transform.localScale = trackScale.Sample(); 
        // camera randomisation code
        mainCamera.transform.position = Random.onUnitSphere * 10;
        while (mainCamera.transform.position.y>sphere_limit | mainCamera.transform.position.y<-sphere_limit)
        {
            mainCamera.transform.position = Random.onUnitSphere * 10;
        }
        float randomX = Random.Range(-2,2);
        float randomY = Random.Range(-2,2);
        float randomZ = Random.Range(-2,2);
        Vector3 cameraFocus = new Vector3(randomX,randomY,randomZ);
        mainCamera.transform.LookAt(cameraFocus);
        // material randomisation code
        material1.color = Random.ColorHSV();
        material2.color = Random.ColorHSV();
    }

    protected override void OnUpdate()
    {
        timeCounter+=Time.deltaTime*speed;
        float x1 = -3.5f + timeCounter;
        float y1 = currentInstance1.transform.position.y;
        float z1 = currentInstance1.transform.position.z;
        currentInstance1.transform.position = new Vector3(x1,y1,z1);
    }

    protected override void OnIterationEnd()
    {
        GameObject.Destroy(currentInstance1);
        GameObject.Destroy(currentInstance2);
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
// [AddRandomizerMenu("Perception/along")]
// public class along : Randomizer
// {
    
//     // output colour filepaths
//     string filePath1 = "/Users/haidiazaman/Desktop/dataset_v3/colors_txt_files/colors_along1.txt";
//     string filePath2 = "/Users/haidiazaman/Desktop/dataset_v3/colors_txt_files/colors_along2.txt";

//     // object variables initialisation
//     public GameObjectParameter prefabs_mat1;
//     public GameObjectParameter prefabs_mat2;
//     private GameObject currentInstance1;
//     private GameObject currentInstance2;
//     public FloatParameter trackScaleX;
//     public FloatParameter trackScaleY;
//     public FloatParameter trackScaleZ;
//     private float timeCounter = 0;
//     private float speed = 2.5f;
//     private float cheat_height = 0.1f;

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
//         timeCounter = 0;
//         currentInstance1 = GameObject.Instantiate(prefabs_mat1.Sample());
//         currentInstance2 = GameObject.Instantiate(prefabs_mat2.Sample()); // track

//         //track
//         currentInstance2.transform.position = new Vector3(0,0.25f,0);
//         currentInstance2.transform.localScale = new Vector3(trackScaleX.Sample(),trackScaleY.Sample(),trackScaleZ.Sample());

//         //moving object
//         float largeObjectScale = Random.Range(0.75f,1);
//         float smallObjectScale = Random.Range(1,1.25f);
//         if (currentInstance1.name=="Arch(Clone)" || currentInstance1.name=="Pipe(Clone)")
//         {
//             currentInstance1.transform.position = new Vector3(-5,1.75f+cheat_height,0);
//             currentInstance1.transform.rotation = Quaternion.Euler(90,0,0);    
//             currentInstance1.transform.localScale = Vector3.one * largeObjectScale;
//         }    
//         else if (currentInstance1.name=="Torus(Clone)")
//         {
//             currentInstance1.transform.position = new Vector3(-5,0.8f+cheat_height,0);
//             currentInstance1.transform.rotation = Quaternion.Euler(0,0,0);    
//             currentInstance1.transform.localScale = Vector3.one * smallObjectScale;
//         }      
//         else
//         {
//             currentInstance1.transform.position = new Vector3(-5,1f+cheat_height,0);
//             currentInstance1.transform.rotation = Quaternion.Euler(0,0,90);    
//             currentInstance1.transform.localScale = Vector3.one * smallObjectScale;
//         }  



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

//     protected override void OnUpdate()
//     {
//         timeCounter+=Time.deltaTime*speed;
//         float x = -5+timeCounter;
//         float y = currentInstance1.transform.position.y;
//         float z = 0 ;
//         Debug.Log(x);
//         currentInstance1.transform.position = new Vector3(x,y,z);
//     }

//     protected override void OnIterationEnd()
//     {
//         GameObject.Destroy(currentInstance1);
//         GameObject.Destroy(currentInstance2);
//     }

// }