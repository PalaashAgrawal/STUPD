using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/outside")]
public class outside : Randomizer
{
    // prefabs selection
    public GameObjectParameter prefabs_mat1;
    public GameObjectParameter prefabs_mat2;
    private GameObject currentInstance1;
    private GameObject currentInstance2;
    public FloatParameter centerScale;    
    public FloatParameter containerScale;    
    // camera selection 
    public Camera mainCamera;
    public float camera_radius=10f;
    public float sphere_limit=7.5f;
    // materials selection
    public Material material1;
    public Material material2;

    protected override void OnIterationStart()
    {   
        // object variables code
        currentInstance1=GameObject.Instantiate(prefabs_mat1.Sample());
        currentInstance2=GameObject.Instantiate(prefabs_mat2.Sample());

        currentInstance2.transform.position = Random.insideUnitCircle * 1;
        currentInstance2.transform.localScale = Vector3.one * containerScale.Sample();

        Vector2 randomPoint = Random.insideUnitCircle.normalized;
        // Vector2 normalisedRandomPoint = randomPoint.Normalize();
        float random_float = Random.Range(3,6);
        Vector2 spawnPoint2D = randomPoint * random_float;
        Vector3 spawnPoint3D = new Vector3(spawnPoint2D.x,0,spawnPoint2D.y);
        currentInstance1.transform.position = spawnPoint3D;
        currentInstance1.transform.localScale = Vector3.one * centerScale.Sample();
        

        // camera randomisation code
        mainCamera.transform.position = Random.onUnitSphere * camera_radius;
        // formula for sphere x2+y2+z2=r2, r=10: x2+y2+z2=100
        // set a limit, e.g. limit=7.5
        // while (Mathf.Abs(mainCamera.transform.position.x)>sphere_limit | Mathf.Abs(mainCamera.transform.position.y)>sphere_limit | Mathf.Abs(mainCamera.transform.position.z)>sphere_limit)
        while (Mathf.Abs(mainCamera.transform.position.y)<sphere_limit)
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
// [AddRandomizerMenu("Perception/outside")]
// public class outside : Randomizer
// {
//     // output colour filepaths
//     string filePath1 = "C:/Users/haidi/OneDrive/Desktop/dataset_v3/colors_txt_files/colors_outside1.txt";
//     string filePath2 = "C:/Users/haidi/OneDrive/Desktop/dataset_v3/colors_txt_files/colors_outside2.txt";

//     // objects variables initialisation
//     public GameObjectParameter prefabs_mat1;
//     public GameObjectParameter prefabs_mat2;
//     private GameObject currentInstance1;
//     private GameObject currentInstance2;

//     // camera variables initialisation
//     public Camera mainCamera;

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
//         float rotationX = Random.Range(-15f,15f);
//         float rotationY = Random.Range(-15f,15f);
//         float rotationZ = Random.Range(-15f,15f);
//         float ObjectScale1 = Random.Range(0.5f,0.75f); // currentInstance1, smaller scale, 1-1.5

//         currentInstance1=GameObject.Instantiate(prefabs_mat1.Sample());
//         currentInstance1.transform.position = new Vector3(-4.5f,0,0);
//         currentInstance1.transform.rotation = Quaternion.Euler(rotationX,rotationY,rotationZ);
//         currentInstance1.transform.localScale = Vector3.one * ObjectScale1;

//         currentInstance2=GameObject.Instantiate(prefabs_mat2.Sample());
//         currentInstance2.transform.rotation = Quaternion.Euler(0,0,0);
//         if (currentInstance2.name=="Octagonal_pipe(Clone)")
//         {
//             currentInstance2.transform.position = new Vector3(0,0,0);
//             float ObjectScale2 = Random.Range(1.35f,1.55f); // currentInstance2, larger scale, 3-4
//             currentInstance2.transform.localScale = Vector3.one * ObjectScale2;
//         }
//         if (currentInstance2.name=="Circular_container(Clone)")
//         {
//             currentInstance2.transform.position = new Vector3(0,0,0);
//             float ObjectScale2 = Random.Range(1,1.25f); // currentInstance2, larger scale, 3-4
//             currentInstance2.transform.localScale = Vector3.one * ObjectScale2;
//         }
//         if (currentInstance2.name=="Torus_container(Clone)")
//         {
//             currentInstance2.transform.position = new Vector3(0,0,0);
//             float ObjectScale2 = Random.Range(2.25f,2.5f); // currentInstance2, larger scale, 3-4
//             currentInstance2.transform.localScale = Vector3.one * ObjectScale2;
//         }
//         if (currentInstance2.name=="Square_container(Clone)")
//         {
//             currentInstance2.transform.position = new Vector3(0,0,0);
//             float ObjectScale2 = Random.Range(2.5f,3.5f); // currentInstance2, larger scale, 3-4
//             currentInstance2.transform.localScale = Vector3.one * ObjectScale2;
//         }

//         // camera randomisation code
//         float cameraRotationY = Random.Range(0,360);
//         float cameraPosX = Random.Range(-2,2);
//         float cameraPosZ = Random.Range(-2,2);
//         mainCamera.transform.position = new Vector3(cameraPosX,10,cameraPosZ);
//         mainCamera.transform.rotation = Quaternion.Euler(90,cameraRotationY,0);

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
//     }
// }