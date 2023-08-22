using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/towards")]
public class towards : Randomizer
{
    // objects variables initialisation
    public GameObjectParameter prefabs_mat1; // object outside
    public GameObjectParameter prefabs_mat2; // the hollow central object
    private GameObject currentInstance1;  // object outside
    private GameObject currentInstance2;  // object outside
    private float timeCounter=0;
    private float speed;
    // camera variables initialisation
    public Camera mainCamera;
    public float sphere_limit=4f;
    // material variables initialisation
    public Material material1;
    public Material material2;

    protected override void OnIterationStart()
    {   
        timeCounter=0;
        speed = Random.Range(2,3);
        // object variables code
        float rotationY = Random.Range(-15f,15f);
        float rotationZ = Random.Range(-15f,15f);
    
        float ObjectScale1 = Random.Range(0.75f,1f); // currentInstance1, centre object 
        currentInstance1=GameObject.Instantiate(prefabs_mat1.Sample());
        currentInstance1.transform.position = new Vector3(-5,0,0);
        currentInstance1.transform.rotation = Quaternion.Euler(90,rotationY,rotationZ);
        currentInstance1.transform.localScale = Vector3.one * ObjectScale1;
        if (currentInstance1.name=="Pipe(Clone)" || currentInstance1.name=="Torus(Clone)" )
        {
            currentInstance1.transform.rotation = Quaternion.Euler(0,rotationY,rotationZ);
        }

        float ObjectScale2 = Random.Range(0.75f,1f); // currentInstance2, object moving over
        currentInstance2=GameObject.Instantiate(prefabs_mat2.Sample());
        currentInstance2.transform.position = new Vector3(5f,0,0);
        currentInstance2.transform.rotation = Quaternion.Euler(90,rotationY,rotationZ);
        currentInstance2.transform.localScale = Vector3.one * ObjectScale2;
        if (currentInstance2.name=="Pipe(Clone)" || currentInstance2.name=="Torus(Clone)" )
        {
            currentInstance2.transform.rotation = Quaternion.Euler(0,rotationY,rotationZ);
        }


        // camera randomisation code
        mainCamera.transform.position = Random.onUnitSphere * 10;
        while (mainCamera.transform.position.y>-sphere_limit & mainCamera.transform.position.y<sphere_limit)
        {
            mainCamera.transform.position = Random.onUnitSphere * 10;
        }
        mainCamera.transform.LookAt(Vector3.zero);
        // material randomisation code
        material1.color = Random.ColorHSV();
        material2.color = Random.ColorHSV();
    }

    protected override void OnUpdate()
    {
        timeCounter+=Time.deltaTime*speed;
        float x = -5+timeCounter;
        float y = 0;
        float z = 0;
        currentInstance1.transform.position = new Vector3(x,y,z);
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
// [AddRandomizerMenu("Perception/towards")]
// public class towards : Randomizer
// {
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
//         mainCamera.transform.rotation = Quaternion.Euler(90,0,0);

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

