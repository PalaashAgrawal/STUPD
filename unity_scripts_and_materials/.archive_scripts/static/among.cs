using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/among")]
public class among : Randomizer
{
    // prefabs selection
    public GameObjectParameter prefabs_mat1;
    public GameObjectParameter prefabs_mat2;
    private GameObject currentInstance1;
    private GameObject currentInstance2;
    private int objectCount1;
    private int objectCount2;
    public float spawnRadius1=3;
    public float spawnCollisionCheckRadius1=0.5f;   
    public float spawnRadius2=7;
    public float spawnCollisionCheckRadius2=2;   
    // camera selection 
    public Camera mainCamera;
    // public Vector3Parameter cameraPosition;
    // materials selection
    public Material material1;
    public Material material2;

    protected override void OnIterationStart()
    {   
        objectCount1 = Random.Range(1,5); // "central" objects
        objectCount2 = Random.Range(10,20); 

        Vector3 spawnPoint1 = new Vector3(0,0,0) + Random.insideUnitSphere * spawnRadius1;
        currentInstance1 = GameObject.Instantiate(prefabs_mat1.Sample(),spawnPoint1,Random.rotation);
        Vector3 spawnPoint2 = new Vector3(0,0,0) + Random.onUnitSphere * spawnRadius2;
        currentInstance2 = GameObject.Instantiate(prefabs_mat2.Sample(),spawnPoint2,Random.rotation);

        for (int loop=0;loop<objectCount1;loop++)
        {
            Vector3 spawnPoint3 = new Vector3(0,0,0) + Random.insideUnitSphere * spawnRadius1;
            if (!Physics.CheckSphere(spawnPoint3,spawnCollisionCheckRadius1))
            {
                GameObject.Instantiate(currentInstance1,spawnPoint3,Random.rotation);
            }
        }

        for (int loop=0;loop<objectCount2;loop++)
        {
            Vector3 spawnPoint4 = new Vector3(0,0,0) + Random.insideUnitSphere * spawnRadius2;
            if (!Physics.CheckSphere(spawnPoint4,spawnCollisionCheckRadius2))
            {
                GameObject.Instantiate(currentInstance2,spawnPoint4,Random.rotation);
            }
        }

        // // camera randomisation code
        // mainCamera.transform.position = cameraPosition.Sample();
        // mainCamera.transform.LookAt(new Vector3(0,0,0)); // can set a manual point to look at
        // camera randomisation code
        mainCamera.transform.position = Random.onUnitSphere * 15;
        // formula for sphere x2+y2+z2=r2, r=10: x2+y2+z2=100
        // set a limit, e.g. limit=7.5
        // while (Mathf.Abs(mainCamera.transform.position.x)>sphere_limit | Mathf.Abs(mainCamera.transform.position.y)>sphere_limit | Mathf.Abs(mainCamera.transform.position.z)>sphere_limit)
        // {
        //     mainCamera.transform.position = Random.onUnitSphere * 10;
        // }
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
// [AddRandomizerMenu("Perception/among")]
// public class among : Randomizer
// {
//     // prefabs selection
//     public GameObjectParameter prefabs_mat1;
//     public GameObjectParameter prefabs_mat2;
//     private GameObject currentInstance1;
//     private GameObject currentInstance2;
//     private int objectCount1;
//     private int objectCount2;
//     public float spawnRadius1=3;
//     public float spawnCollisionCheckRadius1=0.5f;   
//     public float spawnRadius2=7;
//     public float spawnCollisionCheckRadius2=2;   
//     // camera selection 
//     public Camera mainCamera;
//     public Vector3Parameter cameraPosition;
//     // materials selection
//     public Material material1;
//     public Material material2;

//     protected override void OnIterationStart()
//     {   
//         objectCount1 = Random.Range(1,4); // "central" objects
//         objectCount2 = Random.Range(6,10); 

//         currentInstance1 = GameObject.Instantiate(prefabs_mat1.Sample());
//         Vector3 spawnPoint1 = new Vector3(0,0,0) + Random.insideUnitSphere * spawnRadius1;
//         currentInstance1.transform.position = spawnPoint1;

//         currentInstance2 = GameObject.Instantiate(prefabs_mat2.Sample());
//         Vector3 spawnPoint2 = new Vector3(0,0,0) + Random.insideUnitSphere * spawnRadius2;
//         currentInstance2.transform.position = spawnPoint2;

//         for (int loop=0;loop<objectCount1;loop++)
//         {
//             Vector3 spawnPoint3 = currentInstance1.transform.position + Random.insideUnitSphere * spawnRadius1;
//             if (!Physics.CheckSphere(spawnPoint3,spawnCollisionCheckRadius1))
//             {
//                 GameObject.Instantiate(currentInstance1,spawnPoint3,Random.rotation);
//             }
//         }

//         for (int loop=0;loop<objectCount2;loop++)
//         {
//             Vector3 spawnPoint4 = currentInstance2.transform.position + Random.insideUnitSphere * spawnRadius2;
//             if (!Physics.CheckSphere(spawnPoint4,spawnCollisionCheckRadius2))
//             {
//                 GameObject.Instantiate(currentInstance2,spawnPoint4,Random.rotation);
//             }
//         }

//         // camera randomisation code
//         mainCamera.transform.position = cameraPosition.Sample();
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