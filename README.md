# Unreal Rooftop Landing

This is the master repository to hold source code and analysis of Rooftop Landing in the Unreal Engine. This repository discusses the use of fusing Polylidar3D and Deep Learning to identify flat surfaces from 3D Point Clouds.

## Install

1. Install [conda](https://conda.io/projects/conda/en/latest/) or create a python virtual environment ([Why?](https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c)). I recommend conda for Windows users.
2. Example - `conda create --name unreal python=3.7 && conda activate unreal` 
2. `pip install -e .`


## Unreal Environment

There are several projects/plugins used to **make** the actual unreal engine world environments. 

1. Custom Fork of AirSim (Unreal Plugin) with noisy LiDAR model and custom handling of semantic segmentation codes - `https://github.com/JeremyBYU/AirSim`
2. Unreal Plugin to generate *classified* airborne points clouds of a level (map/world) - `https://github.com/JeremyBYU/PointCloudGeneratorUE4`
3. Python scripts to analyze classified airborne point clouds of city rooftops and generate *vector* maps - `https://github.com/JeremyBYU/create-map`
4. Unreal Python scripts to randomly generate and place assets on city rooftops (using vector maps) `https://github.com/JeremyBYU/UnrealLanding_UnrealProject`

Note that items (1,2,4) are installed/used inside of the Unreal Project/Editor itself. The Unreal Project Files can be found [here](https://github.com/JeremyBYU/UnrealLanding_UnrealProject).


## Workflow

Below is the workflow to generate training and testing data from the unreal environment.


### Generate Collection Points

First we need to generate "Collection Points" for where the drone will be. To read up on collection points please see `airsimcollect/README.md`.


#### Computer Vision Training Collection Points

Run `poi generate` for polygons and line data to create the Computer Vision Training Set

1. `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_cv_train.npy -ns 2`
2. `poi generate -m assets/maps/poi-line.geojson -o assets/collectionpoints/collection_points_cv_train.npy -nf 100 -yd 90 -pr 45 45 -pd 0 -ri 100 -ao`

Note that (2) doesn't overwrite the file, but will append to it (`-ao` flag).

We are also adding these new ones as well. These are samped as a circle overhead of the roftoop, with the camera pointing down. 10 meters and 15 meters above:

1. `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_cv_train.npy -ho 1000 -ri -500 -sc circle -yd 45`
2. `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_cv_train.npy -ho 1500 -ri -500 -sc circle -yd 45 -ao`

#### Computer Vision Testing Collection Points

~2000 images for RandomWorldSeed3 and RandomManhattanDistribution each. Pitch range is offset by 5. Basically not only is the data different but it captured at different angles.  Probably overkill but still.

1. `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_cv_test.npy -ri 500 -pr 35 80 -yd 45`


#### Rooftop Landing Collection Points

This generated collection points for Rooftop LIDAR collection. Much smaller collection. Used for testing actual landing site selection. 

1. `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_lidar_landing.npy -ho 1000 -ri 1000 -pr 75 75 -pd 0 -yd 90 -rfn class_label`

Proposed- `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_lidar_landing.npy -ho 1000 -ri -500 -sc circle -yd 90 -rfn class_label`

### Generate Images from AirSim

Next we will launch AirSim in Computer Vision mode and collect scene (rgb) and segmentation images. Data is saved inside the folder `AirSimCollectData`. We use a different program called `asc` (**A**ir**S**im**C**ollect) to perform this task. See `airsimcollect/README.md` for more details.  The drone will go to the position and orientation we specified from the previously generated collection points and capture images. Be sure to update your AirSim settings in your home directory to use the computer vision mode.

1. `asc collect -c assets/config/collect_cv_train.json`
2. `asc collect -c assets/config/collect_cv_test.json`

### Generate Images and LiDAR data from AirSim

Next we will launch AirSim in `Multirotor` mode. This time we will generate images and LiDAR point clouds from the collection points mentioned previously. Data is saved inside the folder `AirSimCollectData`. Be sure to update your AirSim settings in your home directory to use the multirotor.

1. `asc collect -c assets/config/collect_lidar_landing.json`

Notes - Sometimes the camera takes time to update position, add more time delay than 0.5 seconds. In other words the lidar and vehicle move to a new position but the camera is still in the old position (AirSim bug).

LiDAR Sensor Settings:

```json
        "0": {
          "SensorType": 6,
          "Enabled": true,
          "NumberOfChannels": 16,
          "RotationsPerSecond": 10,
          "PointsPerSecond": 100000,
          "X": 0.46,
          "Y": 0,
          "Z": 0,
          "Roll": 0,
          "Pitch": 0,
          "Yaw": 0,
          "VerticalFOVUpper": -5,
          "VerticalFOVLower": -45,
          "HorizontalFOVStart": -30,
          "HorizontalFOVEnd": 30,
          "DrawDebugPoints": false,
          "DataFrame": "VehicleInertialFrame"
        }
```

## Gather Data Script

`python assets/rooftop/scripts/gatherstats.py -c assets/rooftop/scripts/config_gather_stats.json`

`python assets/rooftop/scripts/gatherstats.py -c assets/rooftop/scripts/config_gather_stats_tx2.json`

* UID 72 is a great example of a failure
  * Aircraft is too far away, point cloud does not pick up on air vents
 



## Scratch

### TODO

* Collect more data (RGB, SEG) from 2 more worlds, 2000 Images each
  * 1 world randomly generated according to the distribution of Manhattan
  * 1 world manually created according to the distribution of Manhattan
  * Send that data to Brian to evaluate model performance for segmentation
* Collect more data of building rooftops. Use the manually created world
  * Each building will capture - Scene, Segmentation, Lidar, records.json. 3 New folders will be created LidarClassified, Polygons, SegmentationPredicted, LabeledScene
    * 4 pictures of each building, 10 meters and 10 meters up, each 4 sides of the roof
    * records.json will need to be modified to identify building name. Make a simple script that integrates poi-roof-lidar.geoson with records.json

  * For each snapshot perform the following and record timing and metrics
    * Every picture will need to sent to DeepLearningModel to provide a prediction. That prediction (picture) will be saved in SegmentationPredicted
      * Record time to predict, and calculate metrics (IOU?)
    * Lidar Point cloud should be projected into ground truth segmented image. True classified point cloud should be saved in LidarClassified
      * Record time to project point cloud
    * Lidar Point cloud should be projected into predicted segmented image. Predicted classified point cloud should be saved in LidarClassified
      * Record time to project point cloud
    * Use polylidar to extract roof polygon for both point clouds. Save polygon into Polygons folder
      * Record time to generate polygon
    * Use polylabel to find greatest inscribed circle. 
      * Record time to find polylabel, Record position 
    * Project polylabel plane and landing zone into scene picture
      * Save picture into labeled scene

  * Do a few 360 point clouds photos?


### Examples
  * predict_segmentation, laptop, N/A,  0, Building1, 20ms, IOU, N/A
  * classify_point_cloud, laptop, groundtruth, 0, Building1, 20ms, N/A, N/A
  * classify_point_cloud, laptop, predicted, 0, Building1, 20ms, N/A, N/A
  * classify_point_cloud, jetson, predicted, 0, Building1, 20ms, N/A, N/A
  * polylidar, laptop, predicted, 0, Building1, 20ms, N/A, N/A
  * polylidar, jetson, predicted, 0, Building1, 20ms, N/A, N/A
  * polylabel, laptop, predicted, 0, Building1, 20ms, UK, [point,radius]
  * polylabel, laptop, groundtruth, 0, Building1, 20ms, UK, [point,radius]


  ```
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 46.592327117919922, 47.800155639648438, 24.875473022460938 ],
			"boundingbox_min" : [ -45.977882385253906, -59.884254455566406, -13.06738160405574 ],
			"field_of_view" : 59.999999999999993,
			"front" : [ 0.013683577996376688, -0.32966681771908396, -0.94399817213180537 ],
			"lookat" : [ 2.5488070936335969, -8.1752193707871861, 6.6814923975724128 ],
			"up" : [ 0.0051760193170947164, 0.94409726625824619, -0.32962639558708806 ],
			"zoom" : 0.30142939629576992
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 8.4127435684204102, 5.8720441335277966, 0.63159967445807108 ],
			"boundingbox_min" : [ -8.211085319519043, -5.8716968618454288, -2.5818006992340088 ],
			"field_of_view" : 59.999999999999993,
			"front" : [ -0.23736820253299898, 0.01738649469363451, -0.97126414853453824 ],
			"lookat" : [ -0.4090289712471753, -1.0071293015254705, -0.86852741102983266 ],
			"up" : [ 0.97141962599464171, 0.003737243918986703, -0.23733929982268123 ],
			"zoom" : 0.57129903937538118
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
  ```