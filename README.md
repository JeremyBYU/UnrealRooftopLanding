# Unreal Rooftop Landing

This is the master repository to hold source code and analysis of Rooftop Landing in the Unreal Engine. This repository discusses the use of fusing Polylidar3D and Deep Learning to identify flat surfaces from 3D Point Clouds (called **Semantic Polylidar3D**). The Polylidar3D repository has already been updated to include semantic information and can be found [here](https://github.com/JeremyBYU/polylidar).

## Install

1. Install [conda](https://conda.io/projects/conda/en/latest/) or create a python virtual environment ([Why?](https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c)). I recommend conda for Windows users.
2. Example - `conda create --name unreal python=3.6 && conda activate unreal` 
2. `pip install -e .`


## Unreal Environment

There are several projects/plugins used to **make** the actual unreal engine world environments. 

1. Custom Fork of AirSim (Unreal Plugin) with noisy LiDAR model and custom handling of semantic segmentation codes - `https://github.com/JeremyBYU/AirSim`
2. Unreal Plugin to generate *classified* airborne points clouds of a level (map/world) - `https://github.com/JeremyBYU/PointCloudGeneratorUE4`
3. Python scripts to analyze classified airborne point clouds of city rooftops and generate *vector* maps - `https://github.com/JeremyBYU/create-map`
4. Unreal Python scripts to randomly generate and place assets on city rooftops (using vector maps) `https://github.com/JeremyBYU/UnrealLanding_UnrealProject`

Note that items (1,2,4) are installed/used inside of the Unreal Project/Editor itself. The Unreal Project Files can be found [here](https://github.com/JeremyBYU/UnrealLanding_UnrealProject).


## Workflow

Below is the workflow to generate training and testing data from the unreal environment. Copies of the AirSim Settings can be found in `assets/airsim`.

### Generate Collection Points

First we need to generate "Collection Points" for where the drone will be. To read up on collection points please see `airsimcollect/README.md`. These are the points (xyz,roll,pitch,yaw) for which data is collected in AirSim.


#### Computer Vision Training Collection Points

Run `poi generate` for polygons and line data to create the Computer Vision Training Set

1. `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_cv_train.npy -ns 2`
2. `poi generate -m assets/maps/poi-line.geojson -o assets/collectionpoints/collection_points_cv_train.npy -nf 100 -yd 90 -pr 45 45 -pd 0 -ri 100 -ao`

Note that (2) doesn't overwrite the file, but will append to it (`-ao` flag).

We are also adding these new ones as well. These are sampled as a circle overhead of the roftoop, with the camera pointing down. 10 meters and 15 meters above:

1. `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_cv_train.npy -ho 1000 -ri -500 -sc circle -yd 45`
2. `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_cv_train.npy -ho 1500 -ri -500 -sc circle -yd 45 -ao`

#### Computer Vision Testing Collection Points

~2000 images for RandomWorldSeed3 and RandomManhattanDistribution each. Pitch range is offset by 5. Basically not only is the data different but it captured at different angles.  Probably overkill but still.

1. `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_cv_test.npy -ri 500 -pr 35 80 -yd 45`


#### Rooftop Landing Collection Points

This generated collection points for Rooftop LIDAR collection. Much smaller collection. Used for testing actual landing site selection. 

<!-- 1. `poi generate -m assets/maps/poi-roof-lidar-modified.geojson -o assets/collectionpoints/collection_points_lidar_landing.npy -ho 1000 -ri 1000 -pr 75 75 -pd 0 -yd 90 -rfn class_label` -->

1. `poi generate -m assets/maps/poi-roof-lidar-modified.geojson -o assets/collectionpoints/collection_points_lidar_landing.npy -ho 1000 -ri -500 -sc circle -yd 90 -pr 0 0 -rfn class_label`

### Generate Images from AirSim

Next we will launch AirSim in Computer Vision mode and collect scene (rgb) and segmentation images. Data is saved inside the folder `AirSimCollectData`. We use a different program called `asc` (**A**ir**S**im**C**ollect) to perform this task. See `airsimcollect/README.md` for more details.  The drone will go to the position and orientation we specified from the previously generated collection points and capture images. Be sure to update your AirSim settings in your home directory to use the computer vision mode.

1. `asc collect -c assets/config/collect_cv_train.json`
2. `asc collect -c assets/config/collect_cv_test.json`


Note that you have to do this for each random world that has been generated.  The workflow is to launch the random world in UE4, then run the collection. After collection is finished shutdown the UE4 world. Now modify the JSON file to "save_dir" to match the name of the next random world.

### Generate Images and LiDAR data from AirSim for Testing

Next we will launch AirSim in `Multirotor` mode. This time we will generate images and LiDAR point clouds from the collection points mentioned previously. Data is saved inside the folder `AirSimCollectData`. Be sure to update your AirSim settings in your home directory to use the multirotor.

1. `asc collect -c assets/config/collect_lidar_landing.json`

<!-- Notes - Sometimes the camera takes time to update position, add more time delay than 0.5 seconds. In other words the lidar and vehicle move to a new position but the camera is still in the old position (AirSim bug). -->


## Making Sure You are Ready

Several scripts have been added into the `scripts/test` folder. These scripts serve as small isolated components to make sure that everything is setup and working for the data gathering and analysis. A test environments (no rooftops, allowed to be shared) were created for this: `LiDARTest`. This level can be downloaded [here](https://drive.google.com/file/d/1UdfcBkOJIA2WSiWwvUXy9Zx65pt3XhJV/view?usp=sharing).

1. `check_segmentation` - Launch the UE4 world and then launch this script. You should see it change all the segmentation codes.
2. `check_lidar` -  Launch the UE4 world and then launch this script. This ensures that we can receive LiDAR from AirSim.
3. `check_transforms` - Launch the UE4 world and then launch this script. This ensures that we can project LIDAR into the image frame correctly.
4. `check_lidar_smoothing` - Launch the UE4 world and then launch this script. This ensures that the LIDAR/mesh smoothing procedures are working.
5. `check_polygon` - Launch the UE4 world and then launch this script. This ensures that Polygon extraction is working.
6. `check_polygon_segmentation` - (SKIP) Launch the UE4 world and then launch this script. This ensures that Semantic Polylidar3D extraction is working. This only works in an UE4 environment that can not be distributed so you will need to skip this one.

## Calculate Semantic Polylidar3D Accuracy, Execution Time, and Visualize 

This is a great little script that will load and visualize the data that was already collected. You just need to point it to the saved data folder. It launches an Open3D window that visualizes pont clouds, polygons, and even buildings.  It also launches an image viewer that shows the camera image, ground truth segmentation, predicted segmentation, and projected polygons.

1. `python -m scripts.check_saved_lidar --gui --seg --data AirSimCollectData/LidarDecisionPoint --map assets/maps/roof-lidar-decision-point.geojson`


Data will be saved in the folder `results` and a notebook does further analysis: `notebooks/AlgorithmAnalysis.ipynb`

![Open3D](/assets/imgs/o3d_example.PNG "Open3D")
![Open3D](/assets/imgs/opencv_example.PNG "Open3D")


## Decision Point Analysis

This will gather data to perform a decision point analysis. An environment must be loaded where a human is on the rooftop. The script will then gather data over the human at a variety of height levels. Environment can not be distributed, but source code is here.

1. `python -m collect_decision_pont_data`

After gathering the data an analysis can be performed by calling 

1. `python -m check_decision_point`

Data will be saved and a notebook does further analysis: `notebooks/DecisionPointAnalysis.ipynb`


## Manhattan Rooftop Data

All Manhattan rooftop data can be found here: `assets/data/manhattan`. An analysis of all this data can be found in this notebook: `notebooks/RooftopAnalysis-3.ipynb`.

## Extra Notes

* The unreal engine has its own coordinate frame and origin. I call this the Unreal Coordinate Frame (UCF). Z is "up" in the frame.
* AirSim uses its own NED coordinate frame. The origin is the starting position of the drone before takeoff (specify this in your collection JSON file). The Z axis is flipped in UCF, everything else is the same wth UCF.
* The position of the UAV body frame and the camera are the same. However the camera is rotated 90 degrees down.
* The position of the LiDAR is 10 cm further on the x-axis in the body frame of the drone.
* AirSimCollect will collect all points from the lidar, camera, and ground truth segmentation. The position/orientation of these sensors are recorded as well (NED) frame.

 

<!-- ## Scratch

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

{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 99.63488281250001, 118.96718750000001, 10.57262817382817 ],
			"boundingbox_min" : [ -172.06511718750002, -157.73282089603492, -26.46878417968756 ],
			"field_of_view" : 59.999999999999993,
			"front" : [ -0.75057721738910432, -0.023062933894965568, -0.6603801494719429 ],
			"lookat" : [ 14.595872846040965, 4.5219864368272846, 9.3399041178781541 ],
			"up" : [ 0.66077992921403872, -0.023277484157787354, -0.75021866404347348 ],
			"zoom" : 0.16742981091231904
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

  ```


 -->
