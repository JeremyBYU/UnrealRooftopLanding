# Unreal Rooftop Landing


## Install

### Python Instructions

1. Install [conda](https://conda.io/projects/conda/en/latest/) or create a python virtual environment ([Why?](https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c)). I recommend conda for Windows users.
2. Example - `conda create --name unreal python=3.7 && conda activate unreal` 
2. `pip install -e .`


### Other Projects

There are several more projects used throughout this project to **make** the unreal engine environments.


1. Custom Fork of AirSim (Unreal Plugin) with noisy LiDAR model and custom handling of semantic segmentation codes - `https://github.com/JeremyBYU/AirSim`
2. Unreal Plugin to generate *classified* airborne points clouds of a level - `https://github.com/JeremyBYU/PointCloudGeneratorUE4`
3. Python scripts to analyze classified airborne point clouds of city rooftops and generate *vector* maps - `https://github.com/JeremyBYU/create-map`
4. Unreal Python scripts to randomly generate and place assets on city rooftops `https://github.com/JeremyBYU/UnrealLanding_UnrealProject`

Note that items (1,2,4) are installed inside of the Unreal Project itself.
