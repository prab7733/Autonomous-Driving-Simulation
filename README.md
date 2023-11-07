# Auto Pilot

The implementation provides a simple way for collecting, training, and testing a DNN model to drive a car in a virtual (urban) environment [2], using conditional imitation learning [4]. For simplicity traffic and traffic lights/signs are ignored. A single RGB camera is used to take the corresponding actions on steer and throttle.

<p align="center"> 
  <img src="./info/1.gif" alt="" width="250"></a>
  <img src="./info/2.gif" alt="" width="250"></a>
  <img src="./info/3.gif" alt="" width="250"></a>
</p>


### Getting started

First clone the repository
>git clone (https://github.com/prab7733/Autonomous-Driving-Simulation.git)

Install the dependencies for python:
>numpy, cv2, matplotlib, tensorflow, glob, PIL, jupyter lab

Then the CARLA python package is needed, but before CARLA[2] needed to be installed. Follow the instructions from the CARLA page to install the simulator. Search for the `*.egg` file in the `PythonAPI ` folder, copy/ paste the path into `utils/config.py` file.


>#location of CARLA egg file
>
>egg_file_abs_path = './carla-0.9.12-py3.7-linux-x86_64.egg

Now, the python script adds the file location to the system path, so we are ready to go...

### Collect the training data
To collect the data, first, the configuration file needed to be adjusted. There are different types of drivers implemented, the one with the most control is the "manual" driver. Data recording must be enabled and the name of the folder must be specified as to where to save the data. By default RGB camera + depth camera images will be saved in png format, on a resolution of 320x240.

>#set driving mode
>
>driver.set_driver_type(driver.manual)
>
>#record data
>
>record_data = True
>#use out dir, to append to the "out" directory, the location to store recording
>#use this for multiple scenario records, to organize your recordings
>
>out_dir = "/map_uuu_x_2/"
>#database name
>
>db_name = '_info.rec'

Now, the CARLA server can be started using `run_carla.sh` after the desired parameters were updated in the file. To collect the data, the auto_pilot.py script is needed to be started, after a successful connection, the `RGB_Cam` window will appear. This is important while capturing the keyboard, and commands can be exchanged with the server. Use `w a s d q` keys to control the car, `4 8 6 5` to label the direction and `r` to toggle the recording.

<p align="center"> 
  <img src="./info/RGB_cam.gif" alt="" width="250"></a>
</p>

The structure of the captured information:

>{ "index": 0,"throttle": 0.00,"steer": 0.00,"brake": 0.00,"hand_brake": "False","reverse": "False",
>"manual_geat_shift": "False","gear": 0.00,"velo": 1.8,"direction": "forward",
>"rgb_c": "cam_01_241.png","depth_c": "cam_d_01_241.png"}

A significant amount (min 15k samples) of data is needed to be collected to have a reasonably trained auto_pilot. Other types of a driver can be used to ease the data collection (like: `driver.set_driver_type(driver.autopilot)`) but is important to include recovery scenarios in the database.


### Train the DNN

To retrain the DNN use the `auto_pilot.ipynb` file. In the first step the right folder, files are needed to be configured.

>path='./out/map/*/_info.rec'
>
>val_path='./out/map_val/*/_info.rec'
>
>out_net = './ap.h5'

This is basically what is needed, and the retraining can start. In case the dataset is very unbalanced, built-in filtering functions can be used to undersample/oversample the majority respectively the minority classes. Basic augmentation is also possible, read in `auto_pilot.ipynb` file the details.

To load all the data without filtering, use:
>filters=[lambda x:True]
>
Complex querris can be constructed, using the data structure descibed above, like:
> filters = [lambda x: True if x['steer']==0.0 and x['index'] % 180 == 0 else False,
>            lambda x: True if x['steer']<=-0.1 else False,
>            lambda x: True if x['steer']>0.0 and x['steer']<0.1 and x['index']% 5 ==0.0 else False]

<p align="center"> 
  <img src="./info/unfiltered_data.png" alt="" width="300"></a>
  <img src="./info/filtered_data.png" alt="" width="300"></a>
  <img src="./info/oversampled_data.png" alt="" width="300"></a>
</p>

### Run simulation with DNN

To run a trained network use the `auto_pilot.py` script. A first step, ensure that the right driver type is selected, and the recording is inactive in the `utils/config.py` file.

>driver.set_driver_type(driver.inference)
>
>record_data = False

Run the CARLA server (`run_carla.sh`), followed by the `auto_pilot.py` execution. Use `w a s d q` keys to interact with the car, or `4 8 6 5` to provide the `direction` command for the DNN.

### Links

1. [Bojarski et all: End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf)
2. [CARLA - Open-source simulator for autonomous driving research](https://carla.org/)
3. [Paula Branco et all: A Survey of Predictive Modelling under Imbalanced Distributions](https://arxiv.org/pdf/1505.01658.pdf)
4. [Codevilla et all: End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/pdf/1710.02410.pdf)



