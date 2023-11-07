#!/usr/bin/env python
# fvilmos, https://github.com/fvilmos

from utils.data_recorder import DataRecorder

from utils.resize import Resize
from utils.convert_direction import ConvertDirection
from utils.anomaly_injector import AnomalyInjector
from utils.intersection_handler import IntersectonHandler

from utils.pilot_model import PilotModel
from utils import config
from utils.buffer import Buffer


import os
import sys
from queue import Queue
from queue import Empty
import cv2
import numpy as np


# don't use GPU, carla uses allready
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

spath, _ = os.path.split(os.path.realpath(__file__))

sys.path.append(config.egg_file_abs_path)
print("carla .egg path:" + config.egg_file_abs_path)

import carla

def get_camera_image(frame):
    """
    Transforms png format to RGB for opencv
    Args:
        frame (carla.SensorData): carla sensor data object

    Returns:
        uint8: formated RGB frame
    """
    data = np.frombuffer(frame[0].raw_data, dtype=np.dtype("uint8"))
    data = np.reshape(data, (frame[0].height, frame[0].width, 4))
    if frame[1] == 'cam_01':
        data = data[:, :, :3]
    if frame[1] == 'cam_d_01':
        data = data[:,:,:1]

    return np.array(data)

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    """
    Sendor data callback

    Args:
        sensor_data (Sensor_Data): Carla object which holds the sensor data
        sensor_queue (object): Queue object
        sensor_name ([type]): [description]
        vehicle (Vehicle object): Ego vehicle object
    """
    sensor_queue.put((sensor_data, sensor_name))

def min_max_scaler_m1_p1(val,min_max=[0.7,-0.7]):
    return 2*(val-(min_max[1]))/(min_max[0] - (min_max[1]))-1

def min_max_scaler_m1_p1_inv(val,min_max=[0.7,-0.7]):
    return (val+1)/2*(min_max[0]-min_max[1])+min_max[1]

def main():

    # use this to define the record granurality
    sensor_update_time = config.sensor_update_time

    path, _ = os.path.split(os.path.realpath(__file__))

    path += '/out'
    path += config.out_dir

    print (path)

    model = None
    if config.driver.get_driver_type == config.driver.inference:
        
        # prepare model for inference
        if config.use_weights == True:
            PilotModel(config.NETWORK_IN_WIDTH,config.NETWORK_IN_HEIGHT,config.NETWORK_IN_CHANNELS,1,1,nr_of_predictions=config.NR_OF_PREDICTIONS,weights_file=config.model_file,img_buff_len=config.IMG_BUFF_LEN)
            model = PilotModel.get_model()
        else:
            # use full model file
            # TODO: implement here the model load
            pass


    # recorder object
    dr = DataRecorder(path,config.db_name)
    path = dr.enable_recording(config.record_data)

    client = carla.Client('localhost', 2000)
    client.set_timeout(8.0)
    
    # get Trafic Manager, needed to set for synch mode and autopilot
    tm = client.get_trafficmanager(8000)
    
    maps = client.get_available_maps()

    # set the world's map
    world = None
    if config.RANDOM_MAP == True:
        world = client.load_world(np.random.choice(maps))
    else:
        world = client.load_world(maps[config.MAP_INDEX])
    

    sensor_list = []
    sensor_queue = Queue(maxsize=4)

    imgb = Buffer(buff_length=config.IMG_BUFF_LEN,shape=[config.NETWORK_IN_HEIGHT,config.NETWORK_IN_WIDTH,config.NETWORK_IN_CHANNELS])
    anom_inject = AnomalyInjector()

    try:

        # needed to restor the original setup
        original_settings = world.get_settings()
        settings = world.get_settings()

        # Set synch mode
        settings.fixed_delta_seconds = 0.05
        settings.substepping = True
        settings.synchronous_mode = True
        tm.set_synchronous_mode(True)
        
        world.apply_settings(settings)

        w_debug = world.debug

        # Bluepints for the sensors
        blueprint_library = world.get_blueprint_library()
        v_bp = blueprint_library.filter('vehicle')[config.CAR_INDEX]
        
        # get one of the spam points
        map = world.get_map()

        # car will be spammed randomly on the map
        spam_points = map.get_spawn_points()
        if config.RANDOM_START_POSITION == True:
            start_position = np.random.choice(spam_points)
        else:
            start_position = map.get_spawn_points()[11]

        vehicle = world.spawn_actor(v_bp, start_position)

        # ignore traffic lights
        tm.ignore_lights_percentage(vehicle,100)

        # enable autopilot for data colection
        vehicle.set_simulate_physics(True)
        
        if config.driver.get_driver_type == config.driver.autopilot:
            print ('Autopilot on')
            vehicle.set_autopilot(True)
        else:
            print ('Autopilot off')
            vehicle.set_autopilot(False)

        # create camera sensor
        cam_bp = blueprint_library.find('sensor.camera.rgb')
        cam_bp.set_attribute("fov",str(100))
        cam_bp.set_attribute("image_size_x",str(320))
        cam_bp.set_attribute("image_size_y",str(240))
        cam_bp.set_attribute("sensor_tick",str(sensor_update_time))

        
        camera_sensor_transform = carla.Transform(carla.Location(x=2.0, z=1.6),carla.Rotation (pitch = -15.0))
        cam_01 = world.spawn_actor(cam_bp, camera_sensor_transform, attach_to=vehicle)
        cam_01.listen(lambda data: sensor_callback(data, sensor_queue, "cam_01"))
        sensor_list.append(cam_01)

        # imu is uesd as dummy sensor to attach 3rd person view
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_3rd_transform = carla.Transform(carla.Location(x=-4.0,y=0.0, z=3.0), carla.Rotation (pitch = -30.0))
        imu = world.spawn_actor(imu_bp, imu_3rd_transform,attach_to=vehicle)

        # create a depth camera and attach to the vehicle
        d_cam_bp = blueprint_library.find('sensor.camera.depth')
        #d_cam_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        d_cam_bp.set_attribute("fov",str(100))
        d_cam_bp.set_attribute("image_size_x",str(320))
        d_cam_bp.set_attribute("image_size_y",str(240))
        d_cam_bp.set_attribute("sensor_tick",str(sensor_update_time))

        cam_d_01 = world.spawn_actor(d_cam_bp, camera_sensor_transform,attach_to=vehicle)
        cam_d_01.listen(lambda data: sensor_callback(data, sensor_queue, "cam_d_01"))
        sensor_list.append(cam_d_01)
        cc = carla.ColorConverter.LogarithmicDepth

        # get sepectator object, is needed for 3rd person update 
        spectator = world.get_spectator()

        count = 1
        dir = 'forward'
        cmd = 'forward'
        intersection_state_obj = IntersectonHandler(scan_distance_next=8,scan_distance_prev=3,rotation_rate=0.1)

        # Main loop
        while True:

            # Tick the server
            world.tick()

            # dict to hold sensor file info
            file_names = {}
            
            try:
                # update 3rd person view with vehicle position
                spectator.set_transform(imu.get_transform())

                # collect ego vehicle information
                vcontrol = vehicle.get_control()

                # convert veocity to km/h
                vel = vehicle.get_velocity()
                vvel = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                
                # loop over sensors
                for _ in range(len(sensor_list)):

                    # get synchronized data from sensors, andsave it
                    s_frame = sensor_queue.get(True,0.001)

                    if config.record_data == True:

                        if dr.record == True:
                            if s_frame[1] == 'cam_d_01':
                                file_names['depth'] = s_frame[1] + '_' + str(s_frame[0].frame)+'.png'
                                # convert depth info to logaritmic representation
                                s_frame[0].save_to_disk(path + file_names['depth'],cc)
                                
                            else:
                                file_names['rgb'] = s_frame[1] + '_' + str(s_frame[0].frame)+'.png'
                                s_frame[0].save_to_disk(path + file_names['rgb'])

                    #==================
                    # filter sesor out
                    #==================
                    if s_frame[1] != 'cam_01':
                        continue

                    waypoint = map.get_waypoint(vehicle.get_location(),project_to_road=True,lane_type=carla.LaneType.Driving)    

                    cmd = intersection_state_obj.run(wp=waypoint,vehicle=vehicle)

                    img = get_camera_image(s_frame)
                    dimg = img.copy()

                    k = cv2.waitKey(1)

                    ########################  
                    # Autopilot with  
                    # anomaly injection mode active
                    ########################  
                    if config.driver.get_driver_type == config.driver.crazyautopilot:
                        anom_inject.run(vehicle,vcontrol)
                        cv2.putText(dimg,"driver: " + '{}'.format('crazyautopilot'),(30,80),config.typef,config.sizef*1.2,config.color,config.sizeb)

                    ########################  
                    # Autopilot mode active
                    ########################  
                    if config.driver.get_driver_type == config.driver.autopilot: 
                        cv2.putText(dimg,"driver: " + '{}'.format('autopilot'),(30,80),config.typef,config.sizef*1.2,config.color,config.sizeb)
                        dir = cmd

                    ########################  
                    # Manual control active
                    #######################  
                    elif config.driver.get_driver_type == config.driver.manual:
                        cv2.putText(dimg,"driver: " + '{}'.format('manual'),(30,80),config.typef,config.sizef*1.2,config.color,config.sizeb)

                    ######################## 
                    # Inference mode active
                    ########################
                    elif config.driver.get_driver_type == config.driver.inference:
                        cv2.putText(dimg,"driver: " + '{}'.format('NN'),(30,80),config.typef,config.sizef*1.2,config.color,config.sizeb)

                        img,vvel,v_direction = Resize((config.NETWORK_IN_WIDTH,config.NETWORK_IN_HEIGHT))([img,vvel,dir])
                        img,vvel,v_direction = ConvertDirection()([img,vvel,dir])

                        buff = imgb.update(img,skipp_rate=2)
                        buff = np.array(buff[None,:,:,:,:])

                        tvvel = np.array([[vvel]], dtype=np.float32) 
  
                        arr_steer = model.predict({'img_in':buff,'cmd_in': np.array([[[int(v_direction)]]]), 'velo_in': tvvel})

                        key = 'steer_out_0'
                        keyt = 'throttle_out_0'

                        steer_ub = min_max_scaler_m1_p1_inv(arr_steer[key][0])
                        throttle_n = arr_steer[keyt][0]

                        if  (arr_steer is not None):
                            # normalize
                            steer = round(float(min(0.7, max(-0.7, steer_ub))),3)

                            if k ==-1:
                                # ** apply controls to vehicel
                                throttle = round(float(min(0.45, max(0.35, throttle_n))),2)

                                if config.use_constant_thorttle == True:
                                    vcontrol.throttle = config.constant_thorttle
                                else:
                                    vcontrol.throttle = throttle

                                vcontrol.steer = steer
                                vcontrol.brake = 0

                    # ** 
                    #================
                    # TODO: steering, automatic recovery
                    # if k == -1 and np.abs(vcontrol.steer)>0.0:
                    #     if vcontrol.steer < config.drive_increment:
                    #         vcontrol.steer += config.drive_increment
                    #     if vcontrol.steer > config.drive_increment:
                    #         vcontrol.steer -= config.drive_increment


                    # set throttle
                    if k == ord('w'):
                        vcontrol.throttle += config.drive_increment
                    if k == ord('s'):
                        vcontrol.throttle -= config.drive_increment
                    
                    # steer left / right
                    if k == ord('a'):
                        vcontrol.steer -= config.drive_increment
                    
                    if k == ord('d'):
                        vcontrol.steer += config.drive_increment

                    # reverse
                    if k == ord('q'):
                        if  vcontrol.reverse == True:
                            vcontrol.reverse = False
                        else:
                            vcontrol.reverse = True

                    if k == ord('4'):
                        dir = 'left'
                    
                    if k == ord('6'):
                        dir = 'right'

                    if k == ord('8'):
                        dir = 'forward'
                    
                    if k == ord('5'):
                        dir = 'keep_lane'

                    if k == ord('r'):
                        if dr.record == True:
                            dr.record = False
                        else:
                            dr.record = True
                    

                    # exit on ESC
                    if k == 27:
                        return


                    vcontrol.steer = round (min(0.7, max(-0.7, vcontrol.steer)),2)
                    vcontrol.throttle = min(1.0, max(0.0, vcontrol.throttle))

                    vehicle.apply_control(vcontrol)
                    vcontrol.brake = 0 

                    # draw driwing wheel
                    # poloar coordinates
                    r = 20
                    x = int(r * np.cos(-1.57 + 4*vcontrol.steer))
                    y = int(r * np.sin(-1.57 + 4*vcontrol.steer))

                    cv2.circle(dimg,(160,180),r,[255,0,0],1)
                    cv2.circle(dimg,(160+x,180+y),5,[255,0,255],-1)

                    # draw rgb camera image, used to capture the controls
                    cv2.putText(dimg,"steer: " + '{:.2f}'.format(vcontrol.steer),(30,20),config.typef,config.sizef,config.color,config.sizeb)
                    cv2.putText(dimg,"throttle: " + '{:.2f}'.format(vcontrol.throttle),(30,40),config.typef,config.sizef,config.color,config.sizeb)
                    cv2.putText(dimg,"velocity: " + '{:.2f}'.format(vvel),(30,60),config.typef,config.sizef,config.color,config.sizeb)
                    cv2.putText(dimg,"direction: " + '{}'.format(dir),(30,100),config.typef,config.sizef*1.2,config.color,config.sizeb)
                    
                    if dr.record == True:
                        cv2.putText(dimg,"Rec",(280,20),config.typef,config.sizef,[0,0,255],2)

                    cv2.imshow('RGB_CAM', dimg) 

            except Empty:
                # no data in the quieue
                pass

            count +=1
            #save data          
            dr.save_sensor_data(file_names,vcontrol=vcontrol,velo=vvel,direction=dir,junction=cmd,waypoint=str("[]"))
    except :
        # gather error info
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print (exc_type, exc_value,exc_traceback)

    finally:

        # save and restore on exit
        print ('Destroy...')
        dr.close_file()
        world.apply_settings(original_settings)
        for sensor in sensor_list:
            sensor.destroy()

        imu.destroy()
        vehicle.destroy()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Done')
