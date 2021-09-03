#!/usr/bin/env python3

"""
SOURCES
    https://youtu.be/2hM44nr7Wms
"""
# SOURCES
# https://youtu.be/2hM44nr7Wms

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import sys

# ../../Carla0.9.6/PythonAPI/carla/dist/carla-*%d.%d-%s.egg
# ../carla/dist/carla-*%d.%d-%s.egg
# Verwendung einex expliziten Pfades suboptimal.
# python3 muss jetzt verwendet werden.
try:
    sys.path.append(glob.glob(
        '../../Carla0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg')[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
import random
import time
import numpy as np
import cv2

# Constants
IM_WIDTH = 640
IM_HEIGHT = 480

# Variables
actor_list = []

# functions

def process_img(image, l_images):
    i = np.array(image.raw_data) 
    # print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) #rgba, a for alpha (opacity)
    i3 = i2[:, :, :3] # /255.0 # entire height, entire width, only rgb (no alpha)
    print(i3[1 , 1, :])
    #import pdb; pdb.set_trace()
    #cv2.imshow("image", i3)
    #cv2.waitKey(0)
    l_images.append(i3)
    return #i3/255.0 # normalize the data
    

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("model3")[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    actor_list.append(vehicle)
    # vehicle.set_autopilot(True)

    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")
    
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    
    sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(sensor) 
    
    #sensor.listen(lambda data: process_img(data))
    l_images = []
    sensor.listen(lambda data: process_img(data, l_images))

    time.sleep(10)
    
    sensor.stop()
    
    for im in l_images:
        cv2.imshow('image', im)
        cv2.waitKey(16)


finally:
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")
