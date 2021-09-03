import glob
import os
import sys
import carla
import random
import time
import numpy as np
import cv2

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass


# Setting the image size of camera image
IM_WIDTH = 640
IM_HEIGHT = 480


# def process_img(image):
#     i = np.array(image.raw_data)
#     # print(i.shape)
#     # print(dir(image))         # if you wanted to know the attributes used in the numpy image
    
#     # image is just in flat array of one number need to reshape to proper image using the dimensions given above 
#     # `4` denote the image is in RGBA `A is for Alpha (information)`
#     i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) 
#     # first index denote the total height, 2nd denote the total width and `:3` denote the image type which is RGB
#     # Remember we don't need the 4th parameter that is of information about the image
#     i3 = i2[:, :, :3] 
    
#     cv2.imshow("", i3)
#     cv2.waitKey(1) 
    
#     return i3/255.0   # for normalization of the data we want data in `0` & `1`

    
def process_img(image, l_images):
    i = np.array(image.raw_data)
    # print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))  # rgba, a for alpha (opacity)
    # /255.0 # entire height, entire width, only rgb (no alpha)
    i3 = i2[:, :, :3]
    print(i3[1, 1, :])
    l_images.append(i3)
    return  # i3/255.0 # normalize the data

    
actor_list = (
    []
)  # List of actors used in the simulation, will make easy for us to deal with actors

try:
    client = carla.Client(
        "localhost", 2000
    )  # Means CARLA will run on Local machine. Some public ip in case of remote server which is not recommended

    client.set_timeout(30)  # Defines for how much time the simulation should run

    world = (
        client.get_world()
    )  # consider it an abstract layer where all the simulations will take place

    blueprint_library = (
        world.get_blueprint_library()
    )  #  this where we can import all of our actors etc

    actor_car = blueprint_library.filter("model3")[
        0
    ]  # spawning `car actor` of tesla model3
    print(actor_car)

    spawn_point = world.get_map().get_spawn_points()[
        0
    ]  # specifying where the vehicle should be spawned

    vehicle = world.spawn_actor(
        actor_car, spawn_point
    )  # spawn the vehicle actor at the spawn point specified
    vehicle.apply_control(
        carla.VehicleControl(throttle=2.0, steer=0.0)
    )  # driving the car with the speed of 2.0 in a straight direction

    actor_list.append(
        vehicle
    )  # appending our car actor to a list so that at the end of simulation we can destroy it

    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    camera_bp.set_attribute("fov", "110")  # setting the field of view

    spawn_point = carla.Transform(
        carla.Location(x=2.5, z=0.7)
    )  # point where to fix the camera
    sensor = world.spawn_actor(
        camera_bp, spawn_point, attach_to=vehicle
    )  # passing the attributes set above for the sensor
    actor_list.append(sensor)
    l_images = []
    sensor.listen(lambda data: process_img(data, l_images))

    time.sleep(10)

    sensor.stop()

    for im in l_images:
        cv2.imshow('image', im)
        cv2.waitKey(16)

        


    time.sleep(10)
finally:

    """
    After we are done with our task we need to clear up every thing using the builtin
    function `actor.destroy()`
    """
    for actor in actor_list:
        actor.destroy()
    print("All Clear.....")
