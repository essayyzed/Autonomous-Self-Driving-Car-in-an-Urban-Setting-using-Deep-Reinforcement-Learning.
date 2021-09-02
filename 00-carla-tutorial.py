import glob
import os
import sys
import carla
import random
import time


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

    actor_list.append(vehicle)   # appending our car actor to a list so that at the end of simulation we can destroy it
    time.sleep(10)
finally:

    """
    After we are done with our task we need to clear up every thing using the builtin
    function `actor.destroy()
    """
    for actor in actor_list:
        actor.destroy()
    print("All Clear.....")
