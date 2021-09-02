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
    )  # Means CARLA will run on Local machine. Some public ip in case of remote server which is not recommanded

    client.set_timeout(30)  # Defines for how much time the simulation should run

    world = (
        client.get_world()
    )  # consider it an abstract layer where all the simulations will take place

    blueprint_library = world.get_blueprint_library() #  this where we can import all of our actors etc

    actor_car = blueprint_library.filter("model3")[0] # spawning `car actor` of tesla model3
    print(actor_car)
    
    spawn_point = world.get_map().get_spawn_points()[0]  # specfiying where the vehicle should be spawned



finally:

    """
    After we are done with our task we need to clear up every thing using the builtin
    function `actor.destroy()
    """
    for actor in actor_list:
        actor.destroy()
    print("All Clear.....")
