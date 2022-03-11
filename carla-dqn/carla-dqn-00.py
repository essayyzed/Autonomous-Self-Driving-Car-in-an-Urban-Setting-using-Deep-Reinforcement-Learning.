import glob
import os
from random import random
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


SHOW_PREVIEW = False

    """
    SHOW_CAM is to show preview. Useful for debugging purposes (doesn't need it all the time)
    STEER_AMT is how much we want to steer in the current scenario it is `1` which mean full-steer
    we can change it later on.
    im_width is the image width of the camera's output
    im_height is the image height of the camera's output
    collision_hist is used to record the collision history in case of collision. Anything in this
    list will tell us that we crashed. 
    """


class CarEnv:
    
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    
    im_width = IMG_WIDTH
    im_height = IMG_HEIGHT
    
    actor_list = []
    
    front_camera = None
    collision_hist = []
    
    """
    CARLA work non the client-server architecture where server is the simulator itself and the
    client is created by calling `carla.Client()` 
    Once we have a client we can retrieve the world that is currently running.
    The `self.world` contains the list blueprints (in get_blueprint_library) that we can use for adding new actors into 
    the simulation.
    To retrive and use the actor of our own choice we use the filter() to get the desired actor.
    """
    
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        
        self.world = self.client.get_world()
        
        blueprint_library = self.world.get_blueprint_library()
        self.model3 = blueprint_library.filter('model3')[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')

        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        self.rgb_cam.set_attribute('fov', '110')

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        self.sensor = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.vehicle)

        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0))

        # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        time.sleep(4)

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(
            colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        self.vehicle.apply_control(
            carla.VehicleControl(brake=0.0, throttle=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    """
    `Step` method takes an action, and then returns the observation, reward, done, any_extra_info as per the usual reinforcement learning paradigm. 
    """


    def step(self, action):
        '''
        For now let's just pass steer left, center, right?
        0, 1, 2
        '''
        if action == 0:
            self.vehicle.apply_control(
                carla.VehicleControl(throttle=1.0, steer=0))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, steer=-1*self.STEER_AMT))
        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=1.0, steer=1*self.STEER_AMT))



        """
        grabbing the vehicle's speed, converting from velocity to KMH. I am doing this to avoid the agent learning to just drive in a tight circle
        """
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None
