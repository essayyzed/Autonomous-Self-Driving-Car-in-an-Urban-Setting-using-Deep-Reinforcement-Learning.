import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import tensorflow
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.models import Model


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


SHOW_PREVIEW = False
IMG_WIDTH = 640
IMG_HEIGHT = 480
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"

MEMORY_FRACTION = 0.8
MIN_REWARD = -200

EPISODES = 100

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.95  # 0.9975 99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10



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


class DQNAgent:
    def __init__(self):
        self.model = self.create_model
        self.target_model = self.create_model
        self.target_model.set_weights(self.model.get_weights())

        """
        we have a main network (`self.model`), 
        which is constantly evolving, 
        and then the target network(`target_model`), which we update every `n` things, 
        where `n` is whatever you want and things is something like steps or episodes.
        """

        #?  As we train, we train from randomly selected data from our replay memory:
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        
        # will track when it's time to update the target model
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()
        """
        We're going to use `self.training_initialized` to track when TensorFlow is ready to get going. 
        The first predictions/fitments when a model begins take extra long, 
        so we're going to just pass some nonsense information initially to prime our model to actually get moving.
        """
        self.terminate = False  # Should we quit ?
        self.last_logged_episode = 0
        self.training_initialized = False
        
        """
        Here, we're just going to make use of the premade `Xception` model, but you could make some other model, or import a different one. 
        Note that we're adding `GlobalAveragePooling` to our `output` layer, 
        as well as obviously adding the 3 neuron output that is each possible action for the agent to take_
        """
    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3)
                              )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
         
        predictions = Dense(3, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss='mse', optimizer=Adam(lr=0.001),metrics=["accuracy"] )
        return model
     
        """
        We need quick method in our DQNAgent for updating replay memory.
        As long as our method is this simple, we probably don't even need a method for this
        """
    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)
        
    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)
            
        new_current_states = np.array([transition[0] for transition in minibatch]) / 255
        
        with self.graph.as_default():
            future_qs_list = self.model.predict(
                new_current_states, PREDICTION_BATCH_SIZE)
        
        X = []
        
        y = []
        
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
                
            else:
                new_q = reward
                
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            X.append(current_state)
            y.append(current_qs)
            
            log_this_step = False
            if self.tensorboard.step > self.last_logged_episode:
                log_this_step = True
                self.last_log_episode = self.tensorboard.step

            with self.graph.as_default():
                self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE,
                               verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)
                
            if log_this_step:
                self.target_update_counter += 1
                
                
            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0
                
    def get_qs(self, state):
        return self.model.predict
    
    def train_in_loop(self):
        X = np.random.uniform(
            size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X, y, verbose=False, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
                
                
