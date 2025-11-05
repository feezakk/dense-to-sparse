import gym
import numpy as np
import math
import carla
from gym import spaces
import random
import time
import cv2

from .toolkit.planner import FixedEndingPlanner
from .toolkit import TTCCalculator, get_location_distance, get_vehicle_pos
from .carla_base_env import CarlaBaseEnv

from typing import Tuple

from pathlib import Path

import os

from collections import deque
import weakref, queue

EGO_SPAWN_POINT = [
    # one-turn scenarios
    
    # [71.58, -6.94, 1.01,  0.953298, -62.641895, 0.000000], #ok
    [74.51, 7.51, 1.01,  0.953298, 45, 0.000000], #ok
    [68.92, 72.00, 1.01, 4.703701, -280, 0.000000], #ok
    [-90.68, 121.28, 1.01, 0, 0, 0], # ok
    [-203.12, 64.82, 1.01, 0, 90, 0], #ok

    # straight scenarios

    [-15.60, -161.00, 1.0, 0, 180, 0], #ok
    [-185.95, -159.74, 1.0, 0, 0, 0], #ok
    [-19.18, 57.56, 1.00, 0, 180, 0],  #ok
    [-86.19, 57.56, 1.00, 0, 0, 0], #ok

    # multiple-turn scenarios

    [-23.86, -245.72,1.0, 0, 200, 0], #ok
    [-27.81, -95.85, 1.00, 0, 270, 0], #ok
    
    ]


EGO_END_POINT = [
    # one-turn scenarios
    
    [75.51, 51.44, 1.01,  0.953298, 140, 0.000000], #ok
    [-90.68, 118.28, 1.01, 0, 180, 0], # ok
    [72.92, 72.00, 1.01, 4.703701, -100, 0.000000], #ok
    [-190.41, 110.6, 1.00, 0, 0, 0], #ok

    # straight scenarios

    [-58.60, -161.00, 1.0, 0, 180, 0], #ok
    [-24.95, -158.74, 1.0, 0, 0, 0], #ok
    [-82.18, 53.56, 1.00, 0, 180, 0],  #ok
    [-22.19, 60.56, 1.00, 0, 0, 0], #ok

    # multiple-turn scenarios

    [-169.98, -248.17, 1.0, 0, 180, 0], #ok
    [-63.37, -95.85, 1.00, 0, 270, 0], #ok
    
    ]



# DISCRETE_ACC = [0.0, 0.2, 0.4, 0.6, 1.0] # discrete value of accelerations
DISCRETE_ACC = [0.0 , 0.3] # discrete value of accelerations
DISCRETE_STEER = [-0.2, -0.1, 0.0, 0.1, 0.2] # discrete value of steering angles

SWING_STEER = 0.04 # The background vehicle steer for swing .
SWING_AMPLITUDE = 0.2 # The y-axis amplitude of background vehicle steer.
SWING_TRIGGER_DIST = 20 # The distance between ego and background vehicle that triggers swing.
PID_COEFFS = [0.03, 0.0, 0.03] # The PID controller parameter for background vehicle lane keeping.

REWARD = {
      'desired_speed': 5, # desired speed (m/s)
      'reward_overtake_dist': 8, # The distance that triggers overtake reward.
      'early_lane_change_dist': 10, # The distance that penalizes early lane change.
      'lane_width': 3.5,
      'stay_same_lane': 0.3,
      'exceeding': 200.0,
      'overtake': 200.0,
      'early_lane_change': 0.0,
      'scales':
        {
          'waypoint': 2.0,
          'speed': 0.5,
          'out_of_lane': 3.0,
          'collision': 30.0,
          'time': 0.0,
          'destination_reached': 20.0,
          'early_lane_change': 0.0,
          'speed': 0.5,
        }
}

TERMINAL = {
      'out_lane_thres': 5, # threshold for out of lane
      'time_limit': 1000, # maximum timesteps per episode
      'left_lane_boundry': 3.7, # out of lane boundry
      'right_lane_boundry': 17.7,
      'lane_width': 3.4,
      'terminal_dist': 100, # terminate tasks
}



# --------------------------------------------------------------------------------
# A helper function to compute 2D distances
# --------------------------------------------------------------------------------
def distance_2d(loc1, loc2):
    return math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)

# --------------------------------------------------------------------------------
# Example single-file environment for overtaking
# --------------------------------------------------------------------------------
class CarlaLaneFollowingEnv(gym.Env):
    def __init__(self, config):
        super().__init__()

        self._config = config

        # Connect to a running CARLA instance or create a new one
        self.client = carla.Client("localhost", 3000)
        self.client.set_timeout(300.0)

        w = self.client.get_world()

        try:
            name = w.get_map().name
        except RuntimeError:
            name = ""
        if name != "Carla/Maps/Town07":
            w = self.client.load_world("Town07")

        self.world = w
        self.map = self.world.get_map()

        settings = self.world.get_settings()
        if not settings.synchronous_mode or settings.fixed_delta_seconds != 0.05:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
        self._sync_enabled = True

        # self.world.wait_for_tick(5.0)

        print("CARLA environment initialized")
        print("Map name:", self.map.name)

        # remove old vehicles and sensors (in case they survived)
        self.world.tick()
        

        # Load or get the world
        # self.world = self.client.get_world()
        # self.map = self.world.get_map()

        # Keep track of spawned actors to destroy them on reset
        self.ego = None
        self.nonego = None
        self.actors = []

        # Time step for counting
        self._time_step = 0
        self._max_time_step = 2000

        # Track collisions
        self.collision_detected = False
        self.collision_sensor = None

        # Action/Observation space
        self.action_space = self._setup_action_space()
        self.observation_space = self._setup_observation_space()

        # PID error memory for nonego
        self.prev_errors = {"last_error": 0.0, "integral": 0.0}
        self.swing_direction = 1

        # Camera sensor
        self.camera_image = None
        self.camera_image2 = None
        # Add a deque to store the last 4 frames
        self.frame_buffer = deque(maxlen=4)

        # Lane invasion detection
        self.lane_invasion_detected = False
        self.lane_invasion_hist = []

        # Collision Detection
        self.collision_detected = False
        self.collision_hist = []

        # Setup blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        self.spawn_index = np.random.randint(0, len(EGO_SPAWN_POINT))

        self.ego_transform = carla.Transform(
            carla.Location(x = EGO_SPAWN_POINT[self.spawn_index][0], y = EGO_SPAWN_POINT[self.spawn_index][1], z = EGO_SPAWN_POINT[self.spawn_index][2]),
            carla.Rotation(pitch = EGO_SPAWN_POINT[self.spawn_index][3] , yaw = EGO_SPAWN_POINT[self.spawn_index][4], roll = EGO_SPAWN_POINT[self.spawn_index][5]),
        ) 

        self.exceeding = False
        self.overtake = False
        self.last_ego_y = EGO_SPAWN_POINT[self.spawn_index][1]

        self.swing_direction = 1

        self.prev_errors = {"last_error": 0.0, "integral": 0.0}  # For PID controller

        self.low_speed_start_time = None

        self.speed_kmh = None

        # For Data Collection

        # Keep track of episode and step
        save_dir="data"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.episode_id = 0
        self.timestep = 0

        self.episode_buffer = []
        self.all_episodes_data = []  # optional: store all episodes in memory if you want

        self.initial_distance_to_goal = 270
        self.previous_lane_invasions = 0
        self.previous_collisions = 0


    # --------------------------------------------------------------------------------
    # Reset the ego vehicle spawning
    # --------------------------------------------------------------------------------

    def reset_vehicle(self):
        
        vehicle_blueprint = self.blueprint_library.find('vehicle.tesla.model3')
        self.ego = self.world.spawn_actor(vehicle_blueprint, transform=self.ego_transform)

        if self.ego is None:
            # create vehicle
            blueprint_library = self.world.get_blueprint_library()
            self.ego = self.world.try_spawn_actor(vehicle_blueprint, self.ego_transform)

            if self.ego is None:
                raise RuntimeError("Unable to spawn vehicle at the chosen spawn point.")
            
            print("Ego Vehicle Spawned")
        else:
            self.ego.set_transform(self.ego_transform)

            print("Ego Vehicle Spawned")

        self.ego.set_target_velocity(carla.Vector3D())

    # --------------------------------------------------------------------------
    # Observations
    # --------------------------------------------------------------------------

    def _pull_latest_image(self, timeout=2.0):
        rgb = None
        if hasattr(self, "_img_q"):
            # drain to most recent frame
            end = time.time() + timeout
            while True:
                try:
                    rgb = self._img_q.get(timeout=max(0, end - time.time()))
                    # try to get newer ones without blocking
                    while True:
                        rgb = self._img_q.get_nowait()
                except queue.Empty:
                    break
        if rgb is None:
            rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        return rgb

    def _get_observation(self):
        rgb = self._pull_latest_image(timeout=0.5)
        self.camera_image2 = rgb
        self.camera_image = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
        return {
            "image": self.camera_image,
            "collision": 1 if self.collision_detected else 0,
            "lane_invasion": 1 if self.lane_invasion_detected else 0,
        }

    # --------------------------------------------------------------------------
    # Gym methods: reset, step, (optional) render, close
    # --------------------------------------------------------------------------
    def reset(self):
        self._clean_actors()


        self.spawn_index = np.random.randint(0, len(EGO_SPAWN_POINT))

        self.ego_transform = carla.Transform(
            carla.Location(x = EGO_SPAWN_POINT[self.spawn_index][0], y = EGO_SPAWN_POINT[self.spawn_index][1], z = EGO_SPAWN_POINT[self.spawn_index][2]),
            carla.Rotation(pitch = EGO_SPAWN_POINT[self.spawn_index][3] , yaw = EGO_SPAWN_POINT[self.spawn_index][4], roll = EGO_SPAWN_POINT[self.spawn_index][5]),
        ) 

        self.end_point = carla.Transform(
            carla.Location(x = EGO_END_POINT[self.spawn_index][0], y = EGO_END_POINT[self.spawn_index][1], z = EGO_END_POINT[self.spawn_index][2]),
            carla.Rotation(pitch = EGO_END_POINT[self.spawn_index][3] , yaw = EGO_END_POINT[self.spawn_index][4], roll = EGO_END_POINT[self.spawn_index][5]),
        ) 

        print(self.ego_transform)
        print(self.end_point)

        
        # Keep track of spawned actors to destroy them on reset
        self.actors = []

        # Time step for counting
        self._time_step = 0
        self._max_time_step = 2000

        # Track collisions
        self.collision_detected = False
        self.collision_sensor = None

        # Camera sensor
        self.camera_image = None

        # Lane invasion detection
        self.lane_invasion_detected = False
        self.lane_invasion_hist = []

        # Collision Detection
        self.collision_detected = False
        self.collision_hist = []

        self.world.tick()

        self.reset_vehicle()
        if self.ego is not None:
            self.actors.append(self.ego)

        # Keep track of actors to destroy later
        # self.actors = [self.nonego, self.ego]

        # Initialize the vehicle with default controls
        self.ego.apply_control(carla.VehicleControl(manual_gear_shift=False, reverse=False, hand_brake=False,steer=0.0, throttle=0.0, brake=0.0))
        
        self.episode_start = time.time()

        # Attach collision sensor to ego to detect collisions
        self.setup_collision_sensor()

        # Attach lane invasion sensor to ego to detect lane invasions
        self.setup_lane_invasion_sensor()

        # Attach camera sensor to ego
        self.setup_camera()

        self.world.tick()
        
        # Path planning
        ego_dest = EGO_END_POINT[self.spawn_index]
        dest_location = carla.Location(x=ego_dest[0], y=ego_dest[1], z=ego_dest[2])
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]

        # Set spectator for debugging
        spectator = self.world.get_spectator()
        self.ego_transform.location.z += 50
        self.ego_transform.rotation.pitch = -70
        spectator.set_transform(self.ego_transform)
        self.swing_direction = 1

        self.prev_errors = {"last_error": 0.0, "integral": 0.0}  # For PID controller

        self.low_speed_start_time = None
        self.speed_kmh = None
        self.previous_collisions = 0
        self.previous_lane_invasions = 0
        

        print("Environment reset")

        # IMPORTANT: clear out the old buffer
        self.episode_buffer = []
        self.timestep = 0

        self.initial_distance_to_goal = self.ego.get_location().distance(self.end_point.location)
        self.last_distance_to_goal = self.ego.get_location().distance(self.end_point.location)

        self.last_num_completed = 0

        # Return initial observation
        return self._get_observation()
    
    # ------------------------------------------------
    def _save_episode_to_disk(self, episode_buffer):
        # This is just a placeholder showing how you might do it
        import pickle

        # Suppose you have an episode counter
        ep_id = self.episode_id # or track it differently

        save_path = self.save_dir / f"episode_{ep_id}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(episode_buffer, f)
        print(f"Episode {ep_id} saved to {save_path}")

    def step(self, action):
        """
        Applies action (acc, steer) to the ego vehicle, applies
        nonego control, ticks the world, calculates reward, checks terminal.
        """

        # obs_current = self._get_observation()

        # if self.save_every_n and (self.timestep % self.save_every_n == 0):
        #     np.save(str(img_path.with_suffix('.npy')), obs_current["image"])
        #     oc = str(img_path)
        # else:
        #     oc = None  # or keep a placeholder

        # curr_path = "data/" + str(self.episode_id) + "/curr"
        # curr_path = Path(curr_path)
        # next_path = "data/" + str(self.episode_id) + "/next"
        # next_path = Path(next_path)

        # curr_path.mkdir(parents=True, exist_ok=True)
        # next_path.mkdir(parents=True, exist_ok=True)

        # # 2. Save the image to disk if it exists
        # if obs_current["image"] is not None: 
        #     # Create a filename like episode_0_step_0.jpg
        #     img_name = f"episode_{self.episode_id}_step_{self.timestep}"
        #     img_path = curr_path / img_name
            
        #     # obs["image"] is a numpy array in BGR or RGB
        #     # cv2.imwrite(str(img_path), obs_current["image"]) 
        #     np.save(str(img_path.with_suffix('.npy')), obs_current["image"]) 
            
        #     # Replace the image in your transition with the *filename* only
        #     oc = str(img_path)

        # 1. Apply Ego action
        self.apply_control(action)

        # 2. Incrementing time
        self._time_step += 1

        # 3. Tick the world
        self.world.tick()
        
        # 4. Update waypoint

        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]

        #4. compute speed
        self.velocity = self.ego.get_velocity()
        self.speed_kmh = 3.6 * math.sqrt(self.velocity.x**2 + self.velocity.y**2 + self.velocity.z**2)

        # 5. Compute observation
        obs = self._get_observation()

        # 6) Save the *next* obs image to disk
        # if obs["image"] is not None:
        #     img_name = f"episode_{self.episode_id}_step_{self.timestep}_next"
        #     # img_path = next_path / img_name
            
        #     # cv2.imwrite(str(img_path), obs["image"]) 
        #     np.save(str(img_path.with_suffix('.npy')), obs["image"])
        #     on = str(img_path)

        # 7. Compute reward
        reward, info_dict = self._compute_reward()


        # 4. Check termination
        done, terminal_info = self._check_termination()

        if done == True:
            print("terminal_info", terminal_info)


        info = {**info_dict, **terminal_info}

        # store the transition in the buffer
        # transition = {
        #     "observation": oc,
        #     "action": action,
        #     "reward": reward,
        #     "done": done,
        #     "next_observation": on,
        #     "info": info
        # }

        # self.episode_buffer.append(transition)

        # if episode ended, optionally store or process
        if done:
            # Example 1: keep it in `all_episodes_data`
            # self.all_episodes_data.append(self.episode_buffer)

            # Example 2: or write it to disk
            # self._save_episode_to_disk(self.episode_buffer)
            self.episode_id += 1

        # 7. show the image
        if self.camera_image2 is not None:
            resized_image = cv2.resize(self.camera_image2, (512, 512), interpolation=cv2.INTER_LINEAR)
              
            cv2.imshow("EgoCamera", resized_image)
            cv2.waitKey(1)

        self.timestep += 1

        return obs, reward, done, info

    def render(self, mode='human'):
        """
        If you want to visualize. Could do direct PyGame window or
        rely on the CARLA manual_control.py approach, etc.
        """
        pass

    def close(self):
        """
        Properly close the env, destroy actors, etc.
        """
        
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self._sync_enabled = False

        self._clean_actors()
        pass

    # --------------------------------------------------------------------------
    # Setup Spaces
    # --------------------------------------------------------------------------
    def _setup_action_space(self):
                
        self.n_steer = len(DISCRETE_STEER)
        self.n_acc = len(DISCRETE_ACC)
        return spaces.Discrete(self.n_steer * self.n_acc)
    
    def _setup_observation_space(self):
        """
        We have:
        - 'image': an image of shape [3, 128, 128], dtype uint8, range [0..255].
        - 'collision': a discrete flag (0 or 1).
        - 'lane_invasion': a discrete flag (0 or 1).
        """
        camera_space = spaces.Box(
            low=0, high=255, 
            shape=(128 , 128 , 3), 
            dtype=np.uint8
        )

        collision_space = spaces.Discrete(2)     # 0 or 1
        lane_invasion_space = spaces.Discrete(2) # 0 or 1

        return spaces.Dict({
            "image": camera_space,
            "collision": collision_space,
            "lane_invasion": lane_invasion_space
        })

    # --------------------------------------------------------------------------
    # Sensor Setup
    # --------------------------------------------------------------------------
    
    def setup_camera(self):
        # self.camera = self.blueprint_library.find('sensor.camera.rgb')
        self.camera = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.camera.set_attribute("image_size_x", f"{512}")
        self.camera.set_attribute("image_size_y", f"{512}")
        self.camera.set_attribute("fov", "110")
        self.camera.set_attribute("sensor_tick", "0.05")  # = fixed_delta_seconds

        # camera_spawn = carla.Transform(carla.Location(x=1.5, z=1.8), carla.Rotation(pitch=0)) 
        camera_spawn = carla.Transform(carla.Location(z=10), carla.Rotation(pitch=-90)) 
        self.camera_sensor = self.world.spawn_actor(self.camera, camera_spawn, attach_to=self.ego)
        self.actors.append(self.camera_sensor)

        self._img_q = queue.Queue(maxsize=8)
        ws = weakref.ref(self)

        def _on_image(img, q=self._img_q, ws=ws):
            s = ws()
            if s is None:
                return
            # minimal and fast; no resize here
            img.convert(carla.ColorConverter.CityScapesPalette)
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))[:, :, :3]
            try:
                q.put_nowait(arr)
            except queue.Full:
                pass  # drop

        self.camera_sensor.listen(_on_image)

    def setup_collision_sensor(self):
        collision_sensor_bp = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=self.ego)
        self.actors.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

    def setup_lane_invasion_sensor(self):
        lane_invasion_sensor_bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_invasion_sensor_bp, carla.Transform(), attach_to=self.ego)
        self.actors.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self.lane_invasion_data(event))

    def camera_callback(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        # Convert raw data to a numpy array (H x W x 4) => (H x W x 3)
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # Remove alpha channel and convert BGR -> RGB if needed:
        rgb = array[:, :, :3][:, :, ::-1]
        # rgb = array.reshape((512, 512, 4))[:, :, :3]

        rgb = rgb.copy()

        car_mask = np.all(rgb == [0, 0, 255], axis=-1)
        rgb[car_mask] = [0, 0, 255]  # pure blue in RGB

        rgb_128 = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
        self.camera_image2 = rgb
        self.camera_image = rgb_128

    def collision_data(self, event):
        self.collision_hist.append(event)
        self.collision_detected = True

    def lane_invasion_data(self, event):
        self.lane_invasion_hist.append(event)
        self.lane_invasion_detected = True

    # --------------------------------------------------------------------------
    #Apply Control
    # --------------------------------------------------------------------------
    def apply_control(self, action) -> None:
        control = self._get_vehicle_control(action)
        self.ego.apply_control(control)

    # --------------------------------------------------------------------------
    # Control: EGO
    # --------------------------------------------------------------------------
    def _get_vehicle_control(self, action):
        """
        Convert (acc, steer) to throttle/brake and CARLA steer.
        action: np.array([acc, steer]) in continuous domain.
        """
        # print("action", action)

        acc = DISCRETE_ACC[action // self.n_steer]
        steer = DISCRETE_STEER[action % self.n_steer]
       
        throttle = acc #np.clip(acc, 0, 0.2)
        brake = 0.0

        # steer in CARLA is left-negative, right-positive,
        # but it can vary depending on your coordinate system.
        # We invert the sign if needed:
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

    # --------------------------------------------------------------------------
    # Reward
    # --------------------------------------------------------------------------
    def get_vehicle_pos(self , vehicle: carla.Actor) -> Tuple[float, float]:
        location = vehicle.get_transform().location
        return location.x, location.y
    
    def get_vehicle_velocity(self, vehicle: carla.Actor) -> Tuple[float, float]:
        velocity = vehicle.get_velocity()
        return velocity.x, velocity.y
    
    def get_lane_offset(self):
        """
        Calculate the lane offset (distance between the vehicle and the lane center).
        """
        # Get the vehicle location
        vehicle_location = self.ego.get_location()

        # Get the waypoint corresponding to the vehicle's current position
        map = self.world.get_map()
        waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    
        # Get the location of the lane center (waypoint)
        lane_center_location = waypoint.transform.location

        # Calculate the distance between the vehicle and the lane center
        lane_offset = math.sqrt((vehicle_location.x - lane_center_location.x)**2 +
                            (vehicle_location.y - lane_center_location.y)**2)
    
        return lane_offset
    

    def get_angle_offset(self):
        """
        Calculate the lane offset (distance between the vehicle and the lane center).
        """
        vehicle_location = self.ego.get_location()
        waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        waypoint_vector = np.array([waypoint.transform.get_forward_vector().x, waypoint.transform.get_forward_vector().y])
        vehicle_forward_vector = np.array([self.ego.get_transform().get_forward_vector().x, self.ego.get_transform().get_forward_vector().y])
        angle_offset = np.arccos(np.clip(np.dot(waypoint_vector, vehicle_forward_vector) /
                               (np.linalg.norm(waypoint_vector) * np.linalg.norm(vehicle_forward_vector)), -1.0, 1.0))
        angle_offset = angle_offset/ np.pi
    
        return angle_offset
    

    def _compute_reward(self):

        """
        Reward breakdown:
            + lane keeping  (– if outside bounds)
            + heading alignment
            + smooth speed around target
            – low-speed timeout
            – lane invasions
            – collisions
            + progress / goal reached
        Returns
        -------
        total_reward : float
        info         : dict   # individual components for logging
        """

        total_reward = 0.0
        # -----------------------------
        # cached state
        # -----------------------------
        speed  = self.speed_kmh
        lane_d = self.get_lane_offset()      # metres
        ang_d  = self.get_angle_offset()     # radians
        t_now  = time.time()

        reward_components = {}

        r_waypoints = 0.0
        if self.num_completed > 0:
            r_waypoints = 60.0 * self.num_completed
            reward_components["waypoint"] = r_waypoints
        else:
            r_waypoints = -2.0
            reward_components["waypoint"] = r_waypoints

        ego = self.ego
        ego_location = np.array([*self.get_vehicle_pos(ego)])
        ego_velocity = np.array([*self.get_vehicle_velocity(ego)])

        r_speed = 0.0
        speed_parallel = 0.0
        speed_perpendicular = 0.0
        if len(self.waypoints) > 0:
            # compute the wpt line direction
            next_waypoint = self.waypoints[0]
            next_location = np.array([next_waypoint[0], next_waypoint[1]])
            yaw_radius = next_waypoint[2] * np.pi / 180
            waypoint_direction = np.array([np.cos(yaw_radius), np.sin(yaw_radius)])

            # compute the perpendicular direction
            goal_offset = next_location - ego_location
            perp_direction = goal_offset - np.dot(goal_offset, waypoint_direction) * waypoint_direction
            perp_direction_norm = np.linalg.norm(perp_direction)
            if perp_direction_norm > 0.05:
                perp_direction = perp_direction / perp_direction_norm
            else:
                perp_direction = np.array([0.0, 0.0])

            # compute the speed reward
            desired_speed = REWARD["desired_speed"]
            speed_parallel = np.dot(ego_velocity, waypoint_direction)
            speed_perpendicular = np.dot(ego_velocity, perp_direction)
            reward_components["r_speed"] = (desired_speed - np.abs(speed_parallel - desired_speed) - 2 * max(speed_perpendicular, -0.5)) * 2.0
        else:
            # No waypoints available, set speed reward to 0
            reward_components["r_speed"] = 0.0

       

        # (A) Lane centre – quadratic well, clamped at 2 m
        lane_k   = 40.0          # shape factor
        lane_rad = min(lane_d, 2.0)
        r_lane   = -lane_k * lane_rad**2
        reward_components["lane"] = r_lane

        # (B) Heading alignment – quadratic in radians
        head_k = 100.0
        r_head = -head_k * min(ang_d, 0.5)**2
        reward_components["heading"] = r_head

        # (C) Speed profile – Gaussian around 20 km/h
        # target_v = 20.0
        # sigma_v  = 5.0
        # r_speed  = 30.0 * math.exp(-0.5*((speed-target_v)/sigma_v)**2)
        # reward_components["speed"] = r_speed

        # (D) Low-speed penalty (<1 km/h for >10 s)
        if speed < 1.0:
            if self.low_speed_start_time is None:
                self.low_speed_start_time = t_now
            low_speed_elapsed = t_now - self.low_speed_start_time
            if low_speed_elapsed > 10.0:
                r_low_speed = -50.0          # one-off per step
            else:
                r_low_speed = -2.0           # gentle nudging
        else:
            self.low_speed_start_time = None
            r_low_speed = 0.0
        reward_components["low_speed"] = r_low_speed

        # (E) Lane invasion – penalise *new* invasions
        new_inv = len(self.lane_invasion_hist) - self.previous_lane_invasions
        r_invasion = -20.0 * new_inv
        self.previous_lane_invasions += new_inv
        reward_components["invasion"] = r_invasion

        # (F) Collision – penalise each collision
        new_col = len(self.collision_hist) - self.previous_collisions
        r_collision = -500.0 * new_col
        self.previous_collisions += new_col
        reward_components["collision"] = r_collision

        # reward_components = {}

        # (G) Goal / progress
        dist_goal = self.ego.get_location().distance(self.end_point.location)
        distance_diff = self.last_distance_to_goal - dist_goal
        

        r_goal = 0.0# if dist_goal < 2.0:
        if dist_goal < 2.0:
            r_goal = 200.0
        else:
            # print("self.initial_distance_to_goal", self.initial_distance_to_goal)
            # print("self.last_distance_to_goal", self.last_distance_to_goal)
            # print("dist_goal", dist_goal)
            # print("distance_diff", distance_diff)
            # progress = (self.initial_distance_to_goal - dist_goal) / self.initial_distance_to_goal
            progress = max(0.0, distance_diff) * 60.0  # 5.0 is a scaling factor
            # r_goal = 60.0 * max(0.0, progress)
            r_goal = progress
        reward_components["goal"] = r_goal

        self.last_distance_to_goal = dist_goal
        # -----------------------------
        total_reward = sum(reward_components.values())

        # print("reward_components", reward_components)

        # total_reward = 0.0
        # reward_components = {}

        return total_reward, reward_components

    # --------------------------------------------------------------------------
    # Termination Conditions
    # --------------------------------------------------------------------------
    def get_location_distance(self, location1: Tuple[float, float], location2: Tuple[float, float]) -> float:
        return np.linalg.norm(np.array([location1[0] - location2[0], location1[1] - location2[1]]))

    def get_wpt_dist(self, ego_location):
        if len(self.waypoints) == 0:
            return 0
        else:
            return self.get_location_distance(ego_location, self.waypoints[0])


    def _check_termination(self):

        """
        Returns
        -------
        terminated : bool    # True = task success/failure that should propagate gradients
        truncated  : bool    # True = time-limit or external cut
        info       : dict    # diagnostics
        """
        # --- gather episode facts ---------------------------------------------
        collision     = self.collision_detected
        reached_goal  = self.ego.get_location().distance(self.end_point.location) < 2.0
        time_exceeded = self._time_step >= self._max_time_step

        # low-speed: share the threshold with the reward code
        LOW_SPEED_KMH       = 1.0
        LOW_SPEED_TIMEOUT_S = 10.0

        if self.speed_kmh < LOW_SPEED_KMH:
            if self.low_speed_start_time is None:
                self.low_speed_start_time = time.time()
        else:
            self.low_speed_start_time = None

        stuck_too_long = (
            self.low_speed_start_time is not None and
            (time.time() - self.low_speed_start_time) > LOW_SPEED_TIMEOUT_S
        )

        # --- check if the vehicle has driven past the goal ---------------------
        past_goal = False
        # Ensure start_point is defined (if your environment uses it)
        if self.spawn_index is not None:
            # 2D version for clarity (discard z if you like)
            v_goal = np.array([
                self.end_point.location.x - EGO_SPAWN_POINT[self.spawn_index][0],
                self.end_point.location.y - EGO_SPAWN_POINT[self.spawn_index][1]
            ])
            v_current = np.array([
                self.ego.get_location().x - EGO_SPAWN_POINT[self.spawn_index][0],
                self.ego.get_location().y - EGO_SPAWN_POINT[self.spawn_index][1]
            ])
            dot_goal = np.dot(v_goal, v_goal)          # ||SE||^2
            dot_current = np.dot(v_goal, v_current)    # SE · SC

            # If the projection is larger than the squared distance to the goal,
            # ego is "beyond" the endpoint in terms of that main direction
            if dot_current > dot_goal:
                past_goal = True

        # --- decide outcome ----------------------------------------------------
        info = {}
        terminated = False
        truncated  = False

        # recommended precedence: collision > goal > stuck > time-limit
        if collision:
            terminated = True
            info["collision"] = True

        elif reached_goal:
            terminated = True
            info["goal_reached"] = True

        elif past_goal:
            terminated = True
            info["past_goal"] = True

        elif stuck_too_long:
            terminated = True
            info["not_moving"] = True
            info["stuck_duration"] = time.time() - self.low_speed_start_time

        elif time_exceeded:
            terminated = True          # Gymnasium’s “time-limit”
            info["time_exceeded"] = True
            info["elapsed_steps"] = self._time_step

        return terminated , info

    # --------------------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------------------
    def _clean_actors(self):
        try:
            for s in ("camera_sensor", "colsensor", "lane_sensor"):
                sensor = getattr(self, s, None)
                if sensor:
                    sensor.stop()
            batch = []
            for a in list(self.actors):
                if a and a.is_alive:
                    batch.append(carla.command.DestroyActor(a))
            if self.ego and self.ego.is_alive:
                batch.append(carla.command.DestroyActor(self.ego))
            if batch:
                self.client.apply_batch_sync(batch, True)
            self.actors.clear()
            self.ego = self.nonego = self.camera_sensor = self.colsensor = self.lane_sensor = None
            if getattr(self, "_sync_enabled", False):
                self.world.tick()
        except Exception as e:
            print(f"An error occurred during cleanup: {e}")

        