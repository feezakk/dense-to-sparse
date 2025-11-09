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

from typing import Tuple

from pathlib import Path

import os

from collections import deque


EGO_SPAWN_POINT = [[-16.890745162963867, -211.24720764160156, 0.2819424271583557, 0.0, 89.7751235961914, 0.0],
                   [-13.395880699157715, -212.56092834472656, 0.2819424271583557, 0.0, 89.7751235961914, 0.0],
                   [-9.890790939331055, -211.27468872070312,  0.2819424271583557, 0.0, 89.7751235961914, 0.0],
                   [-6.395920276641846, -212.58840942382812, 0.2819424271583557, 0.0, 89.7751235961914, 0.0]]


NON_EGO_SPAWN_POINT = [[-16.712175369262695, -165.74740600585938, 0.2819424271583557, 0.0, 89.7751235961914, 0.0],
                   [-13.221610069274902, -168.1611328125, 0.2819424271583557, 0.0, 89.7751235961914, 0.0], 
                   [-9.712218284606934, -165.77487182617188, 0.2819424271583557, 0.0, 89.7751235961914, 0.0], 
                   [-6.2216644287109375, -168.18861389160156, 0.2819424271583557, 0.0, 89.7751235961914, 0.0]]


# EGO_END_POINT =    [[-16.520898818969727, -117.01853942871094, 0.2819424271583557, 0.0, 89.77516174316406, 0.0],
#                    [-13.030336380004883, -119.4322738647461, 0.2819424271583557, 0.0, 89.77516174316406, 0.0],
#                    [-9.520939826965332, -117.04601287841797, 0.2819424271583557, 0.0, 89.77516174316406, 0.0],
#                    [-6.030386447906494, -119.45975494384766, 0.2819424271583557, 0.0, 89.77516174316406, 0.0]]

EGO_END_POINT =    [[-16.520898818969727, -155.01853942871094, 0.2819424271583557, 0.0, 89.77516174316406, 0.0],
                   [-13.030336380004883, -155.4322738647461, 0.2819424271583557, 0.0, 89.77516174316406, 0.0],
                   [-9.520939826965332, -155.04601287841797, 0.2819424271583557, 0.0, 89.77516174316406, 0.0],
                   [-6.030386447906494, -155.45975494384766, 0.2819424271583557, 0.0, 89.77516174316406, 0.0]]


DISCRETE_ACC = [0.0, 0.3] # discrete value of accelerations
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
      'time_limit': 500, # maximum timesteps per episode
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
class CarlaOvertakeStudentTestEnv(gym.Env):
    def __init__(self, config):
        super().__init__()

        # Connect to a running CARLA instance or create a new one
        self.client = carla.Client("localhost", 3000)
        self.client.set_timeout(100.0)

        self.world = self.client.load_world("Town04")
        self.map = self.world.get_map()
        # assert self.map.name == "Town01"

        print("CARLA environment initialized")
        print("Map name:", self.map.name)

        # remove old vehicles and sensors (in case they survived)
        self.world.tick()
        actor_list = self.world.get_actors()
        for vehicle in actor_list.filter("*vehicle*"):
            print("Warning: removing old vehicle")
            vehicle.destroy()
        for sensor in actor_list.filter("*sensor*"):
            print("Warning: removing old sensor")
            sensor.destroy()

        # Load or get the world
        self.world = self.client.get_world()
        self.map = self.world.get_map()

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

        self.spawn_index = np.random.randint(0, len(NON_EGO_SPAWN_POINT) - 1)

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

    # --------------------------------------------------------------------------------
    # Reset the non-ego vehicle spawning
    # --------------------------------------------------------------------------------

    def reset_other_vehicles(self):
        
        # Clear out old vehicles 
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actors])
        self.world.tick()

        traffic_manager = self.client.get_trafficmanager()  
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        traffic_manager.set_synchronous_mode(True)
        
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprint = blueprint_library.find('vehicle.tesla.model3')


        self.nonego_spawn_point = NON_EGO_SPAWN_POINT[self.spawn_index]

        nonego_transform = carla.Transform(
            carla.Location(*self.nonego_spawn_point[:3]),
            carla.Rotation(*self.nonego_spawn_point[-3:]),
        )

        # First, if self.nonego was previously alive, destroy it properly
        if self.nonego is not None and self.nonego.is_alive:
            print("Destroying old non-ego vehicle.")
            self.nonego.destroy()
            self.nonego = None
            # Let CARLA process the destruction
            self.world.tick()
            time.sleep(0.1)

        max_attempts = 10
        for attempt in range(max_attempts):
            actor = self.world.try_spawn_actor(vehicle_blueprint, nonego_transform)
            if actor is not None:
                self.nonego = actor
                print(f"Non-Ego Vehicle Spawned on attempt {attempt+1}")
                break
            else:
                print(f"Spawn failed (collision) on attempt {attempt+1}. Retrying...")
                self.world.tick()
                time.sleep(0.1)

        if self.nonego is None:
            raise RuntimeError(f"Could not spawn Non-Ego Vehicle after {max_attempts} attempts.")

        if self.nonego is not None:
            self.nonego.set_autopilot(True, traffic_manager.get_port())
            # For example, 70% slower than usual
            traffic_manager.vehicle_percentage_speed_difference(self.nonego, 100)
            self.actors.append(self.nonego)

            print("Non-Ego Vehicle Set")

        self.nonego_spawn_point = [nonego_transform.location.x, nonego_transform.location.y, nonego_transform.location.z]     

    # --------------------------------------------------------------------------
    # Observations
    # --------------------------------------------------------------------------
    def _get_observation(self):

        camera_obs = self.camera_image

        if camera_obs is not None:
            pass
            # for _ in range(4):
            #     self.frame_buffer.append(camera_obs.copy())  # Ensure a separate copy is stored
        else:
            camera_obs = np.zeros((128 , 128 , 3), dtype=np.uint8)
            # for _ in range(4):
            #     self.frame_buffer.append(camera_obs.copy())  # Ensure a separate copy is stored

        # camera_obs = np.concatenate(list(self.frame_buffer), axis=-1).astype(np.float32) / 255.0

        collision_flag = 1 if self.collision_detected else 0
        lane_invasion_flag = 1 if self.lane_invasion_detected else 0

        obs = {
            "image": camera_obs,
            "collision": collision_flag,
            "lane_invasion": lane_invasion_flag
        }
        
        return obs

    # --------------------------------------------------------------------------
    # Gym methods: reset, step, (optional) render, close
    # --------------------------------------------------------------------------
    def reset(self):
        self._clean_actors()
    
        self.spawn_index = np.random.randint(0, len(NON_EGO_SPAWN_POINT) - 1)

        self.ego_transform = carla.Transform(
            carla.Location(x = EGO_SPAWN_POINT[self.spawn_index][0], y = EGO_SPAWN_POINT[self.spawn_index][1], z = EGO_SPAWN_POINT[self.spawn_index][2]),
            carla.Rotation(pitch = EGO_SPAWN_POINT[self.spawn_index][3] , yaw = EGO_SPAWN_POINT[self.spawn_index][4], roll = EGO_SPAWN_POINT[self.spawn_index][5]),
        ) 

        self.end_point = carla.Transform(
            carla.Location(x = EGO_END_POINT[self.spawn_index][0], y = EGO_END_POINT[self.spawn_index][1], z = EGO_END_POINT[self.spawn_index][2]),
            carla.Rotation(pitch = EGO_END_POINT[self.spawn_index][3] , yaw = EGO_END_POINT[self.spawn_index][4], roll = EGO_END_POINT[self.spawn_index][5]),
        ) 
        
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

        self.reset_other_vehicles()
        if self.nonego is not None:
            self.actors.append(self.nonego)

        # Keep track of actors to destroy later
        # self.actors = [self.nonego, self.ego]

        # Initialize the vehicle with default controls
        self.ego.apply_control(carla.VehicleControl(steer=0.0, throttle=0.0, brake=0.0))
        time.sleep(1)  # Allow time for sensors to initialize
        
        self.episode_start = time.time()

        # Attach collision sensor to ego to detect collisions
        self.setup_collision_sensor()

        # Attach lane invasion sensor to ego to detect lane invasions
        self.setup_lane_invasion_sensor()

        # Attach camera sensor to ego
        self.setup_camera()

        # Path planning
        ego_dest = EGO_END_POINT[self.spawn_index]
        dest_location = carla.Location(x=self.nonego_spawn_point[0], y=ego_dest[1], z=ego_dest[2])
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]

        self.exceeding = False
        self.overtake = False
        self.last_ego_y = EGO_SPAWN_POINT[self.spawn_index][1]

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
        

        print("Environment reset")

        # IMPORTANT: clear out the old buffer
        self.episode_buffer = []
        self.timestep = 0

        self.initial_distance_to_goal = self.ego.get_location().distance(self.end_point.location)
        self.last_distance_to_goal = self.ego.get_location().distance(self.end_point.location)


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

        obs_current = self._get_observation()

        curr_path = "data/" + str(self.episode_id) + "/curr"
        curr_path = Path(curr_path)
        next_path = "data/" + str(self.episode_id) + "/next"
        next_path = Path(next_path)

        curr_path.mkdir(parents=True, exist_ok=True)
        next_path.mkdir(parents=True, exist_ok=True)

        # 2. Save the image to disk if it exists
        if obs_current["image"] is not None: 
            # Create a filename like episode_0_step_0.jpg
            img_name = f"episode_{self.episode_id}_step_{self.timestep}"
            img_path = curr_path / img_name
            
            # obs["image"] is a numpy array in BGR or RGB
            # cv2.imwrite(str(img_path), obs_current["image"]) 
            np.save(str(img_path.with_suffix('.npy')), obs_current["image"]) 
            
            # Replace the image in your transition with the *filename* only
            oc = str(img_path)

        # 1. Apply Ego action
        self.apply_control(action)

        # 2. Incrementing time
        self._time_step += 1

        # 3. Tick the world
        self.world.tick()

        if self.nonego is not None and self.nonego.is_alive:
            pass
        else:
            if self.nonego is not None:
                self.nonego.destroy()
            if self.nonego.is_alive:
                self.nonego.destroy()
            self.reset_other_vehicles()

        # 4. Update waypoint

        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]

        #compute speed
        self.velocity = self.ego.get_velocity()
        self.speed_kmh = 3.6 * math.sqrt(self.velocity.x**2 + self.velocity.y**2 + self.velocity.z**2)

        # 5. Compute observation
        obs = self._get_observation()

        # 6) Save the *next* obs image to disk
        if obs["image"] is not None:
            img_name = f"episode_{self.episode_id}_step_{self.timestep}_next"
            img_path = next_path / img_name
            
            # cv2.imwrite(str(img_path), obs["image"]) 
            np.save(str(img_path.with_suffix('.npy')), obs["image"])
            on = str(img_path)
        # 6. Compute reward
        reward, info_dict = self._compute_reward()

        # 4. Check termination
        done, terminal_info = self._check_termination()

        if done == True:
            print("terminal_info", terminal_info)

        info = {**info_dict, **terminal_info}

        # store the transition in the buffer
        transition = {
            "observation": oc,
            "action": action,
            "reward": reward,
            "done": done,
            "next_observation": on,
            "info": info
        }

        self.episode_buffer.append(transition)

        # if episode ended, optionally store or process
        if done:
            # Example 1: keep it in `all_episodes_data`
            # self.all_episodes_data.append(self.episode_buffer)

            # Example 2: or write it to disk
            self._save_episode_to_disk(self.episode_buffer)
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

        # camera_spawn = carla.Transform(carla.Location(x=1.5, z=1.8), carla.Rotation(pitch=0)) 
        camera_spawn = carla.Transform(carla.Location(z=20), carla.Rotation(pitch=-90)) 
        self.camera_sensor = self.world.spawn_actor(self.camera, camera_spawn, attach_to=self.ego)
        self.actors.append(self.camera_sensor)
        self.camera_sensor.listen(lambda data: self.camera_callback(data))

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
        nonego_control = self._get_nonego_vehicle_control()
        self.ego.apply_control(control)
        self.nonego.apply_control(nonego_control)

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

        # if acc > 0:
        #     # throttle = np.clip(acc / 3.0, 0, 1)
        #     throttle = acc
        #     brake = 0.0
        # else:
        #     throttle = 0.0
        #     brake = np.clip(-acc / 3.0, 0, 1)

        # steer in CARLA is left-negative, right-positive,
        # but it can vary depending on your coordinate system.
        # We invert the sign if needed:
        return carla.VehicleControl(throttle=throttle, steer=-steer, brake=brake)

    # --------------------------------------------------------------------------
    # Control: NONEGO
    # --------------------------------------------------------------------------
    def _get_nonego_vehicle_control(self):
        """
        Non-ego vehicle control is designed for the scenario.
        """
        ego_loc = self.ego.get_transform().location
        nonego_loc = self.nonego.get_transform().location

        # Keep constant speed
        if abs(self.nonego.get_velocity().y) < 2:
            acc = 2
        else:
            acc = 0

        dist = math.sqrt((ego_loc.x - nonego_loc.x) ** 2 + (ego_loc.y - nonego_loc.y) ** 2)
        swing_steer = SWING_STEER
        swing_amplitude = SWING_AMPLITUDE
        swing_trigger_dist = SWING_TRIGGER_DIST
        if dist < swing_trigger_dist:
            # Swing when ego vehicle approaching
            if self.nonego_spawn_point[0] + swing_amplitude <= nonego_loc.x:
                self.swing_direction = 1
            if self.nonego_spawn_point[0] - swing_amplitude >= nonego_loc.x:
                self.swing_direction = -1
            steer = swing_steer * self.swing_direction
            self.prev_errors = {
                "last_error": 0.0,
                "integral": 0.0,
            }  # Reset the prev_error
        else:
            # Implement PID controller for lane keeping
            coeffs = PID_COEFFS
            steer, updated_errors = self.pid_controller(self.nonego_spawn_point[0], nonego_loc.x, self.prev_errors, coeffs)
            self.prev_errors.update(updated_errors)

        # Convert acceleration to throttle and brake
        if acc > 0:
            throttle = np.clip(acc / 3, 0, 1)
            brake = 0
        else:
            throttle = 0
            brake = np.clip(-acc / 3, 0, 1)

        return carla.VehicleControl(throttle=float(throttle), steer=float(-steer), brake=float(brake))

    def pid_controller(self, target, current, prev_errors, coeffs):
        """
        Calculate the PID control output to minimize the deviation.

        Args:
        target (float): The target for the PID controller (central line x-coordinate).
        current (float): The current measurement of the process variable (vehicle x-coordinate).
        prev_errors (dict): A dictionary holding the last error and the integral of errors.
        coeffs (tuple): A tuple of PID coefficients (Kp, Ki, Kd).

        Returns:
        float: The control output (steering angle adjustment).
        dict: Updated dictionary with the last error and integral.
        """
        Kp, Ki, Kd = coeffs
        error = current - target
        integral = prev_errors["integral"] + error
        derivative = error - prev_errors["last_error"]

        output = (Kp * error) + (Ki * integral) + (Kd * derivative)

        # Update the errors for the next call
        updated_errors = {"last_error": error, "integral": integral}

        return output, updated_errors   

    # --------------------------------------------------------------------------
    # Destination
    # --------------------------------------------------------------------------

    def is_destination_reached(self):
        return len(self.waypoints) <= 3 

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

        reward_components = {}

        dist_goal = self.ego.get_location().distance(self.end_point.location)

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

        if dot_current > dot_goal:
                r_goal = 200.0
        elif dist_goal < 2.0:
            r_goal = 200.0
        else: 
            r_goal = 0.0
        
        reward_components["goal"] = r_goal
        
        total_reward = sum(reward_components.values())

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

        # Out of lane bounding box logic (if you want a simple check)
        # E.g. if ego’s x is beyond left/right boundary
        ego_loc = self.ego.get_transform().location
            
        out_of_lane = False

        if EGO_SPAWN_POINT[self.spawn_index][0] == -16.890745162963867:
            left_bound = EGO_SPAWN_POINT[self.spawn_index][0] - 2.5
            right_bound = EGO_SPAWN_POINT[self.spawn_index][0] + 4.5
        elif EGO_SPAWN_POINT[self.spawn_index][0] == -13.395880699157715:
            left_bound = EGO_SPAWN_POINT[self.spawn_index][0] - 3.5
            right_bound = EGO_SPAWN_POINT[self.spawn_index][0] + 3.5
        elif EGO_SPAWN_POINT[self.spawn_index][0] == -9.890790939331055:
            left_bound = EGO_SPAWN_POINT[self.spawn_index][0] - 3.5
            right_bound = EGO_SPAWN_POINT[self.spawn_index][0] + 3.5
        elif EGO_SPAWN_POINT[self.spawn_index][0] == -6.395920276641846:
            left_bound = EGO_SPAWN_POINT[self.spawn_index][0] - 3.5
            right_bound = EGO_SPAWN_POINT[self.spawn_index][0] + 2.5

        if ego_loc.x < left_bound or ego_loc.x > right_bound:
            out_of_lane = True

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


        elif out_of_lane:
            terminated = True
            info["out_of_lane"] = True

        # if the distance between the two vehicles is too long, reset the scenario
        ego_x, ego_y = self.get_vehicle_pos(self.ego)

        if self.nonego is not None and self.nonego.is_alive:
            pass
        else:
            self.reset_other_vehicles()

        return terminated, info

    # --------------------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------------------
    def _clean_actors(self):
        """
        Clean up all actors, sensors, and OpenCV windows.
        """
        try:
            # Destroy actors in the actor list
            for actor in self.actors:
                if actor.is_alive:
                    actor.destroy()
                    time.sleep(0.1)  # Brief delay to ensure destruction
            self.actors.clear()  # Clear the actor list

            # Explicitly set vehicle and sensor references to None
            if self.ego:
                if self.ego.is_alive:
                    self.ego.destroy()
                self.ego = None

            if self.nonego:
                if self.nonego.is_alive:
                    self.nonego.destroy()
                self.nonego = None

            if hasattr(self, 'camera_sensor') and self.camera_sensor:
                if self.camera_sensor.is_alive:
                    self.camera_sensor.destroy()
                self.camera_sensor = None

            if hasattr(self, 'colsensor') and self.colsensor:
                if self.colsensor.is_alive:
                    self.colsensor.destroy()
                self.colsensor = None

            if hasattr(self, 'lane_sensor') and self.lane_sensor:
                if self.lane_sensor.is_alive:
                    self.lane_sensor.destroy()
                self.lane_sensor = None

            # Additional cleanup for remaining vehicle actors as a fallback
            remaining_actors = self.world.get_actors().filter('vehicle.*')
            if remaining_actors:
                print(f"Warning: There are still {len(remaining_actors)} vehicle actors in the world.")
                for actor in remaining_actors:
                    if actor.is_alive:
                        actor.destroy()
                        time.sleep(0.1)
                left_remaining_actors = self.world.get_actors().filter('vehicle.*')
                print(f"Warning: There are still {len(left_remaining_actors)} vehicle actors left in the world.")
                for actor in left_remaining_actors:
                        actor.destroy()
                        time.sleep(0.1)

            # Close OpenCV windows safely
            # cv2.destroyAllWindows()
            self.world.tick()  # ✅ Ensures actors are properly removed

        except Exception as e:
            print(f"An error occurred during cleanup: {e}")