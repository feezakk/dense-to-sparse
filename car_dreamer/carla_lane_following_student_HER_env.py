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

EGO_SPAWN_POINT = [[71.929916, -6.996726, 1.050581, 0.953298, -62.641895, 0.000000],
                   [59.233459, -68.268501, 5.817840, 4.703701, -81.431274, 0.000000],
                   [67.262825, -114.866020, 9.692015, 4.703701, -85.152115, 0.000000],
                   [44.566227, -177.097443, 6.951100, -4.487109, -93.109505, 0.000000]]

EGO_END_POINT =    [[59.233459, -68.268501, 5.817840, 4.703701, -81.431274, 0.000000],
                   [67.262825, -114.866020, 9.692015, 4.703701, -85.152115, 0.000000],
                   [44.566227, -177.097443, 6.951100, -4.487109, -93.109505, 0.000000],
                   [6.476024, -241.105026, 1, 2.601381, -192.499680, 0.000000]]

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
class CarlaLaneFollowingStudentEnv(gym.Env):
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

        try:
            # Remove static parked cars from the map
            self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        except Exception:
            # Older CARLA versions may not have MapLayer; ignore.
            pass

        self.map = self.world.get_map()

        settings = self.world.get_settings()
        if not settings.synchronous_mode or settings.fixed_delta_seconds != 0.1:
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            self.world.apply_settings(settings)
        self._sync_enabled = True

        # self.world.wait_for_tick(5.0)
        

        print("CARLA environment initialized")
        print("Map name:", self.map.name)

        self._hard_world_cleanup()

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

        # self.spawn_index = np.random.randint(0, len(EGO_SPAWN_POINT))

        self._spawn_queue = deque(np.random.permutation(len(EGO_SPAWN_POINT)))
        self.spawn_index = self._spawn_queue.popleft()



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

        self._last_rgb = None

        self.goal_radius = 2.0     # meters
        self.R_goal = 200.0
        self.R_collision = 150.0   # moderate to avoid "always stop"

        self._collision_step = False
        self._lane_invasion_step = False

    def _hard_world_cleanup(self):
        actors = self.world.get_actors()
        victims = []
        for a in actors:
            tid = a.type_id
            if tid.startswith('vehicle.') or tid.startswith('walker.') or tid.startswith('sensor.'):
                victims.append(carla.command.DestroyActor(a))
        if victims:
            self.client.apply_batch_sync(victims, True)
            self.world.tick()

    def _get_teacher_state(self):
        rgb = self._pull_latest_image(timeout=0.5)
        self.camera_image2 = rgb
        self.camera_image = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
        return {
            "image": self.camera_image,
            "collision": 1 if self.collision_detected else 0,
            "lane_invasion": 1 if self.lane_invasion_detected else 0,
        }
    

    def _next_spawn_index(self):
        if not self._spawn_queue:
            self._spawn_queue.extend(np.random.permutation(len(EGO_SPAWN_POINT)))
        return self._spawn_queue.popleft()
    

    def compute_goal_reward(self, ag, dg, info):
        success = np.linalg.norm(ag - dg, axis=-1) < self.goal_radius
        if np.isscalar(success):
            if success: return self.R_goal
            if info.get("collision", False): return -self.R_collision
            return 0.0
        # batched
        r = np.zeros(len(success), np.float32)
        r[success] = self.R_goal
        # if you propagate collision flags per step, set negatives there; else keep 0
        return r


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
    def _pull_latest_image(self, timeout=0.2):
        rgb = None
        if hasattr(self, "_img_q"):
            end = time.time() + timeout
            while True:
                try:
                    # (frame_id, arr)
                    fid, arr = self._img_q.get(timeout=max(0, end - time.time()))
                    rgb = arr
                    # drain to newest without blocking
                    while True:
                        fid, arr = self._img_q.get_nowait()
                        rgb = arr
                except queue.Empty:
                    break
        if rgb is not None:
            self._last_rgb = rgb
            return rgb
        # fallback: last good frame, else keep previous image without refreshing
        return self._last_rgb if self._last_rgb is not None else np.zeros((512,512,3), np.uint8)

    def _get_observation(self):
        rgb = self._pull_latest_image(timeout=0.5)
        self.camera_image2 = rgb
        self.camera_image = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)

        ag = np.array([self.ego.get_location().x, self.ego.get_location().y], np.float32)
        dg = np.array([self.end_point.location.x, self.end_point.location.y], np.float32)
        return {
            "image": self.camera_image,
            "collision": 1 if self.collision_detected else 0,
            "lane_invasion": 1 if self.lane_invasion_detected else 0,
            "achieved_goal": ag,
            "desired_goal": dg,

        }

    # --------------------------------------------------------------------------
    # Gym methods: reset, step, (optional) render, close
    # --------------------------------------------------------------------------
    def reset(self):
        self._hard_world_cleanup()
        self._clean_actors()

        if not self._spawn_queue:
            self._spawn_queue = deque(np.random.permutation(len(EGO_SPAWN_POINT)))
        self.spawn_index = self._spawn_queue.popleft()


        # self.spawn_index = np.random.randint(0, len(EGO_SPAWN_POINT))

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

        self._collision_step = False
        self._lane_invasion_step = False

        

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

        info_dict["collision_step"] = int(self._collision_step)
        self._collision_step = False

        info_dict["lane_invasion_step"] = int(self._lane_invasion_step)
        self._lane_invasion_step = False

        info_dict["off_center_m"] = abs(self.get_signed_lane_offset())


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
            frame = self.camera_image2.copy()
            hud_text = f"Spawn idx: {self.spawn_index}"
            cv2.putText(frame, hud_text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (255, 255, 255), 2, cv2.LINE_AA)


            resized_image = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
              
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

    def get_signed_lane_offset(self) -> float:
        loc = self.ego.get_location()
        wp  = self.world.get_map().get_waypoint(
            loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        center = wp.transform.location
        right  = wp.transform.get_right_vector()
        # signed meters: right positive, left negative
        v = np.array([loc.x - center.x, loc.y - center.y], np.float32)
        r = np.array([right.x, right.y], np.float32)
        return float(np.dot(v, r) / max(1e-6, np.linalg.norm(r)))

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
        goal_space  = spaces.Box(-1e6, 1e6, shape=(2,), dtype=np.float32)

        return spaces.Dict({
            "image": camera_space,
            "collision": collision_space,
            "lane_invasion": lane_invasion_space,
            "achieved_goal": goal_space, 
            "desired_goal": goal_space,
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
        self.camera.set_attribute("sensor_tick", "0.1")  # = fixed_delta_seconds

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
            item = (img.frame, arr)                    
            try:
                q.put_nowait(item)
            except queue.Full:
                try: q.get_nowait()
                except queue.Empty: pass
                q.put_nowait(item)

        self.camera_sensor.listen(_on_image)

        # warm up: wait for first frame so reset() doesn't show black
        for _ in range(3):
            self.world.tick()
            try:
                self._img_q.get(timeout=0.5)
            except queue.Empty:
                pass

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

        self._collision_step = True 

    def lane_invasion_data(self, event):
        self.lane_invasion_hist.append(event)
        self.lane_invasion_detected = True

        self._lane_invasion_step = True

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
        ag = np.array([self.ego.get_location().x, self.ego.get_location().y], np.float32)
        dg = np.array([self.end_point.location.x, self.end_point.location.y], np.float32)

        success = float(np.linalg.norm(ag - dg) < self.goal_radius)
        r = self.R_goal if success else 0.0

        collided = bool(self.collision_detected)
        if collided:
            r -= self.R_collision




        info = {
            "success": success,
            "collision": collided,
        }
        return float(r), info
    
    # Add to class CarlaLaneFollowingStudentEnv
    
    # def _compute_reward(self):

    #     total_reward = 0.0
    #     reward_components = {}



    #     # (A) Goal 
    #     dist_goal = self.ego.get_location().distance(self.end_point.location)
        

    #     r_goal = 0.0
    #     if dist_goal < 5.0:
    #         r_goal = 200.0
        
    #     reward_components["goal"] = r_goal

    #     total_reward = r_goal

    #     collision     = self.collision_detected

    #     if collision > 0:
    #         total_reward -= 300
    #         reward_components["collision"] = -300
    #     else:
    #         reward_components["collision"] = 0

    #     ag = np.array([self.ego.get_location().x, self.ego.get_location().y], np.float32)
    #     dg = np.array([self.end_point.location.x, self.end_point.location.y], np.float32)

    #     total_reward += self.compute_goal_reward(ag[None], dg[None], reward_components)[0]
    #     return total_reward, reward_components

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

        ag = np.array([self.ego.get_location().x, self.ego.get_location().y], np.float32)
        dg = np.array([self.end_point.location.x, self.end_point.location.y], np.float32)
        reached_goal = float(np.linalg.norm(ag - dg) < self.goal_radius)
        # reached_goal  = self.ego.get_location().distance(self.end_point.location) < 2.0
        time_exceeded = self._time_step >= self._max_time_step

        # low-speed: share the threshold with the reward code
        LOW_SPEED_KMH       = 1.0
        LOW_SPEED_TIMEOUT_S = 10.0

        # if self.speed_kmh < LOW_SPEED_KMH:
        spd = float(self.speed_kmh) if self.speed_kmh is not None else LOW_SPEED_KMH + 1.0
        if spd < LOW_SPEED_KMH:
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

        