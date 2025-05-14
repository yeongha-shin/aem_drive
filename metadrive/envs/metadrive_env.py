import copy
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from typing import Union

import numpy as np

from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.utils import clip, Config

import pyttsx3
import threading
import queue
import time
import logitech_steering_wheel as lsw

ENABLE_AUDIO = False
ENABLE_VISUAL = False
ENABLE_HAPTIC = False

# ENABLE_AUDIO = True
# ENABLE_VISUAL = True
# ENABLE_HAPTIC = True

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

tts_queue = queue.Queue()

def tts_worker_loop():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_queue.task_done()

# Start the TTS thread
tts_thread = threading.Thread(target=tts_worker_loop, daemon=True)
tts_thread.start()

def speak(text):
    tts_queue.put(text)

def get_condition_label():
    if ENABLE_AUDIO and not ENABLE_VISUAL and not ENABLE_HAPTIC:
        return "Audio Only"
    elif ENABLE_VISUAL and not ENABLE_AUDIO and not ENABLE_HAPTIC:
        return "Visual Only"
    elif ENABLE_HAPTIC and not ENABLE_AUDIO and not ENABLE_VISUAL:
        return "Haptic Only"
    elif not ENABLE_AUDIO and not ENABLE_VISUAL and not ENABLE_HAPTIC:
        return "No Feedback"
    else:
        return "Mixed Feedback"


METADRIVE_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    num_scenarios=1,

    # ===== PG Map Config =====
    map=None,  # int or string: an easy way to fill map_config
    block_dist_config=PGBlockDistConfig,

    random_lane_width=False,
    random_lane_num=False,
    map_config = {
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: 4,
        BaseMap.LANE_WIDTH: 6,
        BaseMap.LANE_NUM: 1,
        "exit_length": 50,
        "start_position": [0, 0],
    },
    store_map=True,

    # ===== Traffic =====
    traffic_density=0.0,                 # No other cars
    need_inverse_traffic=False,          # No opposite-direction lanes
    traffic_mode=TrafficMode.Trigger,   # Mode doesn't matter here since there's no traffic
    random_traffic=False,               # Keep traffic fixed (but it's empty anyway)
    static_traffic_object=False,        # No static objects
    accident_prob=0.0,                  # No random accidents or obstacles
    traffic_vehicle_config=dict(        # Traffic vehicle settings (not used, but here for completeness)
        show_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=False,
        show_lidar=False,
        show_lane_line_detector=False,
        show_side_detector=False,
    ),


    # ===== Object =====
    #accident_prob=0.0,  # accident may happen on each block with this probability, except multi-exits block
    #static_traffic_object=False,  # object won't react to any collisions

    # ===== Others =====
    use_AI_protector=False,
    save_level=0.5,
    horizon=1000,

    # ===== Agent =====
    random_spawn_lane_index=True,
    vehicle_config=dict(navigation_module=NodeNetworkNavigation),
    agent_configs={
        DEFAULT_AGENT: dict(
            use_special_color=True,
            spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
        )
    },

    # ===== Reward Scheme =====
    # See: https://github.com/metadriverse/metadrive/issues/283
    success_reward=10.0,
    out_of_road_penalty=5.0,
    crash_vehicle_penalty=5.0,
    crash_object_penalty=5.0,
    crash_sidewalk_penalty=0.0,
    driving_reward=1.0,
    speed_reward=0.1,
    use_lateral_reward=False,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
    out_of_road_done=True,
    on_continuous_line_done=True,
    on_broken_line_done=False,
    crash_vehicle_done=True,
    crash_object_done=True,
    crash_human_done=True,
)


class MetaDriveEnv(BaseEnv):
    @classmethod
    def default_config(cls) -> Config:
        config = super(MetaDriveEnv, cls).default_config()
        config.update(METADRIVE_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self, config: Union[dict, None] = None):
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        super(MetaDriveEnv, self).__init__(config)

        # scenario setting
        self.start_seed = self.start_index = self.config["start_seed"]
        self.env_num = self.num_scenarios

        # Add a flag to track out-of-road state
        self._is_currently_out_of_road = False
        self._is_currently_crash_vehicle = False
        self._is_currently_crash_object = False

        self._last_out_of_road_audio_time = 0
        self._last_crash_vehicle_audio_time = 0


    def _post_process_config(self, config):
        config = super(MetaDriveEnv, self)._post_process_config(config)
        if not config["norm_pixel"]:
            self.logger.warning(
                "You have set norm_pixel = False, which means the observation will be uint8 values in [0, 255]. "
                "Please make sure you have parsed them later before feeding them to network!"
            )

        config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=self.default_config_copy
        )
        config["vehicle_config"]["norm_pixel"] = config["norm_pixel"]
        config["vehicle_config"]["random_agent_model"] = config["random_agent_model"]
        target_v_config = copy.deepcopy(config["vehicle_config"])
        if not config["is_multi_agent"]:
            target_v_config.update(config["agent_configs"][DEFAULT_AGENT])
            config["agent_configs"][DEFAULT_AGENT] = target_v_config
        config["agent_configs"][DEFAULT_AGENT]["max_speed_km_h"] = 60
        return config

    def done_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        done = False
        max_step = self.config["horizon"] is not None and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        done_info = {
            TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
            TerminationState.CRASH_OBJECT: vehicle.crash_object,
            TerminationState.CRASH_BUILDING: vehicle.crash_building,
            TerminationState.CRASH_HUMAN: vehicle.crash_human,
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),
            TerminationState.SUCCESS: self._is_arrive_destination(vehicle),
            TerminationState.MAX_STEP: max_step,
            TerminationState.ENV_SEED: self.current_seed,
            # TerminationState.CURRENT_BLOCK: self.agent.navigation.current_road.block_ID(),
            # crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False,
        }

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING] or done_info[TerminationState.CRASH_SIDEWALK]
            or done_info[TerminationState.CRASH_HUMAN]
        )

        # determine env return
        if done_info[TerminationState.SUCCESS]:
            done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: arrive_dest.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.OUT_OF_ROAD] and self.config["out_of_road_done"]:
            done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: out_of_road.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: crash vehicle ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: crash object ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_BUILDING]:
            done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: crash building ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: crash human".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.MAX_STEP]:
            # single agent horizon has the same meaning as max_step_per_agent
            if self.config["truncate_as_terminate"]:
                done = True
            self.logger.debug(
                "Episode ended! Scenario Index: {} Reason: max step ".format(self.current_seed),
                extra={"log_once": True}
            )
        return done, done_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0

        
        self._last_out_of_road_force_time = getattr(self, "_last_out_of_road_force_time", 0)

        # Check if vehicle is out of road
        out_of_road = self._is_out_of_road(vehicle)
        
        now = time.time()

        if not hasattr(self, "_last_out_of_road_audio_time"):
            self._last_out_of_road_audio_time = 0

        if out_of_road:
            if not self._is_currently_out_of_road:
                self.agent.off_road_count += 1
                step_info["cost"] = self.config["out_of_road_cost"]
                self._is_currently_out_of_road = True

            if ENABLE_AUDIO and (now - self._last_out_of_road_audio_time > 0.5):
                speak("Out of road")
                self._last_out_of_road_audio_time = now

            if ENABLE_VISUAL:
                self._add_out_of_road_visual_alert(vehicle)
        else:
            self._is_currently_out_of_road = False

        #
        # if vehicle.crash_vehicle:
        #     step_info["cost"] = self.config["crash_vehicle_cost"]
        #     speak("Crash with vehicle")
        #
        #
        # elif vehicle.crash_object:
        #     step_info["cost"] = self.config["crash_object_cost"]
        #     speak("Crash with object")

        if not hasattr(self, "_last_crash_vehicle_audio_time"):
            self._last_crash_vehicle_audio_time = 0

        crash_vehicle = vehicle.crash_vehicle

        if crash_vehicle:
            if not self._is_currently_crash_vehicle:
                self.agent.crash_vehicle_count += 1
                step_info["cost"] = self.config["crash_vehicle_cost"]
                self._is_currently_crash_vehicle = True

                if ENABLE_HAPTIC:
                    try:
                        if lsw.is_connected(0):
                            try:
                                lsw.play_frontal_collision_force(0, 30)
                            except:
                                pass
                    except:
                        pass

                if ENABLE_VISUAL:
                    self._add_crash_visual_alert()

            if ENABLE_AUDIO and (time.time() - self._last_crash_vehicle_audio_time > 0.5):
                speak("Crash with vehicle")
                self._last_crash_vehicle_audio_time = time.time()
        else:
            self._is_currently_crash_vehicle = False



        try:
            lateral_pos = vehicle.lane.local_coordinates(vehicle.position)[1]
        except:
            lateral_pos = 0
        
        if ENABLE_HAPTIC:
            if out_of_road:
                force = 25 if lateral_pos > 0 else -25
            else:
                force = 0
            if lsw.is_connected(0):
                try:
                    lsw.play_constant_force(0, force)
                except:
                    pass





        # === Crash with object ===
        crash_object = vehicle.crash_object
        if crash_object and not self._is_currently_crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
            if ENABLE_AUDIO:
                speak("Crash with object")
        self._is_currently_crash_object = crash_object

        return step_info['cost'], step_info

    @staticmethod
    def _is_arrive_destination(vehicle):
        """
        Args:
            vehicle: The BaseVehicle instance.

        Returns:
            flag: Whether this vehicle arrives its destination.
        """
        long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
        flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
            vehicle.navigation.get_current_lane_width() / 2 >= lat >=
            (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
        )
        return flag

    def _is_out_of_road(self, vehicle):
        # A specified function to determine whether this vehicle should be done.
        # return vehicle.on_yellow_continuous_line or (not vehicle.on_lane) or vehicle.crash_sidewalk
        ret = not vehicle.on_lane
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        elif self.config["on_continuous_line_done"]:
            ret = ret or vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or vehicle.crash_sidewalk
        if self.config["on_broken_line_done"]:
            ret = ret or vehicle.on_broken_line
        return ret

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        elif vehicle.crash_sidewalk:
            reward = -self.config["crash_sidewalk_penalty"]
        step_info["route_completion"] = vehicle.navigation.route_completion

        return reward, step_info

    def setup_engine(self):
        super(MetaDriveEnv, self).setup_engine()
        from metadrive.manager.traffic_manager import PGTrafficManager
        from metadrive.manager.pg_map_manager import PGMapManager
        from metadrive.manager.object_manager import TrafficObjectManager
        self.engine.register_manager("map_manager", PGMapManager())
        self.engine.register_manager("traffic_manager", PGTrafficManager())
        # if abs(self.config["accident_prob"] - 0) > 1e-2:
            #self.engine.register_manager("object_manager", TrafficObjectManager())

        try:
            import logitech_steering_wheel as lsw
            lsw.initialize_with_window(True, int(self.engine.win.get_window_handle().get_int_handle()))
        except Exception as e:
            print(f"[WARN] Failed to init Logitech wheel: {e}")

    def _add_out_of_road_visual_alert(self, vehicle):
        """Add a red light visual indicator when vehicle goes out of road"""
        if not hasattr(self, "_out_of_road_alert_node"):
            # Create the alert only once
            from panda3d.core import NodePath, TextNode
            from direct.gui.OnscreenText import OnscreenText
            
            # Create a red overlay for the screen
            self._out_of_road_alert_node = NodePath("OutOfRoadAlert")
            self._out_of_road_alert_node.reparentTo(self.engine.aspect2d)
            
            # Create a semi-transparent red rectangle covering the screen
            from direct.gui.DirectFrame import DirectFrame
            self._out_of_road_frame = DirectFrame(
                frameColor=(1, 0, 0, 0.3),  # Red with 30% opacity
                frameSize=(-1, 1, -1, 1),
                parent=self._out_of_road_alert_node
            )
            
            # Add text warning
            self._out_of_road_text = OnscreenText(
                text="OUT OF ROAD!",
                style=1,
                fg=(1, 1, 1, 1),  # White text
                pos=(0, 0),
                scale=0.15,
                parent=self._out_of_road_alert_node
            )
            
            # Hide initially
            self._out_of_road_alert_node.hide()
            
            # Schedule removal after a short time
            self.engine.taskMgr.doMethodLater(
                0.5,  # Display for 0.5 seconds
                self._remove_out_of_road_visual_alert,
                "remove_out_of_road_alert"
            )
        else:
            # If already created, just show it and reschedule removal
            self._out_of_road_alert_node.show()
            self.engine.taskMgr.remove("remove_out_of_road_alert")
            self.engine.taskMgr.doMethodLater(
                0.5,
                self._remove_out_of_road_visual_alert,
                "remove_out_of_road_alert"
            )

    def _remove_out_of_road_visual_alert(self, task):
        """Remove the visual alert after a delay"""
        if hasattr(self, "_out_of_road_alert_node"):
            self._out_of_road_alert_node.hide()
        return task.done
    
    def _add_crash_visual_alert(self):
        from direct.gui.OnscreenText import OnscreenText
        if not hasattr(self, "_crash_text"):
            self._crash_text = OnscreenText(
                text="CRASH!",
                fg=(1, 0, 0, 1),
                pos=(0, 0.8),
                scale=0.15,
                mayChange=True
            )
            self._crash_text.hide()
        self._crash_text.show()
        self.engine.taskMgr.doMethodLater(
            0.5, lambda task: (self._crash_text.hide(), task.done)[1],
            "hide_crash_text"
        )



if __name__ == '__main__':

    def _act(env, action):
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward)
        assert isinstance(info, dict)

    env = MetaDriveEnv()
    try:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()
        try:
            if lsw.is_connected(0):
                try:
                    lsw.stop_constant_force(0)
                    lsw.shutdown()
                except:
                    pass
        except:
            pass

__all__ = ["MetaDriveEnv", "get_condition_label", "speak"]