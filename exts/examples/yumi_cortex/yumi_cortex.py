# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#


import carb
import numpy as np
import omni
from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.cortex.cortex_utils import load_behavior_module
from omni.isaac.cortex.cortex_world import Behavior, CortexWorld, LogicalStateMonitor
from omni.isaac.cortex.dfb import DfDiagnosticsMonitor
from omni.isaac.cortex.robot_yumi import CortexYumi, add_yumi_to_stage
from omni.isaac.cortex.tools import SteadyRate
from omni.isaac.examples.cortex.cortex_base import CortexBase

from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.core.prims import GeometryPrim, XFormPrim
import os
import pandas as pd

import omni.graph.core as og

class CubeSpec:
    def __init__(self, name, color):
        self.name = name
        self.color = np.array(color)


class ContextStateMonitor(DfDiagnosticsMonitor):
    """
    State monitor to read the context and pass it to the UI.
    For these behaviors, the context has a `diagnostic_message` that contains the text to be displayed, and each
    behavior implements its own monitor to update that.

    """

    def __init__(self, print_dt, diagnostic_fn=None):
        super().__init__(print_dt=print_dt)
        self.diagnostic_fn = diagnostic_fn

    def print_diagnostics(self, context):
        if self.diagnostic_fn:
            self.diagnostic_fn(context)


class YumiCortex(CortexBase):
    def __init__(self, monitor_fn=None):
        super().__init__()
        self._monitor_fn = monitor_fn
        self.behavior = None
        self.robot = None
        self.context_monitor = ContextStateMonitor(print_dt=0.25, diagnostic_fn=self._on_monitor_update)

    def setup_scene(self):
        world = self.get_world()
        self.robot = world.add_robot(add_yumi_to_stage(name="yumi", prim_path="/World/yumi"))

        # Create a Platform under the robot to keep the targets within the reach of robot
        platform_height = 0.115
        robot_platform = world.scene.add(
                DynamicCuboid(
                    prim_path="/World/platform", # The prim path of the cube in the USD stage
                    name="robot_platform", # The unique name used to retrieve the object from the scene later on
                    position=np.array([0.64, 0, platform_height / 2]), # Using the current stage units which is in meters by default.
                    scale=np.array([1.0, 1.0, platform_height]), # most arguments accept mainly numpy arrays.
                    color=np.array([0.0, 0, 1.0]), # RGB channels, going from 0-1
                ))

        mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation") # changed the extension dir
        fixture_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")  # changed the sub root dir
        
        # Import the microscope glass slide fixture (.usd models) from the 'motion_policy_config' location
        usd_path_slide_fixture = fixture_config_dir + "/fixture_description/Slide_Fixture.usd" # changed the usd file location here
        add_reference_to_stage(usd_path=usd_path_slide_fixture, prim_path="/World")
        new_position_slide_fixture = np.array([0.20, -0.07, platform_height])
        Fixt = world.scene.add(
            GeometryPrim(
                prim_path="/World/Fixture",
                name="Fixture", 
                position=new_position_slide_fixture, 
                scale=np.array([0.001, 0.001, 0.001]),
            )
        )
        # Import the microscope glass slide holder (.usd models) from the 'motion_policy_config' location
        usd_path_slide_holder = fixture_config_dir + "/fixture_description/Slide_Holder.usd" 
        add_reference_to_stage(usd_path=usd_path_slide_holder, prim_path="/World")
        new_position_slide_holder = np.array([0.5, -0.01, platform_height])
        Hold = world.scene.add(
            GeometryPrim(
                prim_path="/World/Holder",
                name="Holder", 
                position=new_position_slide_holder, 
                scale=np.array([0.001, 0.001, 0.001]),
            )
        )

        # Read the Excel file containing arrangement sequence of class slides
        file_path = '/home/sanjay/Desktop/Stacking_logic.xlsx'  # Specify the full file path
        df = pd.read_excel(file_path)
        obs_specs = []
        for index, row in df.iterrows():
            name_cube = row[0]
            color_values = [row[1], row[2], row[3]]
            cube_spec = CubeSpec(name_cube, color_values)
            obs_specs.append(cube_spec)

        height = 0.002
        self.Drop_off_location = df.iloc[0:3, 8].values
        for i, (x, spec) in enumerate(zip(np.linspace(0.3, 0.7, len(obs_specs)), obs_specs)):
            obj = world.scene.add(
                DynamicCuboid(
                    prim_path="/World/Obs/{}".format(spec.name),
                    name=spec.name,
                    scale=np.array([0.075, 0.023, height]), #Size/Scale of the glass slide
                    color=spec.color, #This is completely optional
                    # position=np.array([x, -0.2, width / 2 + 0.068]),
                    position=np.array([new_position_slide_fixture[0] + 0.02275 + (0.122 * i), new_position_slide_fixture[1] - 0.046, new_position_slide_fixture[2] + 0.018 + (height / 2)]),
                    orientation=np.array([1, 0, 0, 1])
                )
            )
            self.robot.register_obstacle(obj)
        world.scene.add_default_ground_plane()

        og.Controller.edit(
            {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("PublishJointState", "omni.isaac.ros_bridge.ROS1PublishJointState"),
                    ("SubscribeJointState", "omni.isaac.ros_bridge.ROS1SubscribeJointState"),
                    ("ArticulationController", "omni.isaac.core_nodes.IsaacArticulationController"),
                    ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                    ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),

                    ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),

                    ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                    ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                    ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                    ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    # Providing path to /panda robot to Articulation Controller node
                    # Providing the robot path is equivalent to setting the targetPrim in Articulation Controller node
                    ("ArticulationController.inputs:usePath", True),
                    ("ArticulationController.inputs:robotPath", "/World/yumi"),

                    ("PublishJointState.inputs:targetPrim", "/World/yumi"),
                    ("PublishJointState.inputs:topicName", "joint_states1"),
                ],
            },
        )

    async def load_behavior(self, behavior):
        world = self.get_world()
        self.behavior = behavior
        self.decider_network = load_behavior_module(self.behavior).make_decider_network(self.robot, self.Drop_off_location)
        self.decider_network.context.add_monitor(self.context_monitor.monitor)
        world.add_decider_network(self.decider_network)

    def clear_behavior(self):
        world = self.get_world()
        world._logical_state_monitors.clear()
        world._behaviors.clear()

    async def setup_post_load(self, soft=False):
        world = self.get_world()
        prim_path = "/World/yumi"
        if not self.robot:
            self.robot = world._robots["yumi"]
        self.decider_network = load_behavior_module(self.behavior).make_decider_network(self.robot, self.Drop_off_location)
        self.decider_network.context.add_monitor(self.context_monitor.monitor)
        world.add_decider_network(self.decider_network)
        await omni.kit.app.get_app().next_update_async()

    def _on_monitor_update(self, context):
        diagnostic = ""
        decision_stack = ""
        if hasattr(context, "diagnostics_message"):
            diagnostic = context.diagnostics_message
        if self.decider_network._decider_state.stack:
            decision_stack = "\n".join(
                [
                    "{0}{1}".format("  " * i, element)
                    for i, element in enumerate(str(i) for i in self.decider_network._decider_state.stack)
                ]
            )

        if self._monitor_fn:
            self._monitor_fn(diagnostic, decision_stack)

    def _on_physics_step(self, step_size):
        world = self.get_world()

        world.step(False, False)

    async def on_event_async(self):
        world = self.get_world()
        await omni.kit.app.get_app().next_update_async()
        world.reset_cortex()
        world.add_physics_callback("sim_step", self._on_physics_step)
        await world.play_async()

    async def setup_pre_reset(self):
        world = self.get_world()
        if world.physics_callback_exists("sim_step"):
            world.remove_physics_callback("sim_step")

    def world_cleanup(self):
        pass
