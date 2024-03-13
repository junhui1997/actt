from dm_control import suite
from dm_control import viewer
import numpy as np

# env = suite.load(domain_name="humanoid", task_name="stand")
# action_spec = env.action_spec()
#
# # Define a uniform random policy.
# def random_policy(time_step):   # Unused.
#   return np.random.uniform(low=action_spec.minimum,
#                            high=action_spec.maximum,
#                            size=action_spec.shape)
#
# # Launch the viewer application.
# viewer.launch(env, policy=random_policy)


from dm_control import mujoco
#将上述代码存在相同目录下，命名为dm_render.py
from assets.dm_render import DMviewer
# physics = mujoco.Physics.from_xml_path('assets/bimanual_viperx_ee_transfer_cube.xml') #注意替换文件路径
#physics = mujoco.Physics.from_xml_path('/media/u/新加卷/all_code/robosuite/robosuite/models/assets/robots/panda/robot.xml') #感觉好像还行
physics = mujoco.Physics.from_xml_path('assets/new_panda/single_cube.xml') #感觉好像还行

viewer = DMviewer(physics)
while physics.time() < 1000:
  physics.step()
  viewer.render()