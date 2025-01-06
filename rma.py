import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import time
from scipy.ndimage import gaussian_filter

class MujocoSimulation:
    def __init__(self):
        self.timestep = 0.0025
        self.control_freq = 100
        self.max_steps = 1000
        
        print("\nInitializing simulation...")
        
        # Generate terrain first to avoid pause during visualization
        self._generate_terrain_data()
        
        # Create model and data
        self.model = mujoco.MjModel.from_xml_path("scene.xml")
        self.data = mujoco.MjData(self.model)
        
        # Apply pre-generated terrain
        self.model.hfield_data[:] = self.terrain_data
        
        # Print initial positions
        print(f"Robot initial height: {self.data.qpos[2]:.3f}")
        print(f"Terrain height range: {self.terrain_data.min():.3f} to {self.terrain_data.max():.3f}")
        
        self.paused = False
        self.camera_config = {
            'distance': 3.0,
            'azimuth': 90.0,
            'elevation': -15.0,
        }
    
    def _generate_terrain_data(self, nrow=100, ncol=100):
        def fractal_noise(size, octaves, lacunarity, gain):
            noise = np.zeros(size)
            frequency = 1.0
            amplitude = 1.0
            
            for _ in range(octaves):
                phase = np.random.randint(0, 1000)
                x = np.linspace(0, frequency, size[0])
                y = np.linspace(0, frequency, size[1])
                X, Y = np.meshgrid(x, y, indexing='ij')
                
                Z = np.random.uniform(-1, 1, size)
                noise += amplitude * Z
                
                frequency *= lacunarity
                amplitude *= gain
            
            return noise
        
        # Generate terrain using fractal noise with original parameters
        terrain = fractal_noise((nrow, ncol), octaves=2, lacunarity=2.0, gain=0.25)
        
        # Normalize and scale
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        terrain *= 0.15  # Lower height scaling from original 0.27
        
        self.terrain_data = terrain.flatten()
        
    def key_callback(self, key, scancode, action, mods):
        if action != mujoco.viewer.ACTION_PRESS:
            return
        if key == mujoco.viewer.KEY_SPACE:
            self.paused = not self.paused
        elif key == mujoco.viewer.KEY_R:
            self.reset()
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()
        
    def step(self, action):
        if not self.paused:

            self.data.ctrl[:] = action
            
            for _ in range(int(1.0 / (self.control_freq * self.timestep))):

                # Print state before step
                height = self.data.qpos[2]
                velocity = self.data.qvel[2]
                
                mujoco.mj_step(self.model, self.data)
                
                # Print every 100 steps
            
            obs = self._get_obs()
            done = self._check_termination()
            
            return obs, self._get_reward(), done, {}
        return self._get_obs(), 0.0, False, {}
    
    def _check_termination(self):
        # Get root body position
        height = self.data.qpos[2]
        
        # For rotation, we'll use the rotation matrix instead of euler angles
        # Get rotation matrix of the root body
        root_id = 1  # Usually 1 for the root body after world body
        rot = self.data.xmat[root_id].reshape(3, 3)
        
        # Get roll and pitch from rotation matrix
        # roll = atan2(R32, R33)
        # pitch = atan2(-R31, sqrt(R32^2 + R33^2))
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1]**2 + rot[2, 2]**2))
        
        return (height < 0.28 or 
                abs(roll) > 0.4 or 
                abs(pitch) > 0.2)
    
    def _get_obs(self):
        return np.concatenate([
            self.data.qpos,
            self.data.qvel,
            self.data.ctrl
        ])
    
    def _get_reward(self):
        return 0.0

    def run_visualization(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            viewer.key_callback = self.key_callback
            
            viewer._cam.distance = self.camera_config['distance']
            viewer._cam.azimuth = self.camera_config['azimuth']
            viewer._cam.elevation = self.camera_config['elevation']
            
            print("\nControls:")
            print("  Space: Pause/Resume simulation")
            print("  R: Reset simulation")
            print("  ESC: Exit")
            
            while viewer.is_running():
                step_start = time.time()
                
                action = np.zeros(self.model.nu)
                self.step(action)
                
                viewer.sync()
                
                time_until_next_step = self.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    sim = MujocoSimulation()
    sim.run_visualization()