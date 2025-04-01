import numpy as np
import os
import multiprocessing as mp
from multiprocessing import Array, Pipe
import ctypes
import time

# Constants
NUM_ENVS = 1024
ENVS_PER_WORKER = 4
OBS_DIM = 128
ACTION_DIM = 80

# Load C library
lib_path = os.path.join(os.path.dirname(__file__), "libaudio_rl.cpython-311-darwin.so")
c_lib = ctypes.CDLL(lib_path)

# Define C function signatures
c_lib.init_env.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
c_lib.init_env.restype = None

c_lib.reset_env.argtypes = [ctypes.c_int]
c_lib.reset_env.restype = None

c_lib.step_env.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
c_lib.step_env.restype = ctypes.c_float

c_lib.export_audio_file.argtypes = [ctypes.c_int, ctypes.c_char_p]
c_lib.export_audio_file.restype = ctypes.c_bool

# Worker process function
def worker_process(worker_id, env_ids, shared_mem, parent_pipe, child_pipe):
    """Worker process that handles multiple environments"""
    try:
        # Create numpy views of the shared memory
        np_buf = np.frombuffer(shared_mem.get_obj(), dtype=np.float32)
        
        # Calculate offsets based on PufferLib's approach
        n = len(env_ids)  # Number of environments in this worker
        
        # Split the shared memory according to PufferLib's pattern
        obs_arr = np_buf[:-3*n].reshape(-1, OBS_DIM)  # Observations
        rewards_arr = np_buf[-3*n:-2*n]              # Rewards
        terminals_arr = np_buf[-2*n:-n]              # Terminals
        truncated_arr = np_buf[-n:]                  # Truncated flags
        
        # Initialize environments with observation pointers
        for i, env_id in enumerate(env_ids):
            # Get pointer to observation memory for this environment
            obs_ptr = np.ctypeslib.as_ctypes(obs_arr[i])
            
            # Initialize environment
            c_lib.init_env(env_id, obs_ptr)
            
            # Reset environment
            c_lib.reset_env(env_id)
        
        # Signal that environments are ready
        child_pipe.send(("ready", worker_id))
        
        # Main loop
        running = True
        
        while running:
            # Receive command from main process
            cmd, data = child_pipe.recv()
            
            if cmd == "step":
                # Data contains actions for all environments in this worker
                actions = data
                
                # Process each environment
                for i, env_id in enumerate(env_ids):
                    if i < len(actions):
                        # Convert action to numpy array
                        action_array = np.array(actions[i], dtype=np.float32)
                        
                        # Get pointer to action data
                        action_ptr = np.ctypeslib.as_ctypes(action_array)
                        
                        # Step the environment
                        reward = c_lib.step_env(env_id, action_ptr)
                        
                        # Store reward and done in shared memory
                        rewards_arr[i] = reward
                        terminals_arr[i] = False
                        truncated_arr[i] = False
                
                # Send step completion with empty infos
                child_pipe.send(("step_done", [{} for _ in range(len(env_ids))]))
            
            elif cmd == "reset":
                # Reset all environments
                for i, env_id in enumerate(env_ids):
                    c_lib.reset_env(env_id)
                
                # Send completion signal
                child_pipe.send(("reset_done", worker_id))
            
            elif cmd == "export":
                # Export audio
                env_id, filename = data
                local_idx = env_ids.index(env_id) if env_id in env_ids else -1
                
                if local_idx >= 0:
                    filename_bytes = filename.encode('utf-8')
                    success = c_lib.export_audio_file(env_id, filename_bytes)
                    child_pipe.send(("export_done", success))
                else:
                    child_pipe.send(("export_done", False))
            
            elif cmd == "exit":
                running = False
    
    except Exception as e:
        # Report error
        child_pipe.send(("error", str(e)))
    
    finally:
        # No need to clean up shared memory, parent process handles that
        pass

class ParallelAudioRL:
    def __init__(self, num_envs=NUM_ENVS, envs_per_worker=ENVS_PER_WORKER):
        self.num_envs = num_envs
        self.envs_per_worker = envs_per_worker
        self.num_workers = (num_envs + envs_per_worker - 1) // envs_per_worker
        
        # Create pipes for communication
        self.parent_pipes = []
        self.child_pipes = []
        
        for _ in range(self.num_workers):
            parent_pipe, child_pipe = Pipe()
            self.parent_pipes.append(parent_pipe)
            self.child_pipes.append(child_pipe)
        
        # Create shared memory arrays for each worker
        # Following PufferLib's pattern exactly
        self.shared_mems = []
        for worker_id in range(self.num_workers):
            # Calculate number of environments for this worker
            start_idx = worker_id * envs_per_worker
            end_idx = min(start_idx + envs_per_worker, num_envs)
            n_envs_in_worker = end_idx - start_idx
            
            # Create shared memory with space for:
            # - Observations for each env
            # - Rewards for each env (1 per env)
            # - Terminal flags for each env (1 per env)
            # - Truncated flags for each env (1 per env)
            mem_size = (n_envs_in_worker * OBS_DIM) + (n_envs_in_worker * 3)
            
            # Create the shared memory array
            shared_mem = Array('f', mem_size)
            self.shared_mems.append(shared_mem)
        
        # Start worker processes
        self.processes = []
        self.env_to_worker = {}  # Mapping from env_id to worker_id
        
        for worker_id in range(self.num_workers):
            # Calculate environment IDs for this worker
            start_idx = worker_id * envs_per_worker
            end_idx = min(start_idx + envs_per_worker, num_envs)
            env_ids = list(range(start_idx, end_idx))
            
            # Update mapping
            for env_id in env_ids:
                self.env_to_worker[env_id] = worker_id
            
            # Start process
            p = mp.Process(
                target=worker_process,
                args=(
                    worker_id,
                    env_ids,
                    self.shared_mems[worker_id],
                    self.parent_pipes[worker_id],
                    self.child_pipes[worker_id]
                )
            )
            p.start()
            self.processes.append(p)
        
        # Wait for all workers to be ready
        ready_count = 0
        while ready_count < self.num_workers:
            for pipe in self.parent_pipes:
                if pipe.poll():
                    cmd, worker_id = pipe.recv()
                    if cmd == "ready":
                        ready_count += 1
                    elif cmd == "error":
                        raise RuntimeError(f"Worker {worker_id} failed with error: {worker_id}")
        
        # Create views for easy access to data
        self.obs_views = []
        self.reward_views = []
        self.terminal_views = []
        self.truncated_views = []
        
        for worker_id in range(self.num_workers):
            # Calculate number of environments for this worker
            start_idx = worker_id * envs_per_worker
            end_idx = min(start_idx + envs_per_worker, num_envs)
            n_envs_in_worker = end_idx - start_idx
            
            # Create numpy views
            np_buf = np.frombuffer(self.shared_mems[worker_id].get_obj(), dtype=np.float32)
            
            # Extract views using the same pattern as the worker
            obs_view = np_buf[:-3*n_envs_in_worker].reshape(-1, OBS_DIM)
            rewards_view = np_buf[-3*n_envs_in_worker:-2*n_envs_in_worker]
            terminals_view = np_buf[-2*n_envs_in_worker:-n_envs_in_worker]
            truncated_view = np_buf[-n_envs_in_worker:]
            
            self.obs_views.append(obs_view)
            self.reward_views.append(rewards_view)
            self.terminal_views.append(terminals_view)
            self.truncated_views.append(truncated_view)
    
    def reset(self):
        """Reset all environments"""
        # Send reset command to all workers
        for pipe in self.parent_pipes:
            pipe.send(("reset", None))
        
        # Wait for all resets to complete
        for pipe in self.parent_pipes:
            cmd, _ = pipe.recv()
            assert cmd == "reset_done"
        
        # Gather observations from all workers
        all_obs = np.zeros((self.num_envs, OBS_DIM), dtype=np.float32)
        
        for worker_id in range(self.num_workers):
            # Calculate environment IDs for this worker
            start_idx = worker_id * self.envs_per_worker
            end_idx = min(start_idx + self.envs_per_worker, self.num_envs)
            n_envs_in_worker = end_idx - start_idx
            
            # Copy observations from this worker's shared memory
            all_obs[start_idx:end_idx] = self.obs_views[worker_id][:n_envs_in_worker]
        
        return all_obs
    
    def step(self, actions):
        """Step all environments with given actions"""
        # Prepare actions for each worker
        worker_actions = [[] for _ in range(self.num_workers)]
        
        for env_id, action in enumerate(actions):
            worker_id = self.env_to_worker.get(env_id)
            if worker_id is not None:
                # Store action for the correct worker
                local_env_idx = env_id - (worker_id * self.envs_per_worker)
                
                # Ensure worker_actions[worker_id] has enough slots
                while len(worker_actions[worker_id]) <= local_env_idx:
                    worker_actions[worker_id].append(None)
                
                worker_actions[worker_id][local_env_idx] = action
        
        # Send step command with actions to each worker
        for worker_id, worker_pipe in enumerate(self.parent_pipes):
            worker_pipe.send(("step", worker_actions[worker_id]))
        
        # Wait for all steps to complete and collect infos
        infos = [{} for _ in range(self.num_envs)]
        
        for worker_id, worker_pipe in enumerate(self.parent_pipes):
            cmd, worker_infos = worker_pipe.recv()
            assert cmd == "step_done"
            
            # Calculate environment IDs for this worker
            start_idx = worker_id * self.envs_per_worker
            end_idx = min(start_idx + self.envs_per_worker, self.num_envs)
            
            # Store infos
            for i, info in enumerate(worker_infos):
                if start_idx + i < self.num_envs:
                    infos[start_idx + i] = info
        
        # Gather observations, rewards, and dones from all workers
        all_obs = np.zeros((self.num_envs, OBS_DIM), dtype=np.float32)
        all_rewards = np.zeros(self.num_envs, dtype=np.float32)
        all_dones = np.zeros(self.num_envs, dtype=bool)
        
        for worker_id in range(self.num_workers):
            # Calculate environment IDs for this worker
            start_idx = worker_id * self.envs_per_worker
            end_idx = min(start_idx + self.envs_per_worker, self.num_envs)
            n_envs_in_worker = end_idx - start_idx
            
            # Copy data from this worker's shared memory
            all_obs[start_idx:end_idx] = self.obs_views[worker_id][:n_envs_in_worker]
            all_rewards[start_idx:end_idx] = self.reward_views[worker_id][:n_envs_in_worker]
            all_dones[start_idx:end_idx] = self.terminal_views[worker_id][:n_envs_in_worker]
        
        return all_obs, all_rewards, all_dones, infos
    
    def export_audio(self, env_id, filename):
        """Export audio from environment to file"""
        if env_id < 0 or env_id >= self.num_envs:
            return False
            
        worker_id = self.env_to_worker.get(env_id)
        if worker_id is None:
            return False
            
        # Send export command to worker
        self.parent_pipes[worker_id].send(("export", (env_id, filename)))
        
        # Wait for completion
        cmd, success = self.parent_pipes[worker_id].recv()
        assert cmd == "export_done"
        
        return success
    
    def close(self):
        """Clean up resources"""
        # Send exit command to all workers
        for pipe in self.parent_pipes:
            pipe.send(("exit", None))
        
        # Wait for processes to exit
        for p in self.processes:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()