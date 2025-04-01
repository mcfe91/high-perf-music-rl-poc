// audio_engine_export.cpp - C interface for Python binding
#include "audio_engine.h"

// Global environment instances (pre-allocated)
static AudioEnvironment environments[1024];

// C interface for Python binding
extern "C" {
    // Initialize environment with observation pointer
    void init_env(int env_id, float* obs_ptr) {
        if (env_id >= 0 && env_id < 1024) {
            AudioEnvironment* env = &environments[env_id];
            init_environment(env);
            env->observation_ptr = obs_ptr;
        }
    }
    
    // Reset environment
    void reset_env(int env_id) {
        if (env_id >= 0 && env_id < 1024) {
            reset_environment(&environments[env_id]);
        }
    }
    
    // Step environment with action
    float step_env(int env_id, const float* action) {
        if (env_id >= 0 && env_id < 1024) {
            return step_environment(&environments[env_id], action);
        }
        return 0.0f;
    }
    
    // Export audio to file
    bool export_audio_file(int env_id, const char* filename) {
        if (env_id >= 0 && env_id < 1024) {
            return export_audio(&environments[env_id], filename);
        }
        return false;
    }
}