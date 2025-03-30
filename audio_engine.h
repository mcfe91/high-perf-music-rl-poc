// audio_engine.h - Core audio processing functionality
#ifndef AUDIO_ENGINE_H
#define AUDIO_ENGINE_H

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

// Constants
#define NUM_TRACKS 4
#define OBS_DIM 128
#define ACTION_DIM 16
#define SAMPLE_RATE 22050
#define BUFFER_SIZE 44100  // 1 bar at 120 BPM (2 seconds)

// Audio environment structure (all pre-allocated)
typedef struct {
    // Audio buffers
    float track_buffers[NUM_TRACKS][BUFFER_SIZE];
    float master_buffer[BUFFER_SIZE];
    
    // Synth state (per track)
    struct {
        float oscillator_mix;      // 0-1: Sine to saw mix
        float filter_cutoff;       // 0-1: 20Hz to 20kHz
        float filter_resonance;    // 0-1: No resonance to high resonance
        float envelope_attack;     // 0-1: 0ms to 1000ms
        float envelope_release;    // 0-1: 0ms to 2000ms
        float pitch;               // MIDI note (36-84)
        float volume;              // 0-1: Silent to unity gain
        float pan;                 // -1 to 1: Left to right
        
        // Filter state variables (pre-allocated)
        float filter_state[2];
        
        // Sequencer data
        bool steps[16];            // 16-step pattern
    } tracks[NUM_TRACKS];
    
    // Global state
    int current_step;              // Current sequencer step
    float tempo;                   // BPM
    float master_volume;           // 0-1: Master volume
    
    // Global effects
    struct {
        float reverb_size;         // 0-1: Room size
        float reverb_damping;      // 0-1: High frequency damping
        float delay_time;          // 0-1: 0ms to 1000ms
        float delay_feedback;      // 0-1: No feedback to high feedback
        
        // Effect state variables (pre-allocated)
        float reverb_buffer[8192]; // Reverb memory
        float delay_buffer[22050]; // 1 second delay at 22.05kHz
        int delay_index;           // Current position in delay buffer
    } effects;
    
    // Direct pointer to observation memory (for zero-copy)
    float* observation_ptr;
} AudioEnvironment;

// Initialize environment
void init_environment(AudioEnvironment* env) {
    // Clear audio buffers
    memset(env->track_buffers, 0, sizeof(env->track_buffers));
    memset(env->master_buffer, 0, sizeof(env->master_buffer));
    
    // Initialize tracks with default values
    for (int t = 0; t < NUM_TRACKS; t++) {
        env->tracks[t].oscillator_mix = 0.0f;      // Pure sine
        env->tracks[t].filter_cutoff = 0.8f;       // Fairly open filter
        env->tracks[t].filter_resonance = 0.2f;    // Slight resonance
        env->tracks[t].envelope_attack = 0.1f;     // Quick attack
        env->tracks[t].envelope_release = 0.5f;    // Medium release
        env->tracks[t].pitch = 48.0f + t * 12.0f;  // C3, C4, C5, C6
        env->tracks[t].volume = 0.7f;              // 70% volume
        env->tracks[t].pan = (t % 2 == 0) ? -0.3f : 0.3f; // Alternate panning
        
        // Clear filter state
        memset(env->tracks[t].filter_state, 0, sizeof(env->tracks[t].filter_state));
        
        // Set up a basic pattern
        memset(env->tracks[t].steps, 0, sizeof(env->tracks[t].steps));
        env->tracks[t].steps[0] = true;                 // Beat on first step
        if (t == 0) env->tracks[t].steps[8] = true;     // Bass on 3rd beat
        if (t == 1) env->tracks[t].steps[4] = true;     // Snare on 2nd beat
        if (t == 2) env->tracks[t].steps[2] = env->tracks[t].steps[6] = 
                   env->tracks[t].steps[10] = env->tracks[t].steps[14] = true; // Hi-hat
    }
    
    // Initialize global state
    env->current_step = 0;
    env->tempo = 120.0f;
    env->master_volume = 0.8f;
    
    // Initialize effects
    env->effects.reverb_size = 0.3f;
    env->effects.reverb_damping = 0.5f;
    env->effects.delay_time = 0.25f;
    env->effects.delay_feedback = 0.3f;
    
    // Clear effect buffers
    memset(env->effects.reverb_buffer, 0, sizeof(env->effects.reverb_buffer));
    memset(env->effects.delay_buffer, 0, sizeof(env->effects.delay_buffer));
    env->effects.delay_index = 0;
}

// Generate audio for a full bar
void generate_audio(AudioEnvironment* env) {
    // Clear master buffer
    memset(env->master_buffer, 0, sizeof(env->master_buffer));
    
    // Calculate samples per step
    int samples_per_step = BUFFER_SIZE / 16;
    
    // Process each track
    for (int t = 0; t < NUM_TRACKS; t++) {
        // Clear track buffer
        memset(env->track_buffers[t], 0, sizeof(env->track_buffers[t]));
        
        // Process each step in the pattern
        for (int step = 0; step < 16; step++) {
            // Check if this track should play a note at this step
            if (env->tracks[t].steps[step]) {
                // Convert MIDI note to frequency
                float frequency = 440.0f * powf(2.0f, (env->tracks[t].pitch - 69.0f) / 12.0f);
                
                // Calculate start and end samples for this step
                int step_start = step * samples_per_step;
                int step_end = (step + 1) * samples_per_step;
                
                // Calculate oscillator values based on parameters
                float mix = env->tracks[t].oscillator_mix;
                
                // Generate simple waveform (mix between sine and saw)
                for (int i = step_start; i < step_end; i++) {
                    // Calculate phase relative to step start
                    float sample_offset = (float)(i - step_start);
                    float phase = sample_offset / SAMPLE_RATE * frequency;
                    phase -= floorf(phase); // Normalize to 0-1
                    
                    // Generate sine and saw components
                    float sine_val = sinf(phase * 2.0f * M_PI);
                    float saw_val = 2.0f * phase - 1.0f;
                    
                    // Mix between sine and saw
                    env->track_buffers[t][i] = sine_val * (1.0f - mix) + saw_val * mix;
                }
                
                // Apply simple envelope
                float attack = env->tracks[t].envelope_attack * SAMPLE_RATE * 0.1f;
                float release = env->tracks[t].envelope_release * SAMPLE_RATE * 0.5f;
                
                for (int i = step_start; i < step_end; i++) {
                    float envelope = 1.0f;
                    float position = (float)(i - step_start);
                    
                    if (position < attack) {
                        envelope = position / attack;
                    } else if (position > samples_per_step - release) {
                        envelope = (samples_per_step - position) / release;
                    }
                    
                    env->track_buffers[t][i] *= envelope;
                }
                
                // Apply simple lowpass filter
                float cutoff = env->tracks[t].filter_cutoff;
                float f = cutoff * 0.8f;
                
                for (int i = step_start; i < step_end; i++) {
                    env->track_buffers[t][i] = env->track_buffers[t][i] * 0.5f + 
                                              env->tracks[t].filter_state[0] * f + 
                                              env->tracks[t].filter_state[1] * 0.0f;
                                              
                    env->tracks[t].filter_state[1] = env->tracks[t].filter_state[0];
                    env->tracks[t].filter_state[0] = env->track_buffers[t][i];
                }
            }
        }
        
        // Apply track volume and pan
        float volume = env->tracks[t].volume;
        float pan = env->tracks[t].pan;
        
        for (int i = 0; i < BUFFER_SIZE; i++) {
            // Apply pan (simplified - just attenuates one side)
            float left_gain = (pan <= 0) ? 1.0f : 1.0f - pan;
            float right_gain = (pan >= 0) ? 1.0f : 1.0f + pan;
            
            // For this mono implementation, just use the average pan gain
            float pan_gain = (left_gain + right_gain) * 0.5f;
            
            // Apply to master buffer
            env->master_buffer[i] += env->track_buffers[t][i] * volume * pan_gain;
        }
    }
    
    // Apply global effects
    // Simple delay effect
    int delay_samples = (int)(env->effects.delay_time * SAMPLE_RATE * 0.5f);
    float feedback = env->effects.delay_feedback;
    
    // Create a temporary buffer for delay processing
    float effect_buffer[BUFFER_SIZE];
    memcpy(effect_buffer, env->master_buffer, sizeof(effect_buffer));
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        // Read from delay buffer
        int read_idx = (env->effects.delay_index + i - delay_samples + sizeof(env->effects.delay_buffer) / sizeof(float)) % 
                      (sizeof(env->effects.delay_buffer) / sizeof(float));
                      
        float delayed = env->effects.delay_buffer[read_idx];
        
        // Add delay to master
        env->master_buffer[i] += delayed * 0.5f;
        
        // Write to delay buffer
        env->effects.delay_buffer[(env->effects.delay_index + i) % 
                                 (sizeof(env->effects.delay_buffer) / sizeof(float))] = 
            effect_buffer[i] * feedback;
    }
    
    // Update delay index
    env->effects.delay_index = (env->effects.delay_index + BUFFER_SIZE) % 
                              (sizeof(env->effects.delay_buffer) / sizeof(float));
    
    // Apply master volume
    for (int i = 0; i < BUFFER_SIZE; i++) {
        env->master_buffer[i] *= env->master_volume;
        
        // Simple limiter to prevent clipping
        if (env->master_buffer[i] > 1.0f) env->master_buffer[i] = 1.0f;
        if (env->master_buffer[i] < -1.0f) env->master_buffer[i] = -1.0f;
    }
    
    // Update current step (for observation only)
    env->current_step = 0; // We've generated the full bar, so we reset to the beginning
}

// Apply action to environment
void apply_action(AudioEnvironment* env, const float* action) {
    int action_idx = 0;
    
    // First 4 actions: track volumes
    for (int t = 0; t < NUM_TRACKS && action_idx < ACTION_DIM; t++, action_idx++) {
        env->tracks[t].volume += action[action_idx] * 0.1f;
        env->tracks[t].volume = std::min(1.0f, std::max(0.0f, env->tracks[t].volume));
    }
    
    // Next 4 actions: filter cutoffs
    for (int t = 0; t < NUM_TRACKS && action_idx < ACTION_DIM; t++, action_idx++) {
        env->tracks[t].filter_cutoff += action[action_idx] * 0.1f;
        env->tracks[t].filter_cutoff = std::min(1.0f, std::max(0.0f, env->tracks[t].filter_cutoff));
    }
    
    // Next 4 actions: oscillator mix
    for (int t = 0; t < NUM_TRACKS && action_idx < ACTION_DIM; t++, action_idx++) {
        env->tracks[t].oscillator_mix += action[action_idx] * 0.1f;
        env->tracks[t].oscillator_mix = std::min(1.0f, std::max(0.0f, env->tracks[t].oscillator_mix));
    }
    
    // Last 4 actions: Global effects
    if (action_idx < ACTION_DIM) {
        env->effects.reverb_size += action[action_idx] * 0.1f;
        env->effects.reverb_size = std::min(1.0f, std::max(0.0f, env->effects.reverb_size));
        action_idx++;
    }
    
    if (action_idx < ACTION_DIM) {
        env->effects.delay_time += action[action_idx] * 0.1f;
        env->effects.delay_time = std::min(1.0f, std::max(0.0f, env->effects.delay_time));
        action_idx++;
    }
    
    if (action_idx < ACTION_DIM) {
        env->effects.delay_feedback += action[action_idx] * 0.1f;
        env->effects.delay_feedback = std::min(0.95f, std::max(0.0f, env->effects.delay_feedback));
        action_idx++;
    }
    
    if (action_idx < ACTION_DIM) {
        env->master_volume += action[action_idx] * 0.1f;
        env->master_volume = std::min(1.0f, std::max(0.0f, env->master_volume));
    }
}

// Calculate reward based on audio quality
float calculate_reward(AudioEnvironment* env) {
    // Extract basic audio features for reward calculation
    float rms = 0.0f;         // RMS amplitude
    float peak = 0.0f;        // Peak amplitude
    float spectral_centroid = 0.0f;  // Simplified approximation
    float dynamic_range = 0.0f;
    
    // Use a subset of the buffer for reward calculation (same as original)
    const int sample_window = 512;
    
    // Calculate RMS and peak
    for (int i = 0; i < sample_window; i++) {
        float sample = env->master_buffer[i];
        rms += sample * sample;
        if (fabs(sample) > peak) peak = fabs(sample);
    }
    rms = sqrtf(rms / sample_window);
    
    // Approximate spectral centroid using zero-crossing rate (very simplified)
    int zero_crossings = 0;
    for (int i = 1; i < sample_window; i++) {
        if ((env->master_buffer[i] >= 0 && env->master_buffer[i-1] < 0) ||
            (env->master_buffer[i] < 0 && env->master_buffer[i-1] >= 0)) {
            zero_crossings++;
        }
    }
    spectral_centroid = (float)zero_crossings / sample_window;
    
    // Approximate dynamic range
    float sum_of_diffs = 0.0f;
    for (int i = 1; i < sample_window; i++) {
        sum_of_diffs += fabs(env->master_buffer[i] - env->master_buffer[i-1]);
    }
    dynamic_range = sum_of_diffs / sample_window;
    
    // Calculate reward components
    float reward = 0.0f;
    
    // Reward for appropriate loudness (not too quiet, not clipping)
    if (rms > 0.1f && peak < 0.95f) {
        reward += 2.0f * rms;
    } else if (peak > 1.0f) {
        reward -= 2.0f; // Penalty for clipping
    } else if (rms < 0.05f) {
        reward -= 1.0f; // Penalty for too quiet
    }
    
    // Reward for spectral balance
    if (spectral_centroid > 0.05f && spectral_centroid < 0.3f) {
        reward += 1.0f;
    }
    
    // Reward for dynamics
    reward += dynamic_range * 5.0f;
    
    // Bonus for track separation
    float track_separation = 0.0f;
    for (int t = 0; t < NUM_TRACKS; t++) {
        if (env->tracks[t].pan < -0.2f || env->tracks[t].pan > 0.2f) {
            track_separation += 0.25f;
        }
    }
    reward += track_separation;
    
    return reward;
}

// Write observations to shared memory
void write_observations(AudioEnvironment* env) {
    if (env->observation_ptr) {
        int obs_idx = 0;
        
        // Write track parameters (4 tracks * 8 parameters = 32 values)
        for (int t = 0; t < NUM_TRACKS && obs_idx < OBS_DIM; t++) {
            env->observation_ptr[obs_idx++] = env->tracks[t].oscillator_mix;
            if (obs_idx >= OBS_DIM) break;
            
            env->observation_ptr[obs_idx++] = env->tracks[t].filter_cutoff;
            if (obs_idx >= OBS_DIM) break;
            
            env->observation_ptr[obs_idx++] = env->tracks[t].filter_resonance;
            if (obs_idx >= OBS_DIM) break;
            
            env->observation_ptr[obs_idx++] = env->tracks[t].envelope_attack;
            if (obs_idx >= OBS_DIM) break;
            
            env->observation_ptr[obs_idx++] = env->tracks[t].envelope_release;
            if (obs_idx >= OBS_DIM) break;
            
            env->observation_ptr[obs_idx++] = (env->tracks[t].pitch - 36.0f) / 48.0f;  // Normalize to 0-1
            if (obs_idx >= OBS_DIM) break;
            
            env->observation_ptr[obs_idx++] = env->tracks[t].volume;
            if (obs_idx >= OBS_DIM) break;
            
            env->observation_ptr[obs_idx++] = (env->tracks[t].pan + 1.0f) * 0.5f;  // Convert -1,1 to 0,1
            if (obs_idx >= OBS_DIM) break;
        }
        
        // Write sequencer state (4 tracks * 16 steps = 64 values)
        for (int t = 0; t < NUM_TRACKS && obs_idx < OBS_DIM; t++) {
            for (int s = 0; s < 16 && obs_idx < OBS_DIM; s++) {
                env->observation_ptr[obs_idx++] = env->tracks[t].steps[s] ? 1.0f : 0.0f;
            }
        }
        
        // Write global parameters
        if (obs_idx < OBS_DIM) env->observation_ptr[obs_idx++] = (float)env->current_step / 16.0f;
        if (obs_idx < OBS_DIM) env->observation_ptr[obs_idx++] = (env->tempo - 60.0f) / 120.0f;
        if (obs_idx < OBS_DIM) env->observation_ptr[obs_idx++] = env->master_volume;
        
        // Write effect parameters
        if (obs_idx < OBS_DIM) env->observation_ptr[obs_idx++] = env->effects.reverb_size;
        if (obs_idx < OBS_DIM) env->observation_ptr[obs_idx++] = env->effects.reverb_damping;
        if (obs_idx < OBS_DIM) env->observation_ptr[obs_idx++] = env->effects.delay_time;
        if (obs_idx < OBS_DIM) env->observation_ptr[obs_idx++] = env->effects.delay_feedback;
        
        // Write audio features
        // Calculate basic features from the first 512 samples
        float rms = 0.0f;
        float peak = 0.0f;
        const int sample_window = 512;
        
        for (int i = 0; i < sample_window; i++) {
            rms += env->master_buffer[i] * env->master_buffer[i];
            if (fabs(env->master_buffer[i]) > peak) peak = fabs(env->master_buffer[i]);
        }
        rms = sqrtf(rms / sample_window);
        
        if (obs_idx < OBS_DIM) env->observation_ptr[obs_idx++] = rms;
        if (obs_idx < OBS_DIM) env->observation_ptr[obs_idx++] = peak;
        
        // Fill remaining observation space with zeros
        while (obs_idx < OBS_DIM) {
            env->observation_ptr[obs_idx++] = 0.0f;
        }
    }
}

// Reset environment
void reset_environment(AudioEnvironment* env) {
    init_environment(env);
    generate_audio(env);
    write_observations(env);
}

// Step environment with action
float step_environment(AudioEnvironment* env, const float* action) {
    // Apply action
    apply_action(env, action);
    
    // Generate audio
    generate_audio(env);
    
    // Calculate reward
    float reward = calculate_reward(env);
    
    // Write observations
    write_observations(env);
    
    return reward;
}

// Export audio to WAV file
bool export_audio(AudioEnvironment* env, const char* filename) {
    // Write WAV file
    FILE* file = fopen(filename, "wb");
    if (!file) {
        return false;
    }
    
    // WAV file header
    struct {
        // RIFF header
        char riff_id[4] = {'R', 'I', 'F', 'F'};
        uint32_t riff_size;
        char wave_id[4] = {'W', 'A', 'V', 'E'};
        
        // Format chunk
        char fmt_id[4] = {'f', 'm', 't', ' '};
        uint32_t fmt_size = 16;
        uint16_t audio_format = 1;  // PCM
        uint16_t num_channels = 1;  // Mono
        uint32_t sample_rate = SAMPLE_RATE;
        uint32_t byte_rate = SAMPLE_RATE * 2;  // 16-bit mono
        uint16_t block_align = 2;
        uint16_t bits_per_sample = 16;
        
        // Data chunk
        char data_id[4] = {'d', 'a', 't', 'a'};
        uint32_t data_size;
    } header;
    
    // Fill in sizes
    header.data_size = BUFFER_SIZE * 2;  // 16-bit samples
    header.riff_size = 36 + header.data_size;
    
    // Write header
    fwrite(&header, sizeof(header), 1, file);
    
    // Convert float samples to 16-bit PCM and write
    int16_t* samples = (int16_t*)malloc(BUFFER_SIZE * sizeof(int16_t));
    if (!samples) {
        fclose(file);
        return false;
    }
    
    for (int i = 0; i < BUFFER_SIZE; i++) {
        // Convert -1.0 to 1.0 float to -32768 to 32767 int16_t
        float sample = env->master_buffer[i];
        // Apply a bit of headroom to avoid clipping
        sample *= 0.9f;
        // Convert to int16_t with clipping
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        samples[i] = (int16_t)(sample * 32767.0f);
    }
    
    // Write sample data
    fwrite(samples, sizeof(int16_t), BUFFER_SIZE, file);
    
    // Clean up
    fclose(file);
    free(samples);
    
    return true;
}

#endif // AUDIO_ENGINE_H