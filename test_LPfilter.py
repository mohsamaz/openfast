import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openfast_toolbox import FASTOutputFile

# File paths
output_path = "\Run.out"
debug_path =  "\Run.RO.dbg"

# Read OpenFAST output file
output = FASTOutputFile(output_path)
debug = FASTOutputFile(debug_path)


# Convert to DataFrames
output_df = output.toDataFrame()
debug_df = debug.toDataFrame()

# Extract columns
time_output = output_df['Time_[s]']
noisy_signal = output_df['Wind1VelX_[m/s]']
we_vw_debug = debug_df['WE_Vw_[m/s]']


def lp_filter(input_signal, dt, corner_freq, prev_input, prev_output):
    """Simple first-order low-pass filter for a single signal."""
    # Filter coefficients using bilinear transform
    a1 = 2 + corner_freq * dt
    a0 = corner_freq * dt - 2
    b1 = corner_freq * dt
    b0 = corner_freq * dt
    
    # Compute filtered output
    output = (-a0 * prev_output + b1 * input_signal + b0 * prev_input) / a1
    
    return output, input_signal, output

def main():
    # Parameters
    dt = np.mean(np.diff(time_output))
    
    cornering_frequency_Hz = 1
    corner_freq = 2 * np.pi * cornering_frequency_Hz   #1 Hz corner frequency (6.2832 rad/s)
    
    # Initialize filter
    filtered_signal = np.zeros(len(time_output))
    prev_input = prev_output = noisy_signal[0]  # Initialize with first signal value
    
    # Apply filter
    for i in range(len(time_output)):
        filtered_signal[i], prev_input, prev_output = lp_filter(
            noisy_signal[i], dt, corner_freq, prev_input, prev_output
        )
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_output, noisy_signal, label='Wind1VelX', alpha=0.5)
    plt.plot(time_output, filtered_signal, label=f'Filtered Signal ({cornering_frequency_Hz} Hz)', linewidth=2)
    plt.plot(time_output, we_vw_debug, label='WE_Vw', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Wind Speed (m/s)')
    plt.title('Low-Pass Filter Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
