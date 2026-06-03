# import os
# import sys
# import subprocess
# from datetime import datetime

# # ==================== Configuration ====================
# benchmark = "tpcc"
# database = "epic"
# num_warehouses = 1
# skew_factor = 0.0
# fullread = "true"
# cpu_exec_num_threads = 32
# num_epochs = 1
# num_txns = 100
# split_fields = "true"
# commutative_ops = "false"
# num_records = 10000
# exec_device = "gpu"
# num_repeat = 1

# epic_driver_path = "./build/epic_driver"
# epic_micro_driver_path = "./build/micro_driver"
# output_path = "./epic_profile"

# # ==================== Logging Configuration ====================
# log_file = None
# summary_file = None
# experiment_logs = []

# # ==================== NCU Sections Configuration ====================
# # Using pre-defined sections is more robust than raw metrics.
# # Available sections: Occupancy, MemoryWorkloadAnalysis, WarpState,
# # ComputeWorkloadAnalysis, InstructionStats, LaunchStats, SpeedOfLight, etc.
# NCU_SECTIONS = ["Occupancy"]

# # Base command templates (without the final redirection or ncu specific flags)
# cmd_template = "{} -b {} -d {} -w {} -a {} -r {} -c {} -e {} -s {} -f {} -m {} -n {} -x {}"
# micro_cmd_template = "{} -b {} -d {} -w {} -a {} -r {} -c {} -e {} -s {} -f {} -m {} -n {} -x {} -p {}"

# # NCU command template using sections
# # ncu_cmd_template = "ncu --section {} -c 1 -o {}.ncu-rep -f --target-processes all {}"
# # For multiple sections we will insert repeated --section arguments
# # We'll construct it in run_with_profile

# output_file_template = "output__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__r{}.txt"
# profile_file_template = "profile__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__r{}"

# micro_profile_file_template = "micro__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__p{}__r{}"


# # ==================== Logging Functions ====================
# def init_logging():
#     """Initialize logging files"""
#     global log_file, summary_file
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file = os.path.join(output_path, f"profiling_{timestamp}.log")
#     summary_file = os.path.join(output_path, f"summary_{timestamp}.txt")
    
#     with open(log_file, "w") as f:
#         f.write(f"=== NCU Profiling Log: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
#     return log_file


# def log_message(message, to_console=True):
#     """Write message to log file and optionally to console"""
#     global log_file
    
#     if to_console:
#         print(message)
    
#     if log_file:
#         with open(log_file, "a") as f:
#             f.write(message + "\n")


# def log_experiment(exp_name, profile_files):
#     """Record experiment output files"""
#     experiment_logs.append({
#         "name": exp_name,
#         "profile_files": profile_files,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     })


# print_experiment_count = 0
# def inc_experiment_count():
#     global print_experiment_count
#     print_experiment_count += 1
#     log_message(f"experiment count: {print_experiment_count}")


# def run_with_profile(cmd, profile_name):
#     """
#     Run command with ncu profiling using sections.
    
#     Args:
#         cmd: The command to run
#         profile_name: Base name for profile output file (without extension)
    
#     Returns:
#         Tuple of (stdout, stderr)
#     """
#     profile_path = os.path.join(output_path, profile_name)
#     # Build --section arguments for each section in NCU_SECTIONS
#     section_args = " ".join([f"--section {s}" for s in NCU_SECTIONS])
#     ncu_cmd = f"ncu {section_args} -o {profile_path}.ncu-rep -f -c 1 --target-processes all {cmd}"
    
#     log_message(f"\n[NCU PROFILE] {ncu_cmd}")
#     log_message(f"[Output] {profile_path}.ncu-rep")
    
#     result = subprocess.run(ncu_cmd, capture_output=True, text=True, shell=True)
    
#     # Log command output
#     if result.stdout:
#         log_message(f"\n[STDOUT]\n{result.stdout}")
#     if result.stderr:
#         log_message(f"\n[STDERR]\n{result.stderr}")
    
#     # Generate CSV report from the ncu-rep file for easier analysis
#     csv_cmd = f"ncu --import {profile_path}.ncu-rep --csv > {profile_path}.csv 2>&1"
#     csv_result = subprocess.run(csv_cmd, shell=True, capture_output=True, text=True)
    
#     if csv_result.returncode == 0:
#         log_message(f"[CSV] Generated {profile_path}.csv successfully")
#     else:
#         log_message(f"[CSV] Error generating CSV: {csv_result.stderr}")
    
#     return result.stdout, result.stderr


# # ==================== Experiment Definitions ====================
# def epic_ycsb_experiment():
#     """Profile YCSB workload on EPIC"""
#     database = "epic"
#     benchmark = "ycsbf"
    
#     for split_fields in ["false"]:
#         for skew_factor in [0.99]:
#             for repeat in range(num_repeat):
#                 inc_experiment_count()
#                 cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
#                                           fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
#                                           commutative_ops, num_records, exec_device)
#                 profile_name = profile_file_template.format(benchmark, database, num_warehouses, skew_factor,
#                                                            fullread, cpu_exec_num_threads, num_epochs, num_txns,
#                                                            split_fields, commutative_ops, num_records,
#                                                            exec_device, repeat)
                
#                 stdout, stderr = run_with_profile(cmd, profile_name)
#                 log_experiment(f"epic_ycsb ({benchmark}, skew={skew_factor})", 
#                              [f"{profile_name}.ncu-rep", f"{profile_name}.csv"])


# def epic_tpcc_experiment():
#     """Profile TPCC workload on EPIC"""
#     database = "epic"
#     benchmark = "tpcc"
    
#     for num_warehouses in [64]:
#         for repeat in range(num_repeat):
#             inc_experiment_count()
#             cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
#                                       fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
#                                       commutative_ops, num_records, exec_device)
#             profile_name = profile_file_template.format(benchmark, database, num_warehouses, skew_factor,
#                                                        fullread, cpu_exec_num_threads, num_epochs, num_txns,
#                                                        split_fields, commutative_ops, num_records,
#                                                        exec_device, repeat)
            
#             stdout, stderr = run_with_profile(cmd, profile_name)
#             log_experiment(f"epic_tpcc (warehouses={num_warehouses})", 
#                          [f"{profile_name}.ncu-rep", f"{profile_name}.csv"])


# def epic_tpcc_full_experiment():
#     """Profile full TPCC workload on EPIC"""
#     database = "epic"
#     benchmark = "tpccfull"
    
#     for num_warehouses in [64]:
#         for repeat in range(num_repeat):
#             inc_experiment_count()
#             cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
#                                       fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
#                                       commutative_ops, num_records, exec_device)
#             profile_name = profile_file_template.format(benchmark, database, num_warehouses, skew_factor,
#                                                        fullread, cpu_exec_num_threads, num_epochs, num_txns,
#                                                        split_fields, commutative_ops, num_records,
#                                                        exec_device, repeat)
            
#             stdout, stderr = run_with_profile(cmd, profile_name)
#             log_experiment(f"epic_tpcc_full (warehouses={num_warehouses})", 
#                          [f"{profile_name}.ncu-rep", f"{profile_name}.csv"])


# def epic_microbenchmark():
#     """Profile microbenchmark on EPIC"""
#     database = "epic"
#     benchmark = "micro"
#     skew_factor = 0.8
    
#     for abort_rate in [50]:
#         for repeat in range(num_repeat):
#             inc_experiment_count()
#             cmd = micro_cmd_template.format(epic_micro_driver_path, benchmark, database, num_warehouses, 
#                                            skew_factor, fullread, cpu_exec_num_threads, num_epochs, num_txns, 
#                                            split_fields, commutative_ops, num_records, exec_device, abort_rate)
#             profile_name = micro_profile_file_template.format(
#                 benchmark, database, num_warehouses, skew_factor, fullread, cpu_exec_num_threads, 
#                 num_epochs, num_txns, split_fields, commutative_ops, num_records, exec_device, abort_rate, repeat)
            
#             stdout, stderr = run_with_profile(cmd, profile_name)
#             log_experiment(f"epic_microbenchmark (abort_rate={abort_rate})", 
#                          [f"{profile_name}.ncu-rep", f"{profile_name}.csv"])


# def gacco_ycsb_experiment():
#     """Profile YCSB workload on GACCO"""
#     database = "gacco"
#     num_txns_local = 32768
#     benchmark = "ycsbf"
    
#     for skew_factor in [0.99]:
#         for repeat in range(num_repeat):
#             inc_experiment_count()
#             cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
#                                       fullread, cpu_exec_num_threads, num_epochs, num_txns_local, split_fields,
#                                       commutative_ops, num_records, exec_device)
#             profile_name = profile_file_template.format(benchmark, database, num_warehouses, skew_factor,
#                                                        fullread, cpu_exec_num_threads, num_epochs, num_txns_local,
#                                                        split_fields, commutative_ops, num_records,
#                                                        exec_device, repeat)
            
#             stdout, stderr = run_with_profile(cmd, profile_name)
#             log_experiment(f"gacco_ycsb (skew={skew_factor})", 
#                          [f"{profile_name}.ncu-rep", f"{profile_name}.csv"])


# def gacco_tpcc_experiment():
#     """Profile TPCC workload on GACCO (both new order and payment)"""
#     database = "gacco"
#     num_txns_local = 32768
    
#     for benchmark in ["tpccn", "tpccp"]:
#         for commutative_ops_local in ["true", "false"]:
#             for num_warehouses in [64]:
#                 for repeat in range(num_repeat):
#                     inc_experiment_count()
#                     cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
#                                               fullread, cpu_exec_num_threads, num_epochs, num_txns_local, split_fields,
#                                               commutative_ops_local, num_records, exec_device)
#                     profile_name = profile_file_template.format(benchmark, database, num_warehouses, skew_factor,
#                                                                fullread, cpu_exec_num_threads, num_epochs, num_txns_local,
#                                                                split_fields, commutative_ops_local, num_records,
#                                                                exec_device, repeat)
                    
#                     stdout, stderr = run_with_profile(cmd, profile_name)
#                     log_experiment(f"gacco_tpcc ({benchmark}, commutative_ops={commutative_ops_local})", 
#                                  [f"{profile_name}.ncu-rep", f"{profile_name}.csv"])


# def print_profile_guide():
#     """Print guide for analyzing profiling results"""
#     guide = """
# ================================================================================
#                     NCU PROFILING ANALYSIS GUIDE
# ================================================================================

# Output Files:
#   - *.ncu-rep : Binary profile report (can be opened in NCU GUI)
#   - *.csv     : CSV format metrics for easy analysis

# Key Metrics Collected via Sections:
# ================================================================================

# From "Occupancy" section:
#   - achieved_occupancy
#   - theoretical_occupancy
#   - sm__warps_active.avg.pct_of_peak_sustained_active

# From "MemoryWorkloadAnalysis" section:
#   - gld_throughput, gst_throughput
#   - gld_efficiency, gst_efficiency
#   - l1tex_cache_sector_queries, l1tex_cache_sector_misses
#   - l2_cache_total_transactions

# Additional metrics (depending on NCU version):
#   - sm__inst_executed.sum
#   - smsp__inst_executed.sum
#   - ipc (instructions per cycle)
#   - warp_execution_efficiency
#   - issue_slot_utilization

# How to Analyze CSV Output:
#   1. Open *.csv in Excel/Python for detailed analysis:
#      import pandas as pd
#      df = pd.read_csv('profile__*.csv')
#      print(df.describe())

#   2. NCU GUI Analysis:
#      ncu-ui profile__*.ncu-rep

#   3. Key Metrics to Check:
#      - achieved_occupancy: Aim for >50%
#      - warp_execution_efficiency: Aim for >80%
#      - gld_efficiency: Check memory coalescing (>70%)

# Performance Bottleneck Identification:
#   - Low occupancy → Register/shared memory pressure or limited blocks
#   - Low efficiency → Warp divergence or uncoalesced memory access
#   - Low memory throughput → Uncoalesced access patterns
#   - High cache misses → Poor data locality

# ================================================================================
# """
#     log_message(guide, to_console=True)


# def write_summary_report():
#     """Write summary report with all experiment outputs"""
#     global summary_file
    
#     with open(summary_file, "w") as f:
#         f.write(f"NCU Profiling Summary Report\n")
#         f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#         f.write(f"\nOutput Directory: {output_path}\n")
#         f.write(f"Log File: {log_file}\n")
#         f.write(f"\n{'='*80}\n")
#         f.write(f"EXPERIMENTS COMPLETED: {len(experiment_logs)}\n")
#         f.write(f"{'='*80}\n\n")
        
#         for i, exp in enumerate(experiment_logs, 1):
#             f.write(f"{i}. {exp['name']}\n")
#             f.write(f"   Timestamp: {exp['timestamp']}\n")
#             f.write(f"   Output Files:\n")
#             for profile_file in exp['profile_files']:
#                 f.write(f"     - {profile_file}\n")
#             f.write(f"\n")
        
#         f.write(f"\n{'='*80}\n")
#         f.write(f"NCU SECTIONS USED:\n")
#         f.write(f"{'='*80}\n")
#         f.write(f"Sections: {', '.join(NCU_SECTIONS)}\n")
        
#         f.write(f"\n{'='*80}\n")
#         f.write(f"ANALYSIS INSTRUCTIONS:\n")
#         f.write(f"{'='*80}\n")
#         f.write("""
# 1. CSV Analysis:
#    For quick analysis, open the .csv files in Excel or Python.

# 2. NCU GUI Analysis:
#    ncu-ui profile__*.ncu-rep

# 3. Key Metrics to Check:
#    - achieved_occupancy: Aim for >50%
#    - warp_execution_efficiency: Aim for >80%
#    - gld_efficiency: Check memory coalescing (>70%)
# """)



# if __name__ == "__main__":
#     # Create output directory
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
    
#     # Initialize logging
#     init_logging()
    
#     log_message(f"Profile output directory: {output_path}")
#     log_message(f"Log file: {log_file}")
#     log_message(f"NCU sections: {', '.join(NCU_SECTIONS)}")
#     log_message("")
    
#     # Run profiling experiments
#     log_message("Starting NCU profiling experiments...")
#     log_message("=" * 80)
    
#     epic_ycsb_experiment()
#     epic_tpcc_experiment()
#     epic_tpcc_full_experiment()
#     epic_microbenchmark()
#     gacco_ycsb_experiment()
#     gacco_tpcc_experiment()
    
#     log_message("=" * 80)
#     log_message("Profiling completed!")
#     log_message(f"Results saved to: {output_path}")
#     log_message("")
#     print_profile_guide()
    
#     # Write summary report
#     write_summary_report()
#     log_message(f"\nSummary report saved to: {summary_file}")
#     log_message("\n" + "="*80)
#     log_message("All output files:")
#     log_message(f"  - Main log: {log_file}")
#     log_message(f"  - Summary: {summary_file}")
#     log_message(f"  - Profile outputs: {output_path}/*.ncu-rep")
#     log_message(f"  - CSV metrics: {output_path}/*.csv")
#     log_message("="*80)

import os
import subprocess
from datetime import datetime

# ==================== Configuration ====================
benchmark = "tpcc"
database = "epic"
num_warehouses = 1
skew_factor = 0.0
fullread = "true"
cpu_exec_num_threads = 32
num_epochs = 5
num_txns = 100000
split_fields = "true"
commutative_ops = "false"
num_records = 10000000
exec_device = "gpu"
num_repeat = 1

epic_driver_path = "./build/epic_driver"
epic_micro_driver_path = "./build/micro_driver"
output_path = "./epic_profile"

# ==================== NCU Metrics (stable set) ====================
NCU_METRICS = [
    "gpu__time_duration.sum"  
]

NCU_METRICS_STR = ",".join(NCU_METRICS)

# Command templates
cmd_template = "{} -b {} -d {} -w {} -a {} -r {} -c {} -e {} -s {} -f {} -m {} -n {} -x {}"
micro_cmd_template = "{} -b {} -d {} -w {} -a {} -r {} -c {} -e {} -s {} -f {} -m {} -n {} -x {} -p {}"

profile_file_template = "profile__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__r{}"
micro_profile_file_template = "micro__b{}__d{}__w{}__a{}__r{}__c{}__e{}__s{}__f{}__m{}__n{}__x{}__p{}__r{}"

# ==================== Logging ====================
log_file = None

def init_logging():
    global log_file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_path, f"profiling_{timestamp}.log")
    with open(log_file, "w") as f:
        f.write(f"=== NCU Profiling Log: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

def log_message(msg, to_console=True):
    if to_console:
        print(msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(msg + "\n")

# ==================== Core Profiling Function ====================
def run_with_profile(cmd, profile_name, timeout_sec=300):
    """
    Run command with ncu profiling using stable metrics.
    timeout_sec: kill if profile takes longer than this (default 5 minutes)
    """
    profile_path = os.path.join(output_path, profile_name)
    # Build ncu command as list to avoid shell escaping issues
    ncu_cmd_list = [
        "ncu",
        "--metrics", NCU_METRICS_STR,
        "-o", f"{profile_path}.ncu-rep",
        "-f",
        "--target-processes", "all",
    ] + cmd.split()   # split cmd string into list
    # Use shell=False to prevent extra shell interpretation
    ncu_cmd = " ".join(ncu_cmd_list)  # for logging only
    log_message(f"\n[NCU PROFILE] {ncu_cmd}")
    log_message(f"[Output] {profile_path}.ncu-rep")
    
    try:
        result = subprocess.run(ncu_cmd_list, capture_output=True, text=True, timeout=timeout_sec)
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        log_message(f"[ERROR] Profiling timed out after {timeout_sec} seconds. The command was killed.")
        return "", "Timeout"
    
    if stdout:
        log_message(f"\n[STDOUT]\n{stdout}")
    if stderr:
        log_message(f"\n[STDERR]\n{stderr}")
    
    # Generate CSV report from ncu-rep
    csv_cmd = ["ncu", "--import", f"{profile_path}.ncu-rep", "--csv"]
    try:
        with open(f"{profile_path}.csv", "w") as csv_f:
            subprocess.run(csv_cmd, stdout=csv_f, stderr=subprocess.PIPE, timeout=60)
        log_message(f"[CSV] Generated {profile_path}.csv successfully")
    except Exception as e:
        log_message(f"[CSV] Error generating CSV: {e}")
    
    return stdout, stderr

# ==================== Experiment Functions ====================
# (Keep them same as before, but ensure they use run_with_profile)
# For brevity, I'm including one example; you can reuse your existing experiment functions.

def epic_ycsb_experiment():
    database = "epic"
    benchmark = "ycsbf"
    for split_fields in ["false"]:
        for skew_factor in [0.99]:
            for repeat in range(num_repeat):
                cmd = cmd_template.format(epic_driver_path, benchmark, database, num_warehouses, skew_factor,
                                          fullread, cpu_exec_num_threads, num_epochs, num_txns, split_fields,
                                          commutative_ops, num_records, exec_device)
                profile_name = profile_file_template.format(benchmark, database, num_warehouses, skew_factor,
                                                           fullread, cpu_exec_num_threads, num_epochs, num_txns,
                                                           split_fields, commutative_ops, num_records,
                                                           exec_device, repeat)
                run_with_profile(cmd, profile_name)

# Add other experiment functions (epic_tpcc_experiment, etc.) as in your original code.
# ...

if __name__ == "__main__":
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    init_logging()
    log_message(f"Profile output directory: {output_path}")
    log_message(f"Log file: {log_file}")
    log_message(f"NCU metrics: {len(NCU_METRICS)} metrics")
    
    log_message("Starting NCU profiling experiments...")
    # experiments
    epic_ycsb_experiment()
    
    log_message("Profiling completed.")