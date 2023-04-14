#!/usr/bin/python3

import argparse      # Input arguments parsing
import glob          # Path operations
import os            # Path operations
import pandas as pd  # DataFrame etc.
import re            # regex for string matching

########################################################################################################
## String Operations
def find_between(s: str, first: str, last: str) -> str:
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""

def get_space_separated(s: str) -> str:
   return " ".join(s.split())

def substring_after(s, delim):
    return s.partition(delim)[2]

########################################################################################################
## Dictionary/List Operations
def add_to_dict(dct: dict, entry1: str, entry2: int, col1="Procedure Name", col2="Total Time [s]"):
   dct[col1].append(entry1)
   dct[col2].append(entry2)

def init_dict(col1="Procedure Name", col2="Total Time [s]") -> dict:
   return {col1: [], col2: []}

def get_entries_matching(lst: list, regexes: list) -> list:
   return [entry for entry in lst if any(re.search(regex, entry) for regex in regexes)]

def strip_entries(lst: list):
   return [entry.strip() for entry in lst]

def get_index_of_first_entry_containing(lst: list, contents: str) -> int:
   return list(map(lambda line: contents in line, lst)).index(True)

def get_index_of_first_entry_matching(lst: list, regex: str) -> int:
   indices = [i for i, item in enumerate(lst) if re.search(regex, item)]
   return None if len(indices) == 0 else indices[0]

########################################################################################################
## Preconditioner Functions
def get_index_preconditioner_setup_begin(lst: list) -> int:
   return get_index_of_first_entry_containing(lst, "Preconditioner set-up ...")

def get_index_preconditioner_setup_end(lst: list) -> int:
   return get_index_of_first_entry_containing(lst, "Preconditioner set-up done.")

########################################################################################################
## Decomposer/Solver Name/Time
def get_procedure_name_from_first_entry_containing(lst: list, contents: str) -> str:
   first = next(filter(lambda entry: contents in entry, lst), None)
   procedure_name = find_between(first, "'", "'")
   return procedure_name

def get_time_from_entries_containing_interval(lst: list, contents: str, start: int, end: int) -> float:
   lst_context = [line for line in lst[start:end + 1] if contents in line]
   time = 0
   for entry in lst_context:
      time += float(find_between(get_space_separated(entry), ": ", " s"))
   return time

def get_time_from_entries_containing(lst: list, contents: str) -> float:
   return get_time_from_entries_containing_interval(lst, contents, 0, len(lst))

def get_decomposer_name(lst: list) -> str:
   name = get_procedure_name_from_first_entry_containing(lst, "Decomposing using ")
   return name

def get_decomposer_time(lst: list) -> int:
   return get_time_from_entries_containing(lst, "Time of dense LU factorization")

def get_decomposer_name_time(lst: list) -> tuple: # str, float
   return (get_decomposer_name(lst), get_decomposer_time(lst))

def get_solver_name(lst: list) -> str:
   return get_procedure_name_from_first_entry_containing(lst, "Solving using ")

def get_solver_time_interval(lst: list, start: int, end: int) -> int:
   return get_time_from_entries_containing_interval(lst, "Time of dense LU backsubstitution", start, end)

def get_solver_time(lst: list) -> int:
   return get_solver_time_interval(lst, 0, len(lst))

def get_solver_name_time(lst: list) -> tuple: # str, float
   return (get_solver_name(lst), get_solver_time(lst))

def get_initial_solver_time(lst: list) -> tuple: # str, float
   return get_solver_time_interval(lst, get_index_preconditioner_setup_begin(lst), get_index_preconditioner_setup_end(lst))

########################################################################################################
## Krylov Method Functions
def get_index_krylov_begin(lst: list) -> int:
   return get_index_of_first_entry_containing(lst, "Calling Krylov method ...")

def get_index_krylov_end(lst: list) -> int:
   return get_index_of_first_entry_containing(lst, "Krylov method done.")

def get_index_krylov_iteration(lst: list, iteration) -> int:
   return get_index_of_first_entry_matching(lst, f"^KRYLOV_BDDCPCG:[ ]+iteration:[ ]+[{iteration}]$")

def get_krylov_num_pcg_iterations(lst: list) -> int:
   index_pcg_iterations = get_index_of_first_entry_containing(lst, "KRYLOV_BDDCPCG: Number of PCG iterations:")
   return int(get_space_separated(lst[index_pcg_iterations]).split("iterations: ", 1)[1])

def get_krylov_iteration_residual(lst: list, iteration: int, iteration_index: int = None) -> float:
   if iteration_index is None:
      iteration_index = get_index_krylov_iteration(lst, iteration) + 1
   residual_line = lst[iteration_index + 1]
   return float(substring_after(residual_line, "residual:").strip())

def get_krylov_method_info(lst: list) -> tuple: #int, int, int
   return (get_index_krylov_begin(lst), get_index_krylov_end(lst), get_krylov_num_pcg_iterations(lst))

def get_lines_related_to_krylov(lst: list) -> list:
   krylov_start, krylov_end, _ = get_krylov_method_info(lst)
   return lst[krylov_start:krylov_end + 1]

def get_krylov_lu_backsubstitution_iteration_residual_lines(lst: list) -> tuple: # list, int
   krylov_lines = get_lines_related_to_krylov(lst)
   krylov_lines = get_entries_matching(krylov_lines, ["^Time of dense LU backsubstitution:[ ]+[0-9]+[.][0-9]+[ ]s$", \
                                                      "^KRYLOV_BDDCPCG:[ ]+iteration:[ ]+[0-9]+$", \
                                                      "^KRYLOV_BDDCPCG:[ ]+relative residual:[ ]+.*$"])
   return krylov_lines

def get_iteration_indices(lst: list, num_krylov_iterations: int) -> list: # tuple(int, int)
   indices = [(get_index_of_first_entry_matching(lst, "^Time of dense LU backsubstitution:[ ]+[0-9]+[.][0-9]+[ ]s$"), get_index_krylov_iteration(lst, 1))]
   for iteration in range(2, num_krylov_iterations + 1): # "+ 2" skips the line containing "residual"
      indices.append((get_index_krylov_iteration(lst, iteration - 1) + 2, get_index_krylov_iteration(lst, iteration)))
   return indices

def get_krylov_iteration_tuples(lst: list, num_krylov_iterations: int) -> list: # tuple(int, float, list)
   iterations_indices = get_iteration_indices(lst, num_krylov_iterations)
   iterations = []
   for iter, (iter_start, iter_end) in enumerate(iterations_indices, 1): # "- 1" bcs get_solver_time_interval includes the element at iter_end
      iterations.append((iter, get_krylov_iteration_residual(lst, iter, iter_end), get_solver_time_interval(lst, iter_start, iter_end - 1)))
   return iterations

def get_krylov_iterations(lst: list) -> list:
   krylov_lines = get_krylov_lu_backsubstitution_iteration_residual_lines(lst)
   krylov_iterations = get_krylov_iteration_tuples(krylov_lines, get_krylov_num_pcg_iterations(lst))
   return krylov_iterations

########################################################################################################
## File Operations
def get_file_lines_as_list(file_path) -> list:
   file = open(file_path,"r")
   return file.readlines()

def file_to_dataframe(file_path: str, procedure_type: str = "all", individual_backsubstitutions: bool = False) -> pd.DataFrame:
   file_lines = strip_entries(get_file_lines_as_list(file_path))

   benchmark_data = init_dict()

   if procedure_type in ["decomposers", "all"]:
      # Add Decomposer total time
      decomposer_name, decomposer_time = get_decomposer_name_time(file_lines)
      add_to_dict(benchmark_data, f"{decomposer_name}", decomposer_time)

   if procedure_type in ["solvers", "all"]:
      # Add Solver total time
      solver_name, solver_time = get_solver_name_time(file_lines)
      add_to_dict(benchmark_data, f"{solver_name}", solver_time)

      if individual_backsubstitutions:
         # Add initial Solver total time
         solver_initial_time = get_initial_solver_time(file_lines)
         add_to_dict(benchmark_data, f"{solver_name} - initial", solver_initial_time)
         # Add Krylov method backsubstitution times
         krylov_iterations = get_krylov_iterations(file_lines)
         for iter, residual, time in krylov_iterations:
            add_to_dict(benchmark_data, f"{solver_name} - iter: {iter}", time)

   return pd.DataFrame(benchmark_data)

def get_files_in_dir_matching(dir: str, regex_str: str) -> list:
   regex = re.compile(regex_str)
   matched_files = []
   for root, dirs, files in os.walk(dir):
      for file in files:
         if regex.match(file):
            matched_files.append(f"{dir}/{file}")
   return matched_files

def average_files_to_dataframe(files_dir: str, file_regex: str, procedure_type: str = "all") -> pd.DataFrame:
   file_list = get_files_in_dir_matching(files_dir, file_regex)
   if len(file_list) > 0:
      df_list = [file_to_dataframe(file, procedure_type) for file in file_list]
      return pd.concat(df_list).groupby('Procedure Name').mean().reset_index()

def get_num_el_per_sub_edge(file_path: str) -> int:
   poisson_config = get_entries_matching(file_path.split("/"), ["^[0-9]+_[0-9]+_[0-9]+$"])[0]
   return int(poisson_config.split("_")[0])

def get_subdirectories(path: str) -> list:
   return [f.path for f in os.scandir(path) if f.is_dir()]

def get_procedure_type_from_path(path: str) -> str:
   return "decomposers" if "decomposers" in path.split("/") else "solvers"

def log_files_in_dir_tree_to_dataframe(dir: list, input_file_regex: str, procedure_type: str) -> pd.DataFrame:
   procedure_dirs = get_subdirectories(dir)

   average_dfs = []
   for procedure_dir in procedure_dirs:
      for poisson_config_dir in get_subdirectories(procedure_dir):
         df_avg = average_files_to_dataframe(poisson_config_dir, input_file_regex, procedure_type)
         df_avg.insert(0, "Num. el. per sub-edge", get_num_el_per_sub_edge(poisson_config_dir))
         average_dfs.append(df_avg)
   return pd.concat(average_dfs).sort_values(by=["Num. el. per sub-edge", "Procedure Name"]).reset_index(drop=True)

def create_dir_for_output_files(dir: str) -> str:
   import shutil

   if "log-files" not in dir:
      print("-!> Directory 'dir' does not contain the expected 'log-files' substring -> not removing to be safe. Exiting...")
      exit(1)

   if os.path.exists(dir):
      shutil.rmtree(dir)

   print(f"-> Creating directory for parsed results: '{dir}'")
   os.mkdir(dir)
   return dir

########################################################################################################
## DataFrame operations
def compute_speedup_column(df: pd.DataFrame, baseline_procedure: str) -> pd.DataFrame:
   # Get the taken by baseline_procedure for each number of elements per sub-edge
   # Use the number of elements as the index
   baseline_times = df[df["Procedure Name"] == baseline_procedure].set_index("Num. el. per sub-edge")["Total Time [s]"]

   # Compute and the speedup column for each procedure relative to the baseline_procedure
   df[f"Speedup rel. to {baseline_procedure}"] = baseline_times.loc[df["Num. el. per sub-edge"]].values / df["Total Time [s]"].values
   return df

def save_to_csv(df: pd.DataFrame, output_file_path: str):
   df.to_csv(output_file_path, sep=',', encoding='utf-8', index=False)

def save_to_html(df: pd.DataFrame, output_file_path: str):
   text_file = open(output_file_path, "w")
   text_file.write(df.to_html())
   text_file.close()

########################################################################################################
## Matlab functions
def save_dataframe_plot(df: pd.DataFrame, title: str, y_col: str, dir: str, filename: str):
   import matplotlib.pyplot as plt

   fig, ax = plt.subplots(figsize=(10, 6))
   for procedure in df["Procedure Name"].unique():
      sub_df = df[df["Procedure Name"] == procedure]
      ax.plot(sub_df["Num. el. per sub-edge"], sub_df[y_col], linestyle="--")
      ax.scatter(sub_df["Num. el. per sub-edge"], sub_df[y_col], label=procedure)
   ax.set_title(title)
   ax.set_xlabel("Number of elements per sub-edge")
   ax.set_ylabel("Time [s]" )
   ax.legend()
   ax.grid(axis="y")
   ax.set_xticks(range(df["Num. el. per sub-edge"].min(), df["Num. el. per sub-edge"].max() + 1, 5))
   for i in range(df["Num. el. per sub-edge"].min(), df["Num. el. per sub-edge"].max() + 1, 5):
      ax.axvline(x=i, color="grey", alpha=0.5, linestyle="-")

   plt.savefig(f"{dir}/{filename}.pdf")
   ax.set_yscale("log")
   ax.set_ylabel("Time [s] log-scaled" )

   plt.savefig(f"{dir}/{filename}-log.pdf")
   plt.savefig(f"{dir}/{filename}-log.png", dpi=200)

def save_time_plot(df: pd.DataFrame, procedure_type: str, dir: str):
   save_dataframe_plot(df, f"Comparison of {procedure_type} for poisson_on_cube problem", "Total Time [s]", dir, f"{procedure_type}_time")

def save_speedup_plot(df: pd.DataFrame, procedure_type: str, baseline_procedure: str, dir: str):
   save_dataframe_plot(df, f"Comparison of the speedup of {procedure_type} relative to {baseline_procedure}", f"Speedup rel. to {baseline_procedure}", dir, f"{procedure_type}_speedup")

########################################################################################################
## Arguments
def valid_arguments(args) -> bool:
   # Check existence of input file directory
   if not os.path.exists(args.procedures_dir):
      print(f"Directory '{args.procedures_dir}' not found!")
      return False

   # Check validity of regex
   try:
      re.compile(args.input_file_regex)
   except re.error:
      print("Invalid regex pattern!")
      return False

   # Check existence of output file directory
   if not os.path.exists(os.path.dirname(os.path.realpath(args.output_dir))):
      print(f"Directory of output file '{args.output_dir}' not found!")
      return False

   return True

def parse_arguments() -> tuple: # str, str
   # Parse arguments
   parser = argparse.ArgumentParser(description='Parse input file to load file_lines from.')

   # Required arguments
   requiredNamed = parser.add_argument_group('required named arguments')
   requiredNamed.add_argument('-d', '--procedures-dir', help='Directory containing the directories of procedures which hold their benchmark results.', required=True)
   requiredNamed.add_argument('-i', '--input-file-regex', help='Input file regex to match what files in the benchmark dirs to parse.', required=True)
   requiredNamed.add_argument('-o', '--output-dir', help='Output directory where the file containing benchmark results should be placed.', required=True)

   args = parser.parse_args()

   if not valid_arguments(args):
      print("Exiting...")
      parser.parse_args(['-h'])
      exit(1)

   return(args.procedures_dir, args.input_file_regex, args.output_dir)

########################################################################################################
## Main
def main():
   # Parse arguments
   procedures_dir, input_file_regex, output_dir = parse_arguments()
   procedure_type = get_procedure_type_from_path(procedures_dir)

   # Create dataframe with benchmark results parsed from multiple log files
   df = log_files_in_dir_tree_to_dataframe(procedures_dir, input_file_regex, procedure_type)

   # Compute the speedup relative to the existing decomposer/solver in BDDCML - MAGMAdgetrf_gpu/MAGMAdgetrs_gpu
   baseline_procedure = "MAGMAdgetrf_gpu" if procedure_type == "decomposers" else "MAGMAdgetrs_gpu"
   df = compute_speedup_column(df, baseline_procedure)

   # Create a separate log directory for the parsed results
   dir = create_dir_for_output_files(f"{output_dir}/{procedure_type}/parsed_results")

   # Save the dataframe to a csv/html file
   save_to_csv(df, f"{dir}/{procedure_type}_benchmark_results.csv")
   save_to_html(df, f"{dir}/{procedure_type}_benchmark_results.html")

   # Save MATLAB plots
   print(df)
   save_time_plot(df, procedure_type, dir)
   save_speedup_plot(df, procedure_type, baseline_procedure, dir)


if __name__== "__main__" :
	main()