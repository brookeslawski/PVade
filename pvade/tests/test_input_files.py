import pytest
import subprocess
import os
import glob

def launch_sim(input_file):
    dt = 0.001 # 0.001
    tf = dt*10 # ten timesteps
    l_char = 0.01

    command = f"mpirun -n 2 python ../../ns_main.py --input_file " + input_file + " --domain.l_char " + str(l_char) + " --solver.dt " + str(dt) + " --solver.t_final " + str(tf)
    print(command)

    try:
        tmp = subprocess.check_call(command.split())
        # os.system(command)
        return 1 # no errors
    except: # if any error
        return 0

# files_list = glob.glob('../../input/*.yaml')
# files_list = ["../../input/flag2d.yaml"]
# files_list = ["../../input/flag2d.yaml","../../input/flag2d.yaml"]
files_list = ["../../input/flag2d.yaml","../../input/2d_cyld.yaml"]
print(files_list)

@pytest.mark.parametrize("input_file", files_list)
def test_launch_with_different_input_files(input_file):
  print("input_file = ",input_file)
  result = launch_sim(input_file)
  print(result)

  assert result == 1
  # not sure the best way to test that this ran - with or without the next line shows "Pass" when there are errors, or just hangs if there is an error
  # assert True # just checks if it runs without errors

# os.system("pwd")
# test_launch_with_different_input_files(files_list)
# test_launch_with_different_input_files("../../input/flag2d.yaml")
# test_launch_with_different_input_files("../../input/2d_cyld.yaml")