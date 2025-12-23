import sys
if sys.prefix == '/home/canozkan/Capstone_Project/venv':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/canozkan/Capstone_Project/install/system_nodes'
