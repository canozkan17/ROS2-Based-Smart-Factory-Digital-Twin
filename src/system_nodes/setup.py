from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'system_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages = find_packages(exclude=['test']) + [
                                                    'system_nodes.prediction_handler'
                                                ],  
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('system_nodes/launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='can_ozkan',
    maintainer_email='can.ozkan.de@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hydraulic_press_sensor = system_nodes.Machine_Hydraulic_Press_Node:main',
            'process_pump_sensor = system_nodes.Machine_Process_Pump_Node:main',
            'job_scheduler = system_nodes.Job_Scheduler_Node:main',
            'temp_gui = system_nodes.temp_GUI_Node:main',
            'predictor = system_nodes.Predictor_Node:main',
        ],
    },
)
