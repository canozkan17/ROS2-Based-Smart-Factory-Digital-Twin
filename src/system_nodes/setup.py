from setuptools import find_packages, setup

package_name = 'system_nodes'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
        ],
    },
)
