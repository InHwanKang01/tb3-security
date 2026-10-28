from setuptools import find_packages, setup

package_name = 'tb3_security'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/attack_scenario.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kih',
    maintainer_email='kih@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'security_node = tb3_security.security_node:main',
            'stamp_node = tb3_security.stamp_node:main',
            'abnormal_velocity_node = tb3_security.abnormal_velocity_node:main',
            'flood_attack_node = tb3_security.flood_attack_node:main',
            'delay_attack_node = tb3_security.delay_attack_node:main',
            'dashboard = tb3_security.dashboard:main',
            'data_collector_node = tb3_security.data_collector_node:main',
            'train_model = tb3_security.train_model:main',
            'attack_manager_node = tb3_security.attack_manager_node:main',
            'evaluate = tb3_security.evaluate:main',
            'web_sim = tb3_security.web_sim:main',
            'stealth_eval = tb3_security.stealth_eval:main',
        ],
    },
)
