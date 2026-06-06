from setuptools import setup, find_packages

package_name = 'final_proj'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    package_data={
        'final_proj': ['data/*.yaml', 'data/*.pgm', 'data/*.json'],
    },
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sreshth',
    maintainer_email='sreshthtiwari13@gmail.com',
    description='LLM-assisted uncertainty-aware navigation for autonomous delivery robots',
    license='MIT',
    entry_points={
        'console_scripts': [
            'nav_node = final_proj.nodes.nav_node:main',
            'orchestrator = final_proj.nodes.orchestrator:main',
        ],
    },
)