from setuptools import setup, find_packages

setup(
    name='m_yolo',  # Name of your package
    version='0.1',
    packages=find_packages(),  # Automatically finds `m_yolo` and submodules
    install_requires=[],       # List dependencies, e.g., ['numpy', 'opencv-python']
    include_package_data=True,
    description='A YOLO-based medical image segmentation package',
    author='Your Name',
    author_email='supa@di.ku.dk',
    url='https://github.com/your-repo/m_yolo',  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)
