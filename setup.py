from setuptools import setup, find_packages
setup(
    name="UnrealLanding",
    version="0.0.1",
    # geo_utils will NOT function unless extra requires develop is selected
    packages=['airsimcollect'],
    scripts=[],

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['Click', 'geojson', 'shapely>=1.5', 'airsim>=1.2.0', 'numpy>=1.15', 'colorama', 'Pillow', 'numpy-quaternion', 'matplotlib', 'numba', 'pyyaml'],

    entry_points='''
        [console_scripts]
        asc=airsimcollect.scripts.collect:cli
        poi=airsimcollect.scripts.generatepoi:cli
    ''',

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        'airsimcollect': ['*.txt', '*.rst', '*.md', '*.pyx', '*.json'],
    },

    # metadata to display on PyPI
    author="Jeremy Castagno",
    author_email="jdcasta@umich.edu",
    description="Package to SetUp Unreal Rooftop Landing",
    license="MIT",
    keywords="airsim data collect",
    url="https://github.com/JeremyBYU/UnrealRooftopLanding",   # project home page, if any
    project_urls={
        "Bug Tracker": "https://github.com/JeremyBYU/UnrealRooftopLanding/issues",
    }

)