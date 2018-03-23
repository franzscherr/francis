from setuptools import setup, find_packages

setup(name='francis',
        version='0.1',
        description='Small helper tools and imports for fast prototyping and convenience in machine learning and similar',
        keywords='machine learning tensorflow numpy data science',
        url='https://github.com/franzscherr/francis',
        author='Franz Scherr',
        license='MIT',
        packages=find_packages(),
        install_requires=open('requirements.txt').read().split(),
        include_package_data=True,
        zip_safe=False)

