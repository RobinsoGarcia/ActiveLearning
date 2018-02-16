from setuptools import setup,find_packages
from setuptools.command.install import install

class adjustments(install):
    def run(self):
        install.run(self)
        import portfolio_optimizer
        import sys
        import os
        #path_port = os.path.abspath(os.path.dirname(sys.argv[0]))
        path_port = 'usr/local/lib/python3.6/dist-packages/learning*'
        os.system('chmod -R 777 /' + path_port)
        os.system('chmod -R 777 /usr/local/bin/learning_main.py')
        print('done adjustments')


setup(name='learning',
      version='0',
      description='Pool based active learning class (active) that can be used under several conditions',
      url='https://github.com/RobinsoGarcia/PoolBased---Active-Learning',
      author='Robinson Garcia',
      author_email='rlsg.mec@hotmail.com',
      license='MIT',
      include_package_data=True,
      entry_points={'console_scripts':['learning_main = learning.__main__:main']},
      packages=['learning'],
      package_data={'':['*.pdf'],'learning.UCI_datasets':['*.csv']},
      zip_safe=False,
      install_requires=[
      'numpy>=1.14.0',
      'pandas>-0.22.0'
      'matplotlib>=2.1.2',
      'scikit-image>=0.13.0',
      'scikit-learn>=0.19.1',
      ])
