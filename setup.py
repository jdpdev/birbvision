from distutils.core import setup

setup(name='birbvision',
      version='0.1.2',
      description='Bird classifier for Birbvision',
      author='Jason DuPertuis',
      author_email='jdpdev@jdpdev.net',
      packages=['birbvision'],
      package_data={'mypkg': ['aiy/lite-model_aiy_vision_classifier_birds_V1_3.tflite', 'aiy/*.txt']}
     )