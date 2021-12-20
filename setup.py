from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='num2cat',
      version='0.1',
      description='cat2num (Categorical to numerical) is a class to deal with categorical variables before being used in ML algorithms.',
      url='https://github.com/rcasal/num2cat.git',
      author='Ramiro Casal',
      author_email='ramiro.casal@mightyhive.com',
      license='MIT',
      packages=['categorical_to_numerical'],
      entry_points={},
      install_requires=[
          'numpy',
          'category_encoders',
          'feature_engine',
          'pandas',
          'sklearn',
          'imblearn',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      zip_safe=False)
