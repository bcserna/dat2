from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='dialog_act_tagger',
    version='17.10.04',
    description='Dialog act tagger for chat messages',
    url='https://bitbucket.3amlabs.net/projects/INTEAM/repos/dialogact_tagger/',
    author='Balazs Cserna',
    author_email='balazs.cserna@logmein.com',
    packages=find_packages(exclude=('data', 'dialog_act_tagger.egg-info', 'models')),
    zip_safe=False,
    install_requires=[
        'tqdm==4.15.0',
        'pandas==0.20.3',
        'textacy==0.3.4',
        'spacy==1.8.2',
        'plac==0.9.6',
        'numpy',
        'scikit-learn==0.19.0',
        'boto3',
        'botocore'
    ],
    include_package_data=True,
)
