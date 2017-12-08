from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='dat2',
    version='17.12.08',
    description='Dialog act tagger for chat messages',
    url='https://github.com/bcserna/dat2',
    author='Balázs Cserna',
    author_email='cserna.balazs@gmail.com',
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
