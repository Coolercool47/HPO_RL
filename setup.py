import os
from setuptools import setup, find_packages

def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

def parse_requirements(filename):
    """
    Читает файл requirements.txt и возвращает список зависимостей.
    Игнорирует комментарии и пустые строки.
    """
    requirements = []
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('-'):
                    continue
                requirements.append(line)
    return requirements

setup(
    name='HPO_RL',
    version='0.1.0',
    description='Hyperparm optimization using reinforcement learning methods',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    
    url='https://github.com/Coolercool47/HPO_RL.git@refactored_project',
    author='Egor Peters',
    author_email='cool47.cool@yandex.ru',
    
    packages=find_packages(),
    
    install_requires=parse_requirements('requirements.txt'),
    
    python_requires='>=3.7',
)