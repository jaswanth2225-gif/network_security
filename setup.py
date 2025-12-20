from setuptools import setup, find_packages
from typing import List


def get_requirements() -> List[str]:
    requirements = []

    try:
        with open("requirements.txt", "r") as file:
            for line in file:
                line = line.strip()
                if line and line != "-e .":
                    requirements.append(line)

    except FileNotFoundError:
        print("requirements.txt not found")

    return requirements


setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Jaswanth",
    author_email="jaswanth2225@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)
