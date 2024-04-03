# SPDX-FileCopyrightText: Copyright (c) 2024 MBition GmbH.
# SPDX-License-Identifier: MIT
import os
from datetime import datetime
from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    requirements_file = Path("requirements.txt")

    with requirements_file.open() as file:
        return file.readlines()


def get_version() -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return os.environ.get("CI_COMMIT_TAG", default=f"v0.0.{timestamp}")


setup(
    packages=find_packages(
        include=["src*"],
        exclude=["tests*"],
    ),
    version=get_version(),
    install_requires=read_requirements(),
    keywords=["python", "nerdt", "mercedes-benz"],
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Ubuntu :: Linux",
        "Operating System :: Microsoft :: Windows",
    ],
)
