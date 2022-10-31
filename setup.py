from setuptools import setup

with open("requirements.txt") as f:
    require = [x.strip() for x in f.readlines() if not x.startswith("git+")]

setup(
    name="inv_vae",
    version="0.1",
    packages=["inv_vae"],
)