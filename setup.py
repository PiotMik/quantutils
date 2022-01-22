from setuptools import setup

setup(
        name="quantutils",
        version="0.0.1",
        author="Piotr Mikler",
        author_email="piotr.mikler1997@gmail.com",
        packages=["quantutils"],
        package_dir={"quantutils":"quantutils"},
        url="https://github.com/PiotMik/quantutils/",
        install_requires=["numpy >= 1.8",
                          "scipy >= 0.13",
                          "pandas >= 1.3.5",
                          "dash >= 1.19.0"
                          ]
)
