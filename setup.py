"""
Setup configuration for AI DAO Hedge Fund
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-dao-hedge-fund",
    version="1.0.0",
    author="mohin-io",
    author_email="mohinhasin999@gmail.com",
    description="Decentralized Autonomous Hedge Fund powered by Multi-Agent RL and Blockchain DAO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mohin-io/AI-DAO-Hedge-Fund",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-dao-train=simulations.backtest.run_multi_agent_training:main",
        ],
    },
)
