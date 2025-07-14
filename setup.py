from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="loki-code",
    version="0.1.0",
    author="Loki Code Team",
    author_email="team@lokicode.dev",
    description="A local coding agent using LangChain and local LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/loki-code",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "langchain>=0.3.0",
        "langchain-core>=0.3.0",
        "langchain-community>=0.3.0",
        "langchain-text-splitters>=0.3.0",
        "langsmith>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Entry points will be added as CLI is developed
        ],
    },
)