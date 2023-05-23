# pyproject template

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/legend-exp/dspeed?logo=git)
[![GitHub Workflow Status](https://img.shields.io/github/checks-status/legend-exp/dspeed/main?label=main%20branch&logo=github)](https://github.com/legend-exp/dspeed/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codecov](https://img.shields.io/codecov/c/github/legend-exp/dspeed?logo=codecov)](https://app.codecov.io/gh/legend-exp/dspeed)
![GitHub issues](https://img.shields.io/github/issues/legend-exp/dspeed?logo=github)
![GitHub pull requests](https://img.shields.io/github/issues-pr/legend-exp/dspeed?logo=github)
![License](https://img.shields.io/github/license/legend-exp/dspeed)
[![Read the Docs](https://img.shields.io/readthedocs/dspeed?logo=readthedocs)](https://dspeed.readthedocs.io)

Template for modern Python package GitHub repositories.

## Quick configuration

1. Clone the repository locally
1. Run the interactive configuration script:
   ```console
   $ cd pyproject-template
   $ ./template-config.sh
   ```
1. Rename the repository folder to your new repository name
1. Fill in the missing information in `setup.cfg`
1. Remove any template instruction from this `README.md` (but keep the footer at the end!)
1. Choose a license and save its statement in `LICENSE`
1. Remove the `template-config.sh` file
1. Create a new (empty) GitHub repository matching your `user/repo`
1. Create a new commit with the unstaged changes and git-push to the remote
1. Activate (you can log in with your GitHub credentials):
    * https://pre-commit.ci
    * https://codecov.io
    * https://readthedocs.io (recommended setting: "Advanced settings" > "Build pull requests for this project")
    * GitHub actions (in the repository settings)
1. [Optional] Get a PyPI token and add it as a repository secret on GitHub
   (name it `PYPI_PASSWORD`) to enable publishing the package.

## Quick start

* Install:
  ```console
  $ pip install .
  $ pip install .[test] # get ready to run tests
  $ pip install .[docs] # get ready to build documentation
  $ pip install .[all]  # get all from above
  ```
* Build documentation:
  ```console
  $ cd docs
  $ make        # build docs for the current version
  ```
* Run tests with `pytest`
* Run pre-commit hooks with `pre-commit run --all-files`
* Release a new version:
  ```console
  $ git tag v0.1.0
  $ git checkout -b releases/v0.1 # to apply patches, if needed, later
  $ git push v0.1.0
  ```

## Optional customization

* Customize the python versions / operative systems to test the package against in
  `.github/workflows/main.yml`
* Edit the pre-commit hook configuration in `.pre-commit-config.yaml`. A long
  list of hooks can be found [here](https://pre-commit.com/hooks.html)
* Adapt the Sphinx configuration in `docs/source/conf.py`
* Building wheels with GitHub actions currently assumes pure Python wheels.
  Have a look at [this Scikit-HEP
  documentation](https://scikit-hep.org/developer/gha_wheels) to learn how to
  configure building of binary wheels.

<sub>*This Python package layout is based on [pyproject-template](https://github.com/gipert/pyproject-template).*</sub>
