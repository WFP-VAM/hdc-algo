Conda packaging for hdc-algo
============================

To install in conda

```bash
conda activate myenv
conda install -c wfp-ram hdc-algo 
```

Or when defining environment

```yaml
name: myenv
channels:
  - conda-forge
  - wfp-ram
dependencies:
  - python=3.10
  - hdc-algo
```

Or add it to channel list

```bash
conda config --append channels wfp-ram
```
