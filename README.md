# ARBOX Project: Refusal in Language Models Is Mediated by a Single Direction

**Content warning**: This repository contains prompts that is offensive, harmful, or otherwise inappropriate in nature.

ABROx project done by Chamod, Nik, Parul!

North poles:
1. base = uncensored + refusal steering vector
    1. if true, how often does refusal steering vector need to be applied
    2. if true, how cosine similarity of refusal steering vectors between models

## 23/1/26

Moved our files into different folders

`exploration/` : chamod's files from project used at the start of project
`nik/` : nik's files
`extension/` : nik's files

We found the setup to be a little inconsistent. We recommend running:

```
conda create -n refusal_env python=3.9.10

pip install -r requirements.txt
pip install -r post_fail_requirements.txt
```