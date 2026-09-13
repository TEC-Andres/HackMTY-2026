# HackMTY 2026

<!-- Demo Video -->

<!-- Summary -->

## Table of contents

## Getting started
### Prerequisites
- [Python 3.11](https://www.python.org/downloads/release/python-3110/)
- [Git](https://git-scm.com/downloads)
- [huggingface-cli](https://huggingface.co/docs/huggingface_hub/quick-start)

### Installation

You'll need to clone the repository in order to get the project running, which can be done by running the following command in your terminal:
```sh
git clone --recurse-submodules https://github.com/TEC-Andres/HackMTY-2026.git 
``` 

### Dependencies
From here, you will need to install any and all dependencies from the project, you can do this manually by entering on `/fastapi` and running the following command:
```sh
pip install -r requirements.txt
```

### Dataset
In order to retrieve the dataset, please enter [hackmty26 releases](https://github.com/alturio/hackmty26/releases/tag/v1.0) and download the `altur-challenge-audio.zip` file. Then, extract the contents into root, preferably in a folder called `audio`. The structure should look like this:
```
HackMTY-2026/
├── audio/
│   ├── call_0a9c546208d1.wav
│   ├── call_0ab7d2a0c0f5.wav
│   ├── ...
│   └── call_fff9e3c7f0d1.wav
├── .../
└── README.md
```

### Environment Variables
In order to run the project, you will need to create a `.env` file on root directory and add the following variables. In order to make this easier, you can copy the `.env.example` file and rename it to `.env`.
```
cp .env.example .env
```

### Local Setup
#### Git settings
In order to properly push, you'll need to run the following git script in order to not accidentally screw up with the rulesets.
```sh
git config core.hooksPath .githooks
```
Upon doing this, you will work on any of the issues placed on the [issues](https://github.com/TEC-Andres/HackMTY-2026/issues). You also need to keep an eye on the [Hack-Monterrey](https://github.com/TEC-Andres/Hack-Monterrey) project in order to see what's needed and what can be omitted for now.

## Execute & test
Open two terminals, one located at `/hackmty26/scripts` and another one located at `/fastapi`. In the first terminal, run the following command to start the FastAPI server:

```sh
# /fastapi/
uvicorn app.main:app --port 8000 --reload
```

In the second terminal, run the following command to test the endpoint:
```sh
# /hackmty26/scripts/
python .\check_endpoint.py --url http://127.0.0.1:8000/detect/YOUR-MODEL-NAME --audio-dir ~\HackMTY-2026\audio\
```

If `YOUR-MODEL-NAME` is left in blank, a combination of all the models weighted will be used to detect the audio files. You can also specify a model name to test a specific model.

## Justification of the models
The models used in this project follow a very particular pattern, that is, these are mere regressional models, not actually a prediction model. The reason for this decision is precisely because of the nature of the problem. We were offered a small dataset (353 audio files); training a model in this dataset would just overfit the model making it useless for any other audio file. In the end, we used three different statistical models to analyze the audio files and then combine them into a single model that would give us the best results.

### Time distribution `/detect/timeDiff`

**What does it measure?**
It measures the difference of $t''_i$ for user-0 (the synthetic caller) and compares it against a Gaussian regression. In practice, this means looking at how the response latency of the synthetic user evolves across the conversation and modeling the distribution of those latencies rather than the raw timestamps.

**Why is it done?**
During exploration we found a pattern: the synthetic user takes slightly longer to answer than a human caller would, but above all, the interval between those delays is extremely constant. In other words, $t''_i \approx 0$ for a large portion of the interaction, revealing a machine-like regularity in the response timing. 

<table align="center">
  <tr>
    <td align="center" width="60%">
      <img src="assets/png/3d-gaussian-dispersion-synthetic-vs-huma.png" width="100%">
      <br>
      <sub><b>Figure 1:</b> Synthetic vs. Human 3D Gaussian dispersion comparison.</sub>
    </td>
    <td align="center" width="60%">
      <img src="assets/png/gaussianConfidence.png" width="100%">
      <br>
      <sub><b>Figure 2:</b> Gaussian confidence score distribution analysis.</sub>
    </td>
  </tr>
</table>

Using this observation, `_playingGround` built a Gaussian dispersion model that separates the synthetic (bot) data from the human (bonafide) data as cleanly as possible, using the shape and spread of the latency distribution as the discriminating signal. With this approach we reached a **VAL AUC of 0.829** on experimental trials. 


| Model Name | Accuracy |
| --- | --- |
| timeDiff | 0.775 |
| resonancia | 0.789 |
| NST | 0.831 |
| acousticEnsemble | 0.930 |

> [!NOTE]
> Based upon running the endpoint with `--n 71` and `--audio-dir ~\HackMTY-2026\audio\` on the `check_endpoint.py` script.

## Team members

| Name | GitHub | Role |
| --- | --- | --- |
| Ethiel Favila | [@efavilaa](https://github.com/efavilaa) | `HRI, Integration, Backend & Frontend` |
| Isabel Mejia Franco | [@IsaMejiaF](https://github.com/IsaMejiaF ) | `Frontend & Web Design` |
| Catherine | [catherinegd7](https://github.com/catherinegd7) | `HRI, Integration & Backend` |
| Andrés Rodríguez Cantú | [@TEC-Andres](https://github.com/TEC-Andres) | `HRI, Integration & Backend` |

## Credits

## License

