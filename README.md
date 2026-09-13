# HackMTY 2026

<!-- Demo Video -->

<!-- Summary -->

## Table of contents

## Getting started
### Prerequisites


### Installation

### Local Setup
#### Git settings
In order to properly push, you'll need to run the following git script in order to not accidentally screw up with the rulesets.
```sh
git config core.hooksPath .githooks
```
Upon doing this, you will work on any of the issues placed on the [issues](https://github.com/TEC-Andres/HackMTY-2026/issues). You also need to keep an eye on the [Hack-Monterrey]() project in order to see what's needed and what can be omitted for now.

## Execute & test
Open two terminals, one located at `/hackmty26/scripts` and another one located at `/fastapi`. In the first terminal, run the following command to start the FastAPI server:

```sh
uvicorn app.main:app --port 8000 --reload
```

In the second terminal, run the following command to test the endpoint:
```sh
python .\check_endpoint.py --url http://127.0.0.1:8000/detect/YOUR-MODEL-NAME --audio-dir C:\Users\andre\Downloads\HackMTY-2026\audio\ --n 50
```

If `YOUR-MODEL-NAME` is left in blank, a combination of all the models weighted will be used to detect the audio files. You can also specify a model name to test a specific model.

The available models are:
| Model Name | AUC |
| --- | --- |
| timeDiff | 0.720 |
| resonancia | 0.760 |
| STTLexicalAnalysis | 0.750 |
| NST | 0.803 |

## Documentaion

## Team members

| Name | GitHub | Role |
| --- | --- | --- |
| Ethiel Favila | [@efavilaa](https://github.com/efavilaa) | `HRI, Integration, Backend & Frontend` |
| Isabel Mejia Franco | [@IsaMejiaF](https://github.com/IsaMejiaF ) | `Frontend & Web Design` |
| Catherine | [catherinegd7](https://github.com/catherinegd7) | `HRI, Integration & Backend` |
| Andrés Rodríguez Cantú | [@TEC-Andres](https://github.com/TEC-Andres) | `HRI, Integration & Backend` |

## Credits

## License

