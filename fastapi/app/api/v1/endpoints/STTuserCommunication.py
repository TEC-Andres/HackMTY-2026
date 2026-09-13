"""Reserved for the issue #20 STT-over-gRPC demonstration channel.

``POST /detect/timeDiff`` intentionally does not depend on this module: the detection
path only needs VAD turn segmentation (see ``app.services.turns``), not full
speech-to-text. This file is where the optional gRPC client that talks to
``colab/hri/microservices/stt/faster-whisper-streaming.py`` belongs if we want
to exercise the full framework communication channel (see docs/detect.md).
"""
