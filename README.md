# SemanticVAD

This is a mini project on finetuning lightweight models to implement semanticVAD.

## Project Goal

The goal of this project is to:
	•	Detect interruptions in a human-machine dialogue system within 200 milliseconds
	•	Accurately identify backchannels (e.g., “yeah”, “uh-huh”, “right”) to avoid unnecessary turn-taking
	•	Achieve low latency and memory usage suitable for real-time applications

## System Overview

The pipeline processes audio in 500ms batches:
	1.	Traditional VAD filters out silence.
	2.	When speech is detected:
	•	A lightweight model analyzes raw audio to detect monosyllabic backchannels.
	•	A semantic model (based on transcribed text) handles longer utterances.
	3.	If a backchannel is detected, the segment is skipped from further processing.
	4.	SemanticVAD runs in parallel with STT to reduce overall processing latency.

## Dataset
    Datasource: [Switchboard1 - release2](https://catalog.ldc.upenn.edu/LDC97S62)
	•	Total ~40000+ labeled audio segments
	•	Two classes:
        0 – Backchannel
        1 – Interruption
	•	Segment length: 0.5 seconds

## Model and Training
	•	Base models: DistilHuBERT, Wav2Vec2
	•	Classification head: simple MLP
	•	Loss function: CrossEntropyLoss
	•	Trained using PyTorch and Hugging Face Transformers

## Huggingface Models
