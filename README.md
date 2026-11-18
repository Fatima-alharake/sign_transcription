# ASL Recognition - Real-Time American Sign Language Recognition - In Progress
A landmark-based American Sign Language (ASL)  recognition system built on MediaPipe, Temporal Convolutional Networks (TCN), Supervised Contrastive Learning (SupCon), and ONNX Runtime for ultra-low latency pure CPU execution.

---

## System Architecture

```
                  ┌───────────────────────────────────────────────┐
                  │          Live Webcam / Video Frames           │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │      MediaPipe Holistic Landmark Extractor    │
                  │        (Pose, Left Hand, Right Hand, Face)    │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │    Landmark Normalisation & Velocity Append   │
                  │           Input Tensor: (B, 64, 366)          │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │     ONNX Accelerated Model (1.14 ms on CPU)   │
                  └───────────────┬───────────────────────┬───────┘
                                  │                       │
                     (Logits)     │                       │  (L2 Embedding)
                                  ▼                       ▼
                  ┌───────────────────────┐   ┌───────────────────────────┐
                  │ Softmax Probabilities │   │  Metric Retrieval Gallery │
                  │     Top-k Ranking     │   │    (Cosine Similarity)    │
                  └───────────────┬───────┘   └───────────┬───────────────┘
                                  │                       │
                                  └───────────┬───────────┘
                                              ▼
                  ┌───────────────────────────────────────────────┐
                  │        Out-of-Vocabulary (OOV) Filter         │
                  │   (Rejects if Cosine Sim < 0.50 Threshold)    │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                  ┌───────────────────────────────────────────────┐
                  │    Live HUD: Debounced Subtitle Accumulator   │
                  │      Top-3 Candidates Bar + [LR] Tracking     │
                  └───────────────────────────────────────────────┘
```

---
