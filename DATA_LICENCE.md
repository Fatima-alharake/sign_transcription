# Data licensing — what is in this repository and what is not

> **The code in this repository is MIT licensed ([LICENSE](LICENSE)). This file
> is about the DATA, which is not MIT and cannot be made MIT by us.** The two
> are governed separately: MIT covers what we wrote, the C-UDA covers what we
> received and anything derived from it that is still Data.

This project pretrains a backbone on **WLASL** (Word-Level American Sign
Language), which is distributed under the **Computational Use of Data Agreement
v1.0 (C-UDA)**. Canonical text: <https://cdla.dev/computational-use-of-data-agreement-v1-0/>

The licence text is not reproduced here — read it at that URL. The operative
clauses, quoted:

- **Data** is *"the material you receive under the C-UDA in modified or
  unmodified form, but not including Results"*.
- **Computational Use** is *"activities necessary to enable the use of Data
  (alone or along with other material) for analysis by a computer"*.
- **Result** is *"anything that you develop or improve from your use of Data
  that does not include more than a de minimis portion of the Data on which the
  use is based"*.
- The core restriction: *"You will use the Data solely for Computational Use."*
- The carve-out that matters most here: *"The C-UDA does not impose any
  restriction with respect to the use, modification, or distribution of
  Results."*
- Redistributing Data obliges you to *"bind each recipient to whom you
  redistribute the Data to the terms of the C-UDA"* and to include *"all credit
  or attribution information that you received with the Data"*, with terms
  requiring *"any Downstream Recipient to do the same"*.

**This is a reading of the licence, not legal advice.** Where the call is a
judgement rather than a plain reading, that is said below rather than hidden.

## What is in this repository

| Artifact | In repo? | Why |
|---|---|---|
| Code, configs, tests | **yes** | Ours; not Data. |
| `msl/data/splits/*.csv` — split assignments (video_id, gloss, gloss_id, signer_id, split) | **yes** | See "the split CSVs" below. |
| `msl/data/splits/wlasl100_separator_trims.csv` — frame ranges to keep | **yes** | Frame indices and counts; derived measurements, no landmark or pixel content. |
| `data/wlasl/download_log_100.csv`, `extract_log_firstpass.csv` | **yes** | Per-clip resolution, fps, frame counts, URLs, HTTP status. Measurements about the Data, not the Data. |
| `checkpoints/` — model checkpoints & ONNX weights | **yes** | Results / trained weights. |
| Raw video (`data/wlasl/raw_videos/`) | **no** | Data. Gitignored. |
| Landmark cache (`data/landmark_cache/*.npy`) | **no** | See "the landmark cache" below — the cautious call. |
| Trained checkpoints (`checkpoints/`) | **not currently** | Permitted as Results; excluded for size, not licensing. |

## The four questions, answered

### Trained weights derived from WLASL — **may be published**

A trained model is a **Result**: *"anything that you develop or improve from
your use of Data that does not include more than a de minimis portion of the
Data"*. A 0.6M-parameter classifier trained on 666 clips does not carry more
than a de minimis portion of those clips. And the C-UDA is explicit that
*"the C-UDA does not impose any restriction with respect to the use,
modification, or distribution of Results."*

So checkpoints could go in the repo. They are excluded here because a `.pt` is
7 MB of binary that git keeps forever, not because the licence forbids it.

Two caveats that are ours, not the licence's:
- Anything published must be labelled **American Sign Language**, never MSL, and
  must carry the 55.3%-complete-corpus caveat. That is an honesty requirement.
- WLASL aggregates 8 dictionary websites, and the C-UDA governs the compilation
  as received. It does not retroactively grant rights the original uploaders
  never gave. Publishing weights is low risk; republishing source video is not.

### The split CSVs — **may be published** (our judgement)

These contain `video_id, gloss, gloss_id, signer_id, split`. The video ids and
glosses come from WLASL's metadata, so this is arguably a small extract of Data
rather than a Result. We publish them anyway, for two reasons:

1. Without them, none of the reported numbers is reproducible. The whole point
   of building a signer-independent split was that WLASL's shipped split is not
   one; a number nobody can reproduce is not a result.
2. They are *de minimis* — a few thousand rows of identifiers and labels, no
   video, no frames, no landmarks.

**This is the one call in this table that is a judgement rather than a plain
reading.** If you are reusing this repository commercially, treat the split CSVs
as C-UDA Data and satisfy yourself independently.

### The landmark cache — **kept out** (the cautious call)

Each `.npy` is a per-frame normalised landmark tensor: 275 numbers per frame for
every frame of every clip. It is a lossy but dense derivative — the body,
hand and face geometry of an identifiable signer, frame by frame.

Reading it as a Result is *arguable*: it is derived from the Data and contains
no pixels. We do not publish it, for two reasons, one legal and one not:

- It is much closer to *"more than a de minimis portion of the Data"* than a
  set of model weights. It is not a summary of the clips, it is a
  re-encoding of them.
- **Sign language video cannot be anonymised.** The face carries grammar and the
  body carries identity; a landmark stream is biometric data about a specific
  person who consented to a dictionary website, not to our redistribution. This
  reason stands regardless of what the licence permits.

It is gitignored anyway for size (84 MB), so nothing turns on it in practice.

### The download and extraction logs — **may be published**

`download_log_100.csv` and `extract_log_firstpass.csv` hold per-clip width,
height, fps, frame counts, byte sizes, source site, HTTP status and the source
URL. These are **measurements about** the Data — metadata and provenance — not
the Data itself, and they contain nothing from which any clip could be
reconstructed. They are committed because spec §4 requires per-clip aspect
ratio and the landmark cache does not store frame dimensions.

They do contain **source URLs**, which is a feature: it is the attribution and
provenance trail the C-UDA asks redistributors to preserve.

## What must stay out, unconditionally

- **Raw video.** This is Data in the plainest sense. Redistributing it would
  oblige us to bind every downstream recipient to the C-UDA, which a public
  GitHub repository cannot do.
- **Any commercial use.** *"You will use the Data solely for Computational
  Use."* This repository is academic and research work.
- **Any WLASL-derived number presented as Mongolian Sign Language**, or without
  the corpus-completeness caveat. Not a licensing rule — a correctness one.

## Attribution

WLASL: Dongxu Li, Cristian Rodriguez Opazo, Xin Yu, Hongdong Li. *Word-level
Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods
Comparison.* WACV 2020. <https://dxli94.github.io/WLASL/>
