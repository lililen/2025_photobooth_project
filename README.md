# Artbooth

> Turn your webcam into an art studio — apply live artistic filters inspired by famous painting styles, then snap a photo to keep.

A real-time artistic photo booth application that applies live visual effects to your webcam feed using OpenCV and PyTorch neural style transfer.

## Features

| Key | Effect |
|-----|--------|
| `1` | Normal (no filter) |
| `2` | Pencil Sketch |
| `3` | Watercolor |
| `4` | Van Gogh — *Starry Night* style transfer |
| `5` | Monet — *Mosaic* style transfer |
| `S` | Capture and save photo as `picture.png` |
| `Q` | Quit |

## Requirements

- Python 3.8+
- Webcam

Install dependencies:

```bash
pip install opencv-python numpy torch torchvision
```

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/photobooth.git
cd photobooth
```

2. Ensure the pretrained style transfer models are in `PB/models/`:

```
PB/models/
├── starry_night_pretrained.pth
├── mosaic_pretrained.pth
└── rain_princess_pretrained.pth
```

3. Run the app:

```bash
python PB/camera.py
```

## How It Works

- **Pencil Sketch** — converts the frame to grayscale, inverts it, applies Gaussian blur, then divides to produce a sketch effect using OpenCV.
- **Watercolor** — uses OpenCV's `stylization` filter for a painterly look.
- **Van Gogh / Monet** — applies neural style transfer using TorchScript pretrained models (`.pth` files). Frames are passed through the model in real time with `torch.no_grad()` for efficiency.

## Project Structure

```
photobooth/
├── PB/
│   ├── camera.py          # Main application
│   └── models/
│       ├── starry_night_pretrained.pth
│       ├── mosaic_pretrained.pth
│       └── rain_princess_pretrained.pth
└── README.md
```

## Notes

- The app resizes all frames to 650×400 and mirrors them horizontally for a natural photo booth feel.
- Neural style transfer runs on CPU by default. For faster performance, a CUDA-capable GPU is recommended.
- Captured photos are saved to `picture.png` in the working directory.
