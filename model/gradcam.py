import numpy as np

def gradcam_overlay_placeholder(mel_db: np.ndarray) -> np.ndarray:
    """Returns a dummy heatmap [0,1] same shape as mel_db. Replace with real Grad-CAM for your CNN."""
    h, w = mel_db.shape
    # simple center-weighted blob
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h/2, w/2
    d = ((yy-cy)**2/(0.15*h*h) + (xx-cx)**2/(0.3*w*w))
    heat = np.exp(-d)
    heat = (heat - heat.min())/(heat.max() - heat.min() + 1e-9)
    return heat