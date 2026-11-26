import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module, plus_plus: bool = False):
        self.model = model
        self.plus_plus = plus_plus
        self._activations = None
        self._gradients = None
        self._handles = [target_layer.register_forward_hook(self._save_activation)]

    def _save_activation(self, _module, _inputs, output):
        self._activations = output.detach()
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self._gradients = grad.detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1)[0])
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()

        acts, grads = self._activations[0], self._gradients[0]
        if self.plus_plus:
            grads_2, grads_3 = grads.pow(2), grads.pow(3)
            denom = 2 * grads_2 + (acts * grads_3).sum(dim=(1, 2), keepdim=True)
            alpha = grads_2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            weights = (alpha * F.relu(grads)).sum(dim=(1, 2))
        else:
            weights = grads.mean(dim=(1, 2))

        cam = F.relu((weights[:, None, None] * acts).sum(dim=0))
        cam = F.interpolate(cam[None, None], size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return cam.cpu().numpy(), class_idx, probs[0].detach().cpu().numpy()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45, cmap: str = "jet") -> Image.Image:
    base = image.convert("RGB")
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(base.size, Image.Resampling.BILINEAR)
    colored = colormaps[cmap](np.asarray(cam_img, dtype=np.float32) / 255.0)[..., :3]
    heat = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(base, heat, alpha)


def gradcam_examples(model, meta, samples, probs, labels, transform, device, per_class=2, n_wrong=4, seed=0):
    from brain_mri.data import load_rgb
    from brain_mri.models import gradcam_target_layer

    rng = np.random.default_rng(seed)
    preds = probs.argmax(1)
    chosen = []
    for c in range(probs.shape[1]):
        idx = np.where((labels == c) & (preds == c))[0]
        chosen += list(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
    wrong = np.where(preds != labels)[0]
    chosen += list(wrong[np.argsort(-probs[wrong].max(1))][:n_wrong])
    items = []
    with GradCAM(model, gradcam_target_layer(model, meta["arch"])) as cam_fn:
        for i in chosen:
            image = load_rgb(samples[i][0]).resize((meta["img_size"], meta["img_size"]))
            cam, pred, p = cam_fn(transform(image).unsqueeze(0).to(device))
            items.append({"image": image, "cam": cam, "true": int(labels[i]), "pred": pred, "conf": float(p.max())})
    return items


def focus_box(cam: np.ndarray, threshold: float = 0.6):
    ys, xs = np.where(cam >= threshold)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
