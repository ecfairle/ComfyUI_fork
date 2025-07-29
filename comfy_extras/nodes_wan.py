import math
import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import comfy.clip_vision
import json
import numpy as np
from typing import Tuple
from einops import rearrange
import folder_paths

class WanImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            image[:start_image.shape[0]] = start_image

            concat_latent_image = vae.encode(image[:, :, :, :3])
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


class WanFunControlToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "control_video": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None, control_video=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)
        concat_latent = concat_latent.repeat(1, 2, 1, 1, 1)

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(start_image[:, :, :, :3])
            concat_latent[:,16:,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(control_video[:, :, :, :3])
            concat_latent[:,:16,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)

class WanFirstLastFrameToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_start_image": ("CLIP_VISION_OUTPUT", ),
                             "clip_vision_end_image": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "end_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            mask[:, :, :start_image.shape[0] + 3] = 0.0

        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            mask[:, :, -end_image.shape[0]:] = 0.0

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_start_image is not None:
            clip_vision_output = clip_vision_start_image

        if clip_vision_end_image is not None:
            if clip_vision_output is not None:
                states = torch.cat([clip_vision_output.penultimate_hidden_states, clip_vision_end_image.penultimate_hidden_states], dim=-2)
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_end_image

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


class WanFunInpaintToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "end_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, end_image=None, clip_vision_output=None):
        flfv = WanFirstLastFrameToVideo()
        return flfv.encode(positive, negative, vae, width, height, length, batch_size, start_image=start_image, end_image=end_image, clip_vision_start_image=clip_vision_output)


class WanVaceToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                },
                "optional": {"control_video": ("IMAGE", ),
                             "control_masks": ("MASK", ),
                             "reference_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "trim_latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    EXPERIMENTAL = True

    def encode(self, positive, negative, vae, width, height, length, batch_size, strength, control_video=None, control_masks=None, reference_image=None):
        latent_length = ((length - 1) // 4) + 1
        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(control_video, (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]), value=0.5)
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        if reference_image is not None:
            reference_image = comfy.utils.common_upscale(reference_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            reference_image = vae.encode(reference_image[:, :, :, :3])
            reference_image = torch.cat([reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))], dim=1)

        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        inactive = vae.encode(inactive[:, :, :, :3])
        reactive = vae.encode(reactive[:, :, :, :3])
        control_video_latent = torch.cat((inactive, reactive), dim=1)
        if reference_image is not None:
            control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

        trim_latent = 0
        if reference_image is not None:
            mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
            mask = torch.cat((mask_pad, mask), dim=1)
            latent_length += reference_image.shape[2]
            trim_latent = reference_image.shape[2]

        mask = mask.unsqueeze(0)

        positive = node_helpers.conditioning_set_values(positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)

        latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent, trim_latent)

class TrimVideoLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),
                              "trim_amount": ("INT", {"default": 0, "min": 0, "max": 99999}),
                             }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "op"

    CATEGORY = "latent/video"

    EXPERIMENTAL = True

    def op(self, samples, trim_amount):
        samples_out = samples.copy()

        s1 = samples["samples"]
        samples_out["samples"] = s1[:, :, trim_amount:]
        return (samples_out,)

class WanCameraImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "camera_conditions": ("WAN_CAMERA_EMBEDDING", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None, camera_conditions=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(start_image[:, :, :, :3])
            concat_latent[:,:,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent})

        if camera_conditions is not None:
            positive = node_helpers.conditioning_set_values(positive, {'camera_conditions': camera_conditions})
            negative = node_helpers.conditioning_set_values(negative, {'camera_conditions': camera_conditions})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)

class WanPhantomSubjectToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"images": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative_text", "negative_img_text", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, images):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        cond2 = negative
        if images is not None:
            images = comfy.utils.common_upscale(images[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            latent_images = []
            for i in images:
                latent_images += [vae.encode(i.unsqueeze(0)[:, :, :, :3])]
            concat_latent_image = torch.cat(latent_images, dim=2)

            positive = node_helpers.conditioning_set_values(positive, {"time_dim_concat": concat_latent_image})
            cond2 = node_helpers.conditioning_set_values(negative, {"time_dim_concat": concat_latent_image})
            negative = node_helpers.conditioning_set_values(negative, {"time_dim_concat": comfy.latent_formats.Wan21().process_out(torch.zeros_like(concat_latent_image))})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, cond2, negative, out_latent)

def parse_json_tracks(tracks):
    """Parse JSON track data into a standardized format"""
    tracks_data = []
    try:
        # If tracks is a string, try to parse it as JSON
        if isinstance(tracks, str):
            parsed = json.loads(tracks.replace("'", '"'))
            tracks_data.extend(parsed)
        else:
            # If tracks is a list of strings, parse each one
            for track_str in tracks:
                parsed = json.loads(track_str.replace("'", '"'))
                tracks_data.append(parsed)

        # Check if we have a single track (dict with x,y) or a list of tracks
        if tracks_data and isinstance(tracks_data[0], dict) and 'x' in tracks_data[0]:
            # Single track detected, wrap it in a list
            tracks_data = [tracks_data]
        elif tracks_data and isinstance(tracks_data[0], list) and tracks_data[0] and isinstance(tracks_data[0][0], dict) and 'x' in tracks_data[0][0]:
            # Already a list of tracks, nothing to do
            pass
        else:
            # Unexpected format
            pass

    except json.JSONDecodeError:
        tracks_data = []
    return tracks_data

def process_tracks(tracks_np: np.ndarray, frame_size: Tuple[int, int], num_frames, quant_multi: int = 8, **kwargs):
    # tracks: shape [t, h, w, 3] => samples align with 24 fps, model trained with 16 fps.
    # frame_size: tuple (W, H)
    tracks = torch.from_numpy(tracks_np).float()

    if tracks.shape[1] == 121:
        tracks = torch.permute(tracks, (1, 0, 2, 3))

    tracks, visibles = tracks[..., :2], tracks[..., 2:3]

    short_edge = min(*frame_size)

    frame_center = torch.tensor([*frame_size]).type_as(tracks) / 2
    tracks = tracks - frame_center

    tracks = tracks / short_edge * 2

    visibles = visibles * 2 - 1

    trange = torch.linspace(-1, 1, tracks.shape[0]).view(-1, 1, 1, 1).expand(*visibles.shape)

    out_ = torch.cat([trange, tracks, visibles], dim=-1).view(121, -1, 4)

    out_0 = out_[:1]

    out_l = out_[1:] # 121 => 120 | 1
    a = 120 // math.gcd(120, num_frames)
    b = num_frames // math.gcd(120, num_frames)
    out_l = torch.repeat_interleave(out_l, b, dim=0)[1::a]  # 120 => 120 * b => 120 * b / a == F

    final_result = torch.cat([out_0, out_l], dim=0)

    return final_result

FIXED_LENGTH = 121
def pad_pts(tr):
    """Convert list of {x,y} to (FIXED_LENGTH,1,3) array, padding/truncating."""
    pts = np.array([[p['x'], p['y'], 1] for p in tr], dtype=np.float32)
    n = pts.shape[0]
    if n < FIXED_LENGTH:
        pad = np.zeros((FIXED_LENGTH - n, 3), dtype=np.float32)
        pts = np.vstack((pts, pad))
    else:
        pts = pts[:FIXED_LENGTH]
    return pts.reshape(FIXED_LENGTH, 1, 3)

def ind_sel(target: torch.Tensor, ind: torch.Tensor, dim: int = 1):
    """Index selection utility function"""
    assert (
        len(ind.shape) > dim
    ), "Index must have the target dim, but get dim: %d, ind shape: %s" % (dim, str(ind.shape))

    target = target.expand(
        *tuple(
            [ind.shape[k] if target.shape[k] == 1 else -1 for k in range(dim)]
            + [
                -1,
            ]
            * (len(target.shape) - dim)
        )
    )

    ind_pad = ind

    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            ind_pad = ind_pad.unsqueeze(-1)
        ind_pad = ind_pad.expand(*(-1,) * (dim + 1), *target.shape[(dim + 1) : :])

    return torch.gather(target, dim=dim, index=ind_pad)

def merge_final(vert_attr: torch.Tensor, weight: torch.Tensor, vert_assign: torch.Tensor):
    """Merge vertex attributes with weights"""
    target_dim = len(vert_assign.shape) - 1
    if len(vert_attr.shape) == 2:
        assert vert_attr.shape[0] > vert_assign.max()
        new_shape = [1] * target_dim + list(vert_attr.shape)
        tensor = vert_attr.reshape(new_shape)
        sel_attr = ind_sel(tensor, vert_assign.type(torch.long), dim=target_dim)
    else:
        assert vert_attr.shape[1] > vert_assign.max()
        new_shape = [vert_attr.shape[0]] + [1] * (target_dim - 1) + list(vert_attr.shape[1:])
        tensor = vert_attr.reshape(new_shape)
        sel_attr = ind_sel(tensor, vert_assign.type(torch.long), dim=target_dim)

    final_attr = torch.sum(sel_attr * weight.unsqueeze(-1), dim=-2)
    return final_attr


def _patch_motion_single(
    tracks: torch.FloatTensor,  # (B, T, N, 4)
    vid: torch.FloatTensor,     # (C, T, H, W)
    temperature: float,
    vae_divide: tuple,
    topk: int,
):
    """Apply motion patching based on tracks"""
    _, T, H, W = vid.shape
    N = tracks.shape[2]
    _, tracks_xy, visible = torch.split(
        tracks, [1, 2, 1], dim=-1
    )  # (B, T, N, 2) | (B, T, N, 1)
    tracks_n = tracks_xy / torch.tensor([W / min(H, W), H / min(H, W)], device=tracks_xy.device)
    tracks_n = tracks_n.clamp(-1, 1)
    visible = visible.clamp(0, 1)

    xx = torch.linspace(-W / min(H, W), W / min(H, W), W)
    yy = torch.linspace(-H / min(H, W), H / min(H, W), H)

    grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1).to(
        tracks_xy.device
    )

    tracks_pad = tracks_xy[:, 1:]
    visible_pad = visible[:, 1:]

    visible_align = visible_pad.view(T - 1, 4, *visible_pad.shape[2:]).sum(1)
    tracks_align = (tracks_pad * visible_pad).view(T - 1, 4, *tracks_pad.shape[2:]).sum(
        1
    ) / (visible_align + 1e-5)
    dist_ = (
        (tracks_align[:, None, None] - grid[None, :, :, None]).pow(2).sum(-1)
    )  # T, H, W, N
    weight = torch.exp(-dist_ * temperature) * visible_align.clamp(0, 1).view(
        T - 1, 1, 1, N
    )
    vert_weight, vert_index = torch.topk(
        weight, k=min(topk, weight.shape[-1]), dim=-1
    )

    grid_mode = "bilinear"
    point_feature = torch.nn.functional.grid_sample(
        vid.permute(1, 0, 2, 3)[:1],
        tracks_n[:, :1].type(vid.dtype),
        mode=grid_mode,
        padding_mode="zeros",
        align_corners=False,
    )
    point_feature = point_feature.squeeze(0).squeeze(1).permute(1, 0) # N, C=16

    out_feature = merge_final(point_feature, vert_weight, vert_index).permute(3, 0, 1, 2) # T - 1, H, W, C => C, T - 1, H, W
    out_weight = vert_weight.sum(-1) # T - 1, H, W

    # out feature -> already soft weighted
    mix_feature = out_feature + vid[:, 1:] * (1 - out_weight.clamp(0, 1))

    out_feature_full = torch.cat([vid[:, :1], mix_feature], dim=1) # C, T, H, W
    out_mask_full = torch.cat([torch.ones_like(out_weight[:1]), out_weight], dim=0)  # T, H, W

    return out_mask_full[None].expand(vae_divide[0], -1, -1, -1), out_feature_full


def patch_motion(
    tracks: torch.FloatTensor,  # (B, TB, T, N, 4)
    vid: torch.FloatTensor,     # (C, T, H, W)
    temperature: float = 220.0,
    vae_divide: tuple = (4, 16),
    topk: int = 2,
):
    B = len(tracks)

    # Process each batch separately
    out_masks = []
    out_features = []

    for b in range(B):
        mask, feature = _patch_motion_single(
            tracks[b],  # (T, N, 4)
            vid[b],        # (C, T, H, W)
            temperature,
            vae_divide,
            topk
        )
        out_masks.append(mask)
        out_features.append(feature)

    # Stack results: (B, C, T, H, W)
    out_mask_full = torch.stack(out_masks, dim=0)
    out_feature_full = torch.stack(out_features, dim=0)

    return out_mask_full, out_feature_full

class WanTrackToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "vae": ("VAE", ),
                    "tracks": ("STRING", {"multiline": True, "default": "[]"}),
                    "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                    "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                    "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                    "temperature": ("FLOAT", {"default": 220.0, "min": 1.0, "max": 1000.0, "step": 0.1}),
                    "topk": ("INT", {"default": 2, "min": 1, "max": 10}),
                    "start_image": ("IMAGE", ),
                },
                "optional": {
                    "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, tracks, width, height, length, batch_size,
               temperature, topk, start_image=None, clip_vision_output=None):

        tracks_data = parse_json_tracks(tracks)

        if not tracks_data:
            return WanImageToVideo().encode(positive, negative, vae, width, height, length, batch_size, start_image=start_image, clip_vision_output=clip_vision_output)

        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8],
                           device=comfy.model_management.intermediate_device())

        if isinstance(tracks_data[0][0], dict):
            tracks_data = [tracks_data]

        processed_tracks = []
        for batch in tracks_data:
            arrs = []
            for track in batch:
                pts = pad_pts(track)
                arrs.append(pts)

            tracks_np = np.stack(arrs, axis=0)
            processed_tracks.append(process_tracks(tracks_np, (width, height), length - 1).unsqueeze(0))

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:batch_size].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            videos = torch.ones((start_image.shape[0], length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            for i in range(start_image.shape[0]):
                videos[i, 0] = start_image[i]

            latent_videos = []
            videos = comfy.utils.resize_to_batch_size(videos, batch_size)
            for i in range(batch_size):
                latent_videos += [vae.encode(videos[i, :, :, :, :3])]
            y = torch.cat(latent_videos, dim=0)

            # Scale latent since patch_motion is non-linear
            y = comfy.latent_formats.Wan21().process_in(y)

            processed_tracks = comfy.utils.resize_list_to_batch_size(processed_tracks, batch_size)
            res = patch_motion(
                processed_tracks, y, temperature=temperature, topk=topk, vae_divide=(4, 16)
            )

            mask, concat_latent_image = res
            concat_latent_image = comfy.latent_formats.Wan21().process_out(concat_latent_image)
            mask = -mask + 1.0  # Invert mask to match expected format
            positive = node_helpers.conditioning_set_values(positive,
                                                            {"concat_mask": mask,
                                                            "concat_latent_image": concat_latent_image})
            negative = node_helpers.conditioning_set_values(negative,
                                                            {"concat_mask": mask,
                                                            "concat_latent_image": concat_latent_image})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 25 # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb
from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput

def get_mask_from_lengths(lengths, max_len=None):
    lengths = lengths.to(torch.long)
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(lengths.shape[0], -1).to(lengths.device)
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = torch.nn.functional.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)

# the implementation of Wav2Vec2Model is borrowed from
# https://github.com/huggingface/transformers/blob/HEAD/src/transformers/models/wav2vec2/modeling_wav2vec2.py
# initialize our encoder with the pre-trained wav2vec 2.0 weights.
class Wav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)

    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        print("output_hidden_states", output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )
            

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Feature extractor class for Wav2Vec2
"""

from typing import List, Optional, Union

import numpy as np

from transformers import Wav2Vec2FeatureExtractor

import librosa
import pyloudnorm as pyln
def audio_prepare_single(audio_path, sample_rate=16000):
    human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    return human_speech_array

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio
    
def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):

    if not (left_path=='None' or right_path=='None'):
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = audio_prepare_single(right_path)
    elif left_path=='None':
        human_speech_array2 = audio_prepare_single(right_path)
        human_speech_array1 = np.zeros(human_speech_array2.shape[0])
    elif right_path=='None':
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    if audio_type=='para':
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type=='add':
        new_human_speech1 = np.concatenate([human_speech_array1[: human_speech_array1.shape[0]], np.zeros(human_speech_array2.shape[0])]) 
        new_human_speech2 = np.concatenate([np.zeros(human_speech_array1.shape[0]), human_speech_array2[:human_speech_array2.shape[0]]])
    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs

class WanMultiTalkToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    # "positive": ("CONDITIONING", ),
                    # "negative": ("CONDITIONING", ),
                    "vae": ("VAE", ),
                    "wav2vec": ("STRING", ),
                    "audio1": ("STRING", ),
                    "audio2": ("STRING", ),
                    "audio_type": ("STRING", {"default": "add", "values": ["add", "para"]}),
                    "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                    "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                    "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                    "start_image": ("IMAGE", ),
                },
                "optional": {
                    "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, vae, wav2vec, width, height, length, batch_size,
               audio1, audio2, audio_type, start_image=None, clip_vision_output=None):
        # wav2vec = folder_paths.get_full_path_or_raise("audio_encoders", wav2vec)
        audio1 = audio1.replace("\\", "/")
        audio2 = audio2.replace("\\", "/")
        
        
        audio_encoder = Wav2Vec2Model.from_pretrained(r"C:/Users/eugen/ComfyUI_fork/models/audio_encoders/", local_files_only=True, attn_implementation="eager", output_hidden_states=True).to(start_image.device)
        audio_encoder.feature_extractor._freeze_parameters()
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(r"C:/Users/eugen/ComfyUI_fork/models/audio_encoders/", local_files_only=True, attn_implementation="eager", output_hidden_states=True)
        new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(audio1, audio2, audio_type)
        audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
        audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
        print(audio_embedding_1.shape, audio_embedding_1.min(), audio_embedding_1.max())
        print(audio_embedding_2.min(), audio_embedding_2.max())
        return 0

NODE_CLASS_MAPPINGS = {
    "WanMultiTalkToVideo": WanMultiTalkToVideo,
    "WanTrackToVideo": WanTrackToVideo,
    "WanImageToVideo": WanImageToVideo,
    "WanFunControlToVideo": WanFunControlToVideo,
    "WanFunInpaintToVideo": WanFunInpaintToVideo,
    "WanFirstLastFrameToVideo": WanFirstLastFrameToVideo,
    "WanVaceToVideo": WanVaceToVideo,
    "TrimVideoLatent": TrimVideoLatent,
    "WanCameraImageToVideo": WanCameraImageToVideo,
    "WanPhantomSubjectToVideo": WanPhantomSubjectToVideo,
}
