import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import comfy.clip_vision
import json
import numpy as np
from typing import List, Optional, Tuple, Union

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

            print('start_image:', start_image)
            concat_latent_image = vae.encode(image[:, :, :, :3])
            print('concat_latent_image:', concat_latent_image)
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
            print(f"Warning: Unexpected track format: {type(tracks_data[0])}")
            
    except json.JSONDecodeError as e:
        print(f"Error parsing tracks JSON: {e}")
        tracks_data = []
    return tracks_data


def tracks_to_tensor(tracks_data, length, width, height, batch_size=1):
    """Convert parsed track data to tensor format (B, T, N, 4)"""
    if not tracks_data:
        # Return empty tracks if no data
        return torch.zeros((batch_size, length, 1, 4))
    
    num_tracks = len(tracks_data)
    tracks_tensor = torch.zeros((batch_size, length, num_tracks, 4))
    
    for batch_idx in range(batch_size):
        for track_idx, track in enumerate(tracks_data):
            for frame_idx in range(min(length, len(track))):
                point = track[frame_idx]
                if isinstance(point, dict):
                    x = point.get('x', 0)
                    y = point.get('y', 0)
                    # Normalize coordinates to [-1, 1] range
                    x_norm = (x / width) * 2 - 1
                    y_norm = (y / height) * 2 - 1
                    visible = point.get('visible', 1)
                    
                    tracks_tensor[batch_idx, frame_idx, track_idx] = torch.tensor([
                        track_idx,  # track_id
                        x_norm,     # x coordinate
                        y_norm,     # y coordinate  
                        visible     # visibility
                    ])
    
    return tracks_tensor


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


def patch_motion(
    tracks: torch.FloatTensor,  # (B, T, N, 4)
    vid: torch.FloatTensor,  # (C, T, H, W)
    temperature: float = 220.0,
    vae_divide: tuple = (4, 16),
    topk: int = 2,
):
    """Apply motion patching based on tracks"""
    with torch.no_grad():
        print(f"patch_motion: Input tracks shape: {tracks.shape}")
        print(f"patch_motion: Input tracks value range: min={tracks.min():.4f}, max={tracks.max():.4f}, mean={tracks.mean():.4f}")
        print(f"patch_motion: Input vid shape: {vid.shape}")
        print(f"patch_motion: Input vid value range: min={vid.min():.4f}, max={vid.max():.4f}, mean={vid.mean():.4f}")
        print(f"patch_motion: Parameters - temperature: {temperature}, vae_divide: {vae_divide}, topk: {topk}")
        
        _, T, H, W = vid.shape
        N = tracks.shape[2]
        _, tracks_xy, visible = torch.split(
            tracks, [1, 2, 1], dim=-1
        )  # (B, T, N, 2) | (B, T, N, 1)
        print(f"patch_motion: Split tracks - tracks_xy shape: {tracks_xy.shape}, visible shape: {visible.shape}")
        print(f"patch_motion: tracks_xy value range: min={tracks_xy.min():.4f}, max={tracks_xy.max():.4f}, mean={tracks_xy.mean():.4f}")
        print(f"patch_motion: visible value range: min={visible.min():.4f}, max={visible.max():.4f}, mean={visible.mean():.4f}")
        
        tracks_n = tracks_xy / torch.tensor([W / min(H, W), H / min(H, W)], device=tracks_xy.device)
        tracks_n = tracks_n.clamp(-1, 1)
        visible = visible.clamp(0, 1)
        print(f"patch_motion: Normalized tracks_n value range: min={tracks_n.min():.4f}, max={tracks_n.max():.4f}, mean={tracks_n.mean():.4f}")
        print(f"patch_motion: Clamped visible value range: min={visible.min():.4f}, max={visible.max():.4f}, mean={visible.mean():.4f}")

        xx = torch.linspace(-W / min(H, W), W / min(H, W), W)
        yy = torch.linspace(-H / min(H, W), H / min(H, W), H)

        grid = torch.stack(torch.meshgrid(yy, xx, indexing="ij")[::-1], dim=-1).to(
            tracks_xy.device
        )
        print(f"patch_motion: Grid shape: {grid.shape}")
        print(f"patch_motion: Grid value range: min={grid.min():.4f}, max={grid.max():.4f}, mean={grid.mean():.4f}")

        tracks_pad = tracks_xy[:, 1:]
        visible_pad = visible[:, 1:]

        visible_align = visible_pad.view(T - 1, 4, *visible_pad.shape[2:]).sum(1)
        tracks_align = (tracks_pad * visible_pad).view(T - 1, 4, *tracks_pad.shape[2:]).sum(
            1
        ) / (visible_align + 1e-5)
        print(f"patch_motion: visible_align shape: {visible_align.shape}, value range: min={visible_align.min():.4f}, max={visible_align.max():.4f}")
        print(f"patch_motion: tracks_align shape: {tracks_align.shape}, value range: min={tracks_align.min():.4f}, max={tracks_align.max():.4f}")
        
        dist_ = (
            (tracks_align[:, None, None] - grid[None, :, :, None]).pow(2).sum(-1)
        )  # T, H, W, N
        print(f"patch_motion: Distance tensor shape: {dist_.shape}, value range: min={dist_.min():.4f}, max={dist_.max():.4f}")
        
        weight = torch.exp(-dist_ * temperature) * visible_align.clamp(0, 1).view(
            T - 1, 1, 1, N
        )
        print(f"patch_motion: Weight tensor shape: {weight.shape}")
        print(f"patch_motion: Weight value range: min={weight.min():.4f}, max={weight.max():.4f}, mean={weight.mean():.4f}")
        
        vert_weight, vert_index = torch.topk(
            weight, k=min(topk, weight.shape[-1]), dim=-1
        )
        print(f"patch_motion: Top-k weight shape: {vert_weight.shape}, value range: min={vert_weight.min():.4f}, max={vert_weight.max():.4f}")

    grid_mode = "bilinear"
    vid_slice = vid[vae_divide[0]:].permute(1, 0, 2, 3)[:1]
    print(f"patch_motion: vid_slice for grid_sample shape: {vid_slice.shape}, value range: min={vid_slice.min():.4f}, max={vid_slice.max():.4f}")
    tracks_input = tracks_n[:, :1].type(vid.dtype)
    print(f"patch_motion: tracks_input for grid_sample shape: {tracks_input.shape}, value range: min={tracks_input.min():.4f}, max={tracks_input.max():.4f}")
    
    point_feature = torch.nn.functional.grid_sample(
        vid_slice,
        tracks_input,
        mode=grid_mode,
        padding_mode="zeros",
        align_corners=False,
    )
    print(f"patch_motion: point_feature after grid_sample shape: {point_feature.shape}, value range: min={point_feature.min():.4f}, max={point_feature.max():.4f}")
    
    point_feature = point_feature.squeeze(0).squeeze(1).permute(1, 0) # N, C=16
    print(f"patch_motion: point_feature after squeeze/permute shape: {point_feature.shape}, value range: min={point_feature.min():.4f}, max={point_feature.max():.4f}")

    out_feature = merge_final(point_feature, vert_weight, vert_index).permute(3, 0, 1, 2) # T - 1, H, W, C => C, T - 1, H, W
    print(f"patch_motion: out_feature after merge_final shape: {out_feature.shape}, value range: min={out_feature.min():.4f}, max={out_feature.max():.4f}")
    
    out_weight = vert_weight.sum(-1) # T - 1, H, W
    print(f"patch_motion: out_weight shape: {out_weight.shape}, value range: min={out_weight.min():.4f}, max={out_weight.max():.4f}")

    # out feature -> already soft weighted
    vid_background = vid[vae_divide[0]:, 1:]
    print(f"patch_motion: vid_background shape: {vid_background.shape}, value range: min={vid_background.min():.4f}, max={vid_background.max():.4f}")
    weight_complement = (1 - out_weight.clamp(0, 1))
    print(f"patch_motion: weight_complement value range: min={weight_complement.min():.4f}, max={weight_complement.max():.4f}")
    
    mix_feature = out_feature + vid_background * weight_complement
    print(f"patch_motion: mix_feature after blending shape: {mix_feature.shape}, value range: min={mix_feature.min():.4f}, max={mix_feature.max():.4f}")

    vid_first_frame = vid[vae_divide[0]:, :1]
    print(f"patch_motion: vid_first_frame shape: {vid_first_frame.shape}, value range: min={vid_first_frame.min():.4f}, max={vid_first_frame.max():.4f}")
    
    out_feature_full = torch.cat([vid_first_frame, mix_feature], dim=1) # C, T, H, W
    print(f"patch_motion: out_feature_full shape: {out_feature_full.shape}, value range: min={out_feature_full.min():.4f}, max={out_feature_full.max():.4f}")
    
    ones_like_first = torch.ones_like(out_weight[:1])
    print(f"patch_motion: ones_like_first shape: {ones_like_first.shape}, value range: min={ones_like_first.min():.4f}, max={ones_like_first.max():.4f}")
    
    out_mask_full = torch.cat([ones_like_first, out_weight], dim=0)  # T, H, W
    print(f"patch_motion: out_mask_full shape: {out_mask_full.shape}, value range: min={out_mask_full.min():.4f}, max={out_mask_full.max():.4f}")
    
    mask_expanded = out_mask_full[None].expand(vae_divide[0], -1, -1, -1)
    print(f"patch_motion: mask_expanded shape: {mask_expanded.shape}, value range: min={mask_expanded.min():.4f}, max={mask_expanded.max():.4f}")
    
    final_result = torch.cat([mask_expanded, out_feature_full], dim=0)
    print(f"patch_motion: final_result shape: {final_result.shape}, value range: min={final_result.min():.4f}, max={final_result.max():.4f}")
    
    return final_result

def process_tracks(tracks_np: np.ndarray, frame_size: Tuple[int, int], quant_multi: int = 8, **kwargs):
    # tracks: shape [t, h, w, 3] => samples align with 24 fps, model trained with 16 fps.
    # frame_size: tuple (W, H)
    print(f"process_tracks: Input tracks_np shape: {tracks_np.shape}, dtype: {tracks_np.dtype}")
    print(f"process_tracks: Input tracks_np value range: min={tracks_np.min():.4f}, max={tracks_np.max():.4f}, mean={tracks_np.mean():.4f}")
    print(f"process_tracks: frame_size: {frame_size}")

    tracks = torch.from_numpy(tracks_np).float()
    print(f"process_tracks: tracks after torch conversion shape: {tracks.shape}")
    print(f"process_tracks: tracks after torch conversion value range: min={tracks.min():.4f}, max={tracks.max():.4f}, mean={tracks.mean():.4f}")
    
    if tracks.shape[1] == 121:
        tracks = torch.permute(tracks, (1, 0, 2, 3))
        print(f"process_tracks: tracks after permute shape: {tracks.shape}")
    
    tracks, visibles = tracks[..., :2], tracks[..., 2:3]
    print(f"process_tracks: split tracks shape: {tracks.shape}, visibles shape: {visibles.shape}")
    print(f"process_tracks: split tracks value range: min={tracks.min():.4f}, max={tracks.max():.4f}, mean={tracks.mean():.4f}")
    print(f"process_tracks: split visibles value range: min={visibles.min():.4f}, max={visibles.max():.4f}, mean={visibles.mean():.4f}")
    
    short_edge = min(*frame_size)
    print(f"process_tracks: short_edge: {short_edge}")

    frame_center = torch.tensor([*frame_size]).type_as(tracks) / 2
    print(f"process_tracks: frame_center: {frame_center}")
    tracks = tracks - frame_center
    print(f"process_tracks: tracks after centering value range: min={tracks.min():.4f}, max={tracks.max():.4f}, mean={tracks.mean():.4f}")
    
    tracks = tracks / short_edge * 2
    print(f"process_tracks: tracks after normalization value range: min={tracks.min():.4f}, max={tracks.max():.4f}, mean={tracks.mean():.4f}")

    visibles = visibles * 2 - 1
    print(f"process_tracks: visibles after scaling value range: min={visibles.min():.4f}, max={visibles.max():.4f}, mean={visibles.mean():.4f}")

    trange = torch.linspace(-1, 1, tracks.shape[0]).view(-1, 1, 1, 1).expand(*visibles.shape)
    print(f"process_tracks: trange shape: {trange.shape}, value range: min={trange.min():.4f}, max={trange.max():.4f}")

    out_ = torch.cat([trange, tracks, visibles], dim=-1).view(121, -1, 4)
    print(f"process_tracks: out_ after concat shape: {out_.shape}, value range: min={out_.min():.4f}, max={out_.max():.4f}")
    
    out_0 = out_[:1]
    print(f"process_tracks: out_0 shape: {out_0.shape}, value range: min={out_0.min():.4f}, max={out_0.max():.4f}")
    
    out_l = out_[1:] # 121 => 120 | 1
    print(f"process_tracks: out_l before interpolation shape: {out_l.shape}")
    out_l = torch.repeat_interleave(out_l, 2, dim=0)[1::3]  # 120 => 240 => 80
    print(f"process_tracks: out_l after interpolation shape: {out_l.shape}, value range: min={out_l.min():.4f}, max={out_l.max():.4f}")
    
    final_result = torch.cat([out_0, out_l], dim=0)
    print(f"process_tracks: final result shape: {final_result.shape}, value range: min={final_result.min():.4f}, max={final_result.max():.4f}")
    
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
                },
                "optional": {
                    "clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                    "start_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, tracks, width, height, length, batch_size, 
               temperature, topk, start_image=None, clip_vision_output=None):
        
        print(f"WanTrackToVideo.encode: Starting track-to-video encoding")
        print(f"WanTrackToVideo.encode: Input dimensions - width: {width}, height: {height}, length: {length}, batch_size: {batch_size}")
        print(f"WanTrackToVideo.encode: Track parameters - temperature: {temperature}, topk: {topk}")
        
        # Parse tracks from JSON
        tracks_data = parse_json_tracks(tracks)
        print(f"WanTrackToVideo.encode: Raw tracks input: {tracks}")
        print(f"WanTrackToVideo.encode: Parsed tracks data: {len(tracks_data) if tracks_data else 0} tracks")
        
        if tracks_data:
            print(f"WanTrackToVideo.encode: Processing {len(tracks_data)} track(s)")
            # Convert tracks to tensor format
            arrs = []
            for i, track in enumerate(tracks_data):
                print(f"WanTrackToVideo.encode: Processing track {i} with {len(track)} points")
                pts = pad_pts(track)
                arrs.append(pts)

            tracks_np = np.stack(arrs, axis=0)
            print(f"WanTrackToVideo.encode: Stacked tracks numpy shape: {tracks_np.shape}")
            processed_tracks = process_tracks(tracks_np, (width, height)).unsqueeze(0)
            print(f"WanTrackToVideo.encode: Processed tracks tensor shape: {processed_tracks.shape}")
            
            if start_image is not None:
                print(f"WanTrackToVideo.encode: Processing start image with shape: {start_image.shape}")
                print(f"WanTrackToVideo.encode: Start image value range: min={start_image.min():.4f}, max={start_image.max():.4f}, mean={start_image.mean():.4f}")
                start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                print(f"WanTrackToVideo.encode: Upscaled start image shape: {start_image.shape}")
                print(f"WanTrackToVideo.encode: Upscaled start image value range: min={start_image.min():.4f}, max={start_image.max():.4f}, mean={start_image.mean():.4f}")
                
                lat_h = height // 8
                lat_w = width // 8
                print(f"WanTrackToVideo.encode: Latent dimensions - lat_h: {lat_h}, lat_w: {lat_w}")

                msk = torch.ones(1, 81, lat_h, lat_w, device=start_image.device)
                msk[:, 1:] = 0
                print(f"WanTrackToVideo.encode: Initial mask shape: {msk.shape}")
                print(f"WanTrackToVideo.encode: Initial mask value range: min={msk.min():.4f}, max={msk.max():.4f}, mean={msk.mean():.4f}")
                
                # repeat first frame 4 times
                msk = torch.concat([
                    torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
                ],
                    dim=1)
                print(f"WanTrackToVideo.encode: Mask after first frame repeat: {msk.shape}")
                print(f"WanTrackToVideo.encode: Mask after repeat value range: min={msk.min():.4f}, max={msk.max():.4f}, mean={msk.mean():.4f}")

                # Reshape mask into groups of 4 frames
                msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                print(f"WanTrackToVideo.encode: Mask after reshape: {msk.shape}")

                # first batch
                msk = msk.transpose(1, 2)
                print(f"WanTrackToVideo.encode: Final mask shape: {msk.shape}")
                print(f"WanTrackToVideo.encode: Final mask value range: min={msk.min():.4f}, max={msk.max():.4f}, mean={msk.mean():.4f}")

                zero_frames = torch.zeros(3, 81 - 1, height, width)
                print(f"WanTrackToVideo.encode: Zero frames shape: {zero_frames.shape}")
                print(f"WanTrackToVideo.encode: Zero frames value range: min={zero_frames.min():.4f}, max={zero_frames.max():.4f}, mean={zero_frames.mean():.4f}")
                
                start_image = start_image.permute(3,0,1,2) # C, T, H, W
                print(f"WanTrackToVideo.encode: Start image after permute: {start_image.shape}")
                res = torch.concat([
                        start_image.to(start_image.device),
                        zero_frames
                    ],
                        dim=1).to(start_image.device)
                print(f"WanTrackToVideo.encode: Combined video tensor shape: {res.shape}")
                print(f"WanTrackToVideo.encode: Combined video tensor value range: min={res.min():.4f}, max={res.max():.4f}, mean={res.mean():.4f}")

                res = res.permute(1,2,3,0)[:, :, :, :3]  # T, H, W, C
                print(f"WanTrackToVideo.encode: Video tensor for VAE encoding: {res.shape}")
                print(f"WanTrackToVideo.encode: Video tensor for VAE value range: min={res.min():.4f}, max={res.max():.4f}, mean={res.mean():.4f}")
                
                y = vae.encode(res)
                print(f"WanTrackToVideo.encode: VAE encoded latent shape: {y.shape}")
                print(f"WanTrackToVideo.encode: VAE encoded latent value range: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")
                
                # Add motion features to conditioning
                print(f"WanTrackToVideo.encode: Adding tracks, mask, and latent to conditioning")
                print(f"WanTrackToVideo.encode: Processed tracks value range: min={processed_tracks.min():.4f}, max={processed_tracks.max():.4f}, mean={processed_tracks.mean():.4f}")
                positive = node_helpers.conditioning_set_values(positive,
                                                                {"tracks": processed_tracks,
                                                                 "concat_mask": msk,
                                                                "concat_latent_image": y})
                negative = node_helpers.conditioning_set_values(negative, 
                                                                {"tracks": processed_tracks,
                                                                 "concat_mask": msk,
                                                                "concat_latent_image": y})
                

        # Handle clip vision output if provided
        if clip_vision_output is not None:
            print(f"WanTrackToVideo.encode: Adding CLIP vision output to conditioning")
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], 
                           device=comfy.model_management.intermediate_device())
        print(f"WanTrackToVideo.encode: Created output latent with shape: {latent.shape}")
        out_latent = {}
        out_latent["samples"] = latent
        print(f"WanTrackToVideo.encode: Encoding complete")
        return (positive, negative, out_latent)

NODE_CLASS_MAPPINGS = {
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
