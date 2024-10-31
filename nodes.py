import os
import numpy as np
from PIL import Image, ImageSequence, ImageOps
import folder_paths
import torch
import hashlib

class ETM_SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "format": (["png", "webp"], {"default": "webp", "tooltip": "The format to save the images in."}),
                "webp_quality": ("INT", {"default": 85, "min": 1, "max": 100, "step": 1, "tooltip": "The quality of the webp images. Only used when format is webp."})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, filename_prefix="ComfyUI", format="webp", webp_quality=85):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            
            if format == "webp":
                file = f"{filename_with_batch_num}_{counter:05}_.webp"
                img.save(os.path.join(full_output_folder, file), quality=webp_quality, optimize=True)
            else:
                file = f"{filename_with_batch_num}_{counter:05}_.png"
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
    

class ETM_LoadImageFromLocal:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "root_dir": ("STRING", {"default": "/root/autodl-tmp/local_images", "multiline": False, "dynamicPrompts": False}),
                "image_path": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
            },
                }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, root_dir, image_path):
        img = Image.open(os.path.join(root_dir, image_path))
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, root_dir, image_path):
        m = hashlib.sha256()
        with open(os.path.join(root_dir, image_path), 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, root_dir,image_path):
        if not os.path.exists(os.path.join(root_dir, image_path)):
            return "File not exist: {}".format(image_path)

        return True
    
