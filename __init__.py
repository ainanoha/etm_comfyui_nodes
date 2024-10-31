from .nodes import  ETM_SaveImage, ETM_LoadImageFromLocal

NODE_CLASS_MAPPINGS = {
    "ETM_SaveImage": ETM_SaveImage,
    "ETM_LoadImageFromLocal": ETM_LoadImageFromLocal,

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ETM_SaveImage": "ETM Save Image",
    "ETM_LoadImageFromLocal": "ETM Load Image From Local",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]