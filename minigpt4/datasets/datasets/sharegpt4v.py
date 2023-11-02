import os
from PIL import Image
import webdataset as wds
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset

class ShareGPT4VDataset(CaptionDataset):

    def __getitem__(self, index):

        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann["conversations"][1]["value"]

        return {
            "image": image,
            "answer": caption,
            "image_id": self.img_ids[ann["id"]],
        }