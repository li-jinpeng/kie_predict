import json
import logging
import os

import datasets

from utils import load_image, merge_bbox, normalize_bbox, simplify_bbox
from transformers import AutoTokenizer



def generate_data(data, filepath):
    for doc in data["documents"]:
        doc["pic_name"] = os.path.join(filepath, doc["pic_name"])
        image, size = load_image(doc["pic_name"])
        document = doc["document"]
        tokenized_doc = {"input_ids": [], "bbox": []}
        for line in document:
            tokenized_inputs = self.tokenizer(
                line["text"],
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
            text_length = 0
            ocr_length = 0
            bbox = []
            last_box = None
            for token_id, offset in zip(tokenized_inputs["input_ids"], tokenized_inputs["offset_mapping"]):
                if token_id == 6:
                    bbox.append(None)
                    continue
                text_length += offset[1] - offset[0]
                tmp_box = []
                while ocr_length < text_length:
                    ocr_word = line["words"].pop(0)
                    ocr_length += len(
                        self.tokenizer._tokenizer.normalizer.normalize_str(ocr_word["text"].strip())
                    )
                    tmp_box.append(simplify_bbox(ocr_word["box"]))
                if len(tmp_box) == 0:
                    tmp_box = last_box
                bbox.append(normalize_bbox(merge_bbox(tmp_box), size))
                last_box = tmp_box
            bbox = [
                [bbox[i + 1][0], bbox[i + 1][1], bbox[i + 1][0], bbox[i + 1][1]] if b is None else b
                for i, b in enumerate(bbox)
            ]
            tokenized_inputs.update({"bbox": bbox})
            for i in tokenized_doc:
                tokenized_doc[i] = tokenized_doc[i] + tokenized_inputs[i]

        chunk_size = 512
        for chunk_id, index in enumerate(range(0, len(tokenized_doc["input_ids"]), chunk_size)):
            item = {}
            for k in tokenized_doc:
                item[k] = tokenized_doc[k][index : index + chunk_size]
            item.update(
                {
                    "id": f"{doc['id']}_{chunk_id}",
                    "image": image,
                }
            )
            return item
def generate_random_str(randomlength=16):
    """
    生成一个指定长度的随机字符串
    """
    random_str = ''
    base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length = len(base_str) - 1
    for i in range(randomlength):
        random_str += base_str[random.randint(0, length)]
    return random_str

def my_data_json(data,pic_dirpath,output):
    j = generate_data(data, pic_dirpath)
    name = 'j'+ generate_random_str(20) + '.json'
    path = open(os.path.join(output,name),'w',encoding='utf-8')
    json.dump(j,path  ,ensure_ascii = False)
    return path
    
