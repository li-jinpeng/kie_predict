# Lint as: python3
import json
import logging
import os

import datasets

from layoutlmft.data.utils import load_image, merge_bbox, normalize_bbox, simplify_bbox
from transformers import AutoTokenizer


class SHOPPINGDATA(datasets.GeneratorBasedBuilder):

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    def _info(self):
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "id": datasets.Value("string"),
                        "input_ids": datasets.Sequence(datasets.Value("int64")),
                        "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                        "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    }
                ),
                supervised_keys=None,
            )
    def _split_generators(self, dl_manager):
        test_files_for_many_langs = ['./output/data.json','./pic_data']
        return [
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": test_files_for_many_langs}),
        ]

    def _generate_examples(self, filepaths):
        with open('/home/lijinpeng/kie_predict/output/data.json', "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc in data["documents"]:
            doc["pic_path"] = os.path.join('/home/lijinpeng/kie_predict/pic_data', doc['pic_name'])
            image, size = load_image(doc['pic_path'])
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
                yield f"{doc['id']}_{chunk_id}", item