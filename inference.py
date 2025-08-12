import os
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import ast
import json
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import predict, annotate
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

def get_args_parser():
    parser = argparse.ArgumentParser('Visual grounding', add_help=False)

    # dataset parameters
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--model_dir', default=r'./weights/groundingdino_swint_ogc.pth') # path/to/your/local/model
    parser.add_argument('--json_path', default='./VG-RS-question.json')
    parser.add_argument('--save_results', default=False, action='store_true')
    parser.add_argument('--json_save_path',
                        default='./predict_grounding_full_3b.json')
    return parser

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def read_json_and_extract_fields(file_path='VG-RS-question.json'):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def load_model(model_config_path, model_checkpoint_path):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def main(args):
    
    model = load_model("config.py", args.model_dir)

    # fix the seed for reproducibility
    seed = args.seed
    print('seed')
    print(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    img_path = r'./images/'
    data_infer = read_json_and_extract_fields(args.json_path)
    data_infer = data_infer[:]
    batch_size = args.batch_size
    # default processer
    # min_pixels = 256 * 28 * 28
    max_pixels = 2560 * 2560

    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25

    content_list = []
    for i in tqdm(range(len(data_infer)//batch_size)):
        # if i/(len(data_infer)//batch_size) <= 0.72:
        #     continue
        for i_batch in range(batch_size):
            image_path = data_infer[i * batch_size + i_batch].get('image_path').replace('\\', '/')
            image_pil, image = load_image(image_path)
            text_query = data_infer[i * batch_size + i_batch].get('question')

            W, H = image_pil.size

            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_query, BOX_TRESHOLD, TEXT_TRESHOLD
            )

            for j, box in enumerate(boxes_filt):
                box = box * torch.Tensor([W, H, W, H])
                # from xywh to xyxy
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]

                x0, y0, x1, y1 = box
                #x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                assert x0 < x1 and y0 < y1, f"Invalid box coordinates: {x0}, {y0}, {x1}, {y1}"

                content = {
                    "image_path": image_path.replace('/', '\\'),
                    'question': text_query,
                    "result": [[x0.item(), y0.item()], [x1.item(), y1.item()]]
                    }

                content_list.append(content)

            pred_dict = {
                "boxes": boxes_filt,
                "size": [H, W],  # H,W
                "labels": pred_phrases,
            }

            if args.save_results:
                if not os.path.exists("./results"):
                    os.makedirs("./results")
                image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
                image_with_box.save(f"./results/{i_batch+batch_size*i}_{text_query.replace('/','')}.jpg")
        
    with open(args.json_save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(content_list, ensure_ascii=False, indent=2) + '\n')
    return


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

    return boxes_filt, pred_phrases

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Infer result', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)