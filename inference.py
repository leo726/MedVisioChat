import os
import sys
sys.path.append('/mnt/sdc/yangling/MedXchat/')
from models.Xchat_qwenvl import XChatModel
from configs.config_qwen import parser
import torch
from PIL import Image
import json
from tqdm import tqdm
    
if __name__ == '__main__':
    ckpt_file = '' #checkpoint.pth
    state_dict = torch.load(ckpt_file, map_location='cpu')['model']
    args = parser.parse_args()
    medxchat = XChatModel(args).to('cuda:0')
    medxchat.load_state_dict(state_dict=state_dict, strict=False)
    
    generate = {'generate':[]}
    
    path = 'vg_VinDr_class_normalize.json'
    data = json.load(open(path))
    for i in tqdm(data['test']):
        # print(i)
        id = i['id']
        path1 = i['conversations'][0]['value'].split('<img>')[1].split('</img>')[0]
        question = i['conversations'][0]['value'].split('<img>')[1].split('</img>\n')[1]
        query = medxchat.tokenizer.from_list_format([
        {'image': path1},
        {'text': question},
    ])
        response, history = medxchat.model.chat(medxchat.tokenizer, query, history=None)
        conver = {}
        conver['from']='generate'
        conver['value']=response
        i['conversations'].append(conver)
        generate['generate'].append(i)
        
        image = medxchat.tokenizer.draw_bbox_on_latest_picture(response, history)
        generate_savepath = './generate/' + path1.split('/')[-1]
        image.save(generate_savepath)
        
        response = i['conversations'][1]['value']
        image = medxchat.tokenizer.draw_bbox_on_latest_picture(response, history)
        gt_savepath = './gt/' + path1.split('/')[-1]
        image.save(gt_savepath)
        
    report_str = json.dumps(generate,indent=4)
    with open('generate.json', 'w') as report_file:
        report_file.write(report_str)
        