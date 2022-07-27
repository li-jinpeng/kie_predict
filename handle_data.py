import re
import os
from ocr import run

def my_match(text,x1,y1,x2,y2):
    return_list = []
    l = len(text)
    detax = abs(x2-x1)//l
    detay = abs(y2-y1)//l
    flag = 0
    if detax<detay:
        flag = 1
    c = r'\w'
    cur = ''
    text += ' '
    for index,i in enumerate(text):
        if re.match(c,i):
            cur += i
            continue
        if cur!='':
            words = []
            head = index-len(cur)
            end = index
            if flag == 0:
                head_x = x1 + head*detax
                head_y = y1
                end_x = x1 + end*detax
                end_y = y2
                for w_index,j in enumerate(cur):
                    w_x = head_x + w_index*detax
                    w_y = head_y
                    w_x_ = head_x + w_index*detax + detax
                    w_y_ = end_y
                    words.append([j,w_x,w_y,w_x_,w_y_])
            else:
                head_x = x1
                head_y = y1 + head*detay
                end_x = x2
                end_y = y1 + end*detay
                for w_index,j in enumerate(cur):
                    w_x = head_x 
                    w_y = head_y + w_index*detay
                    w_x_ = head_x 
                    w_y_ = end_y + w_index*detay + detay
                    words.append([j,w_x,w_y,w_x_,w_y_])
            
            return_list.append([cur,head_x,head_y,end_x,end_y,words])
            cur = ''
    return return_list

def my_split(filepath):
    filename = ''
    word_list = []
    with open(filepath,'r',encoding='utf-8') as f:
        for line in f:
            item = line.split('\t')
            if filename == '':
                filename = item[0].split('/')[-1]
            if float(item[-1]) < 0.5:
                continue 
            if item[9] == '':
                continue
            x1 = item[1]
            y1 = item[2]
            x2 = item[5]
            y2 = item[6]
            text = item[9]
            word_list.extend(my_match(text,int(x1),int(y1),int(x2),int(y2)))
    return filename,word_list

def my_file(dirpath):
    return_json = {}
    return_json['path'] = dirpath
    return_json['documents'] = []    
    for filepath,dirnames,filenames in os.walk(dirpath):
        for filename in filenames:
            document = {}
            path = os.path.join(filepath,filename)
            name,word_list = my_split(path)
            document['id'] = name
            document['pic_name'] = name.split('/')[-1]
            document['document'] = []
            id = 0
            for word in word_list:
                item = {}
                item['id'] = id
                item['text'] = word[0]
                item['box'] = [word[1],word[2],word[3],word[4]]
                item['words'] = []               
                for w_ in word[5]:
                    w = {}
                    w['text'] = w_[0]
                    w['box'] = [w_[1],w_[2],w_[3],w_[4]]
                    item['words'].append(w)
                document['document'].append(item)
                id += 1
            return_json['documents'].append(document)
    return return_json


def handle_kie_data(pic_dirpath):
    name = pic_dirpath.split('/')[-1]
    if os.path.exists(os.path.join(pic_dirpath ,name))==False:
        run(pic_dirpath,pic_dirpath,8) 
    return my_file(os.path.join(pic_dirpath,name))
    



                
