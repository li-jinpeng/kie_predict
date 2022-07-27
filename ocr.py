import os
import sys
from optparse import OptionParser
from os.path import join
import time
import grpc
import cv2
import json
import numpy as np
from tqdm import tqdm
from model_serving_pb2 import PredictRequest
from model_serving_pb2 import BatchPredictRequest
from model_serving_pb2 import ModelServingStub
from model_serving_pb2 import SUCCESS, ERROR
from google.protobuf import json_format
from kess.framework import ClientOption, GrpcClient

# det service
addr_det = '10.28.215.32'
port_det = 21010
# rec service horizontal lstm
addr_rec_h = '10.40.57.29'
port_rec_h = 21033
#addr_rec_h = '10.22.137.151'
#port_rec_h = 21058
rec_batchsize = 16
# rec service vertical
addr_rec_v = '10.96.184.169'
port_rec_v = 22697


def run(file_path,ofile_path, batch_size):
    det_stub = ModelServingStub(grpc.insecure_channel(addr_det + ":" + str(port_det)))
    rec_h_stub = ModelServingStub(grpc.insecure_channel(addr_rec_h + ":" + str(port_rec_h)))
    rec_v_stub = ModelServingStub(grpc.insecure_channel(addr_rec_v + ":" + str(port_rec_v)))

    #服务名调用
    #det_stub = GrpcClient(ClientOption(biz_def = 'mmu', grpc_service_name = 'grpc_mmu_videoOcrDetectionV6',grpc_stub_class = ModelServingStub))
    #rec_h_stub = GrpcClient(ClientOption(biz_def = 'mmu', grpc_service_name = 'grpc_mmu_videoOcrRecognitionV6',grpc_stub_class = ModelServingStub))
    #rec_v_stub = GrpcClient(ClientOption(biz_def = 'mmu', grpc_service_name = 'grpc_mmu_ocrRecognitionVerticalVideo',grpc_stub_class = ModelServingStub))
    '''
    f_pid_input = open('/home/tangyejun/code_lib/video_ocr/unenough_pid.txt','r')
    f_pid_dict = {}
    lines = f_pid_input.readlines()
    for line in lines:
        line = line.strip()
        f_pid_dict[line] = 1
    print('Len of input:', len(f_pid_dict))
    '''
    result = list()
    file_list = list()
    '''
    for target in os.listdir(file_path):
        d = os.path.join(file_path, target)
        if not os.path.isdir(d):
            continue
        if target not in f_pid_dict:
            continue
        for root, dirs, files in os.walk(d):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                    #if file!='33911860696_single.jpg':
                    file_list.append(join(root, file))
            break
    '''
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                file_list.append(join(root, file))
                
        break
        
    if not os.path.exists(ofile_path):
        os.makedirs(ofile_path)

    #outfile = open('./LM_json/search_GSB_1229.txt', 'w')
    #检测部分，将所有检测的结果存到辞典里
    #这里检测结果需要组batch，组batch的逻辑是按照8个视频的所有帧组batch
    #先将结果输入到辞典，然后建立一个request_id的宽度辞典，进行排序

    for x in tqdm(range(0, len(file_list), batch_size)):
        file_batch = file_list[x: x + batch_size]
        requests = BatchPredictRequest(id="testimg_" + str(x))
        for file in file_batch:
            with open(file, mode='rb') as fd:
                request = PredictRequest(id=file)
                request.media.data = fd.read()
                requests.requests.append(request)
        responses = det_stub.BatchPredict(requests)
        results = responses.result

        #横排辞典
        norm_h_dict = {}
        quads_dict = {}
        points_dict = {}
        #竖排辞典
        norm_v_dict = {}
        quads_v_dict = {}
        points_v_dict = {}

        for request in requests.requests:
            result = results[request.id]
            num = result.meta.int32_val  # 检测出的数量
            quads = result.meta.int32_quads  # 水平矩形 x,y,w,h
            points = result.meta.int32_arrays  # 四点坐标 x1,y1,x2,y2,x3,y3,x4,y4
            rrecs = result.meta.float_arrays  # 倾斜矩形 center.x,center.y,w,h,angle
            probs = result.meta.float_array.float_elems  #conf
            is_horizontal = result.meta.bool_array.bool_elems
            classes = result.meta.str_array.str_elems

            for i in range(num):
                #竖排部分
                if is_horizontal[i] == True:
                    norm_v_dict[str(request.id)+'#'+str(i)] = float(quads[i].fourth)/float(quads[i].third)
                    quads_v_dict[str(request.id)+'#'+str(i)] = quads[i]
                    points_v_dict[str(request.id)+'#'+str(i)] = points[i]
                #横排部分
                else:
                    norm_h_dict[str(request.id)+'#'+str(i)] = float(quads[i].third)/float(quads[i].fourth)
                    quads_dict[str(request.id)+'#'+str(i)] = quads[i]
                    points_dict[str(request.id)+'#'+str(i)] = points[i]
        #排序
        #list_sorted = sorted(norm_h_dict.items(), key = lambda x:x[1])
        list_sorted = list(norm_h_dict.items())
        #list_v_sorted = sorted(norm_v_dict.items(), key = lambda x:x[1])
        list_v_sorted = list(norm_v_dict.items())
        #调用识别部分
        #横排部分
        for i in range(0, (len(list_sorted)-1)//rec_batchsize + 1):
            rec_requests = BatchPredictRequest(id="recogimg_" + str(i))
            keys = list_sorted[i*rec_batchsize:(i+1)*rec_batchsize]
            file_idx_dict = {}
            for filename_idx in keys:
                filename, idx = filename_idx[0].split('#')
                if filename in file_idx_dict:
                    file_idx_dict[filename].append(idx)
                else:
                    file_idx_dict[filename] = [idx]

            for filename in file_idx_dict:
                rec_request = PredictRequest(id=filename)
                with open(filename, mode='rb') as fd:
                    rec_request.media.data = fd.read()
                rec_request.meta.int32_val = 0
                for key in file_idx_dict[filename]:
                    quad = quads_dict[filename+'#'+key]
                    points = points_dict[filename+'#'+key]
                    rec_request.meta.int32_val = rec_request.meta.int32_val + 1
                    rec_request.meta.int32_quads.append(quad)
                    rec_request.meta.int32_arrays.append(points)
                rec_requests.requests.append(rec_request)
            rec_response = rec_h_stub.BatchPredict(rec_requests)
            if rec_response.status == ERROR:
                print("recogimg_" + str(i), 'error')
                continue
            for filename in file_idx_dict:
                #创建输出文件夹
        
                path = os.path.join(ofile_path, filename.split('/')[-2])
                if not os.path.exists(path):
                    os.mkdir(path)
                ofile = open(os.path.join(path,filename.split('/')[-1].replace('.jpg','.txt').replace('.png','.txt').replace('.jpeg','.txt')),'a')
         
                for j in range(rec_response.result[filename].meta.int32_val):
                    #print(filename+'#'+file_idx_dict[filename][j])
                    #print(rec_response.result[filename].meta.str_array.str_elems[j])
                    points_temp = points_dict[filename+'#'+file_idx_dict[filename][j]]
                    points_str = str(points_temp.int32_elems[0])
                    for x in range(1,len(points_temp.int32_elems)):
                        points_str = points_str + '\t' + str(points_temp.int32_elems[x])
                    #print(filename + '\t' +points_str + '\t' + rec_response.result[filename].meta.str_array.str_elems[j]+'\t' + 'horizontal')
                    print(filename + '\t' +points_str + '\t' + rec_response.result[filename].meta.str_array.str_elems[j] + '\t' + str(rec_response.result[filename].meta.float_array.float_elems[j]),file=ofile)
                ofile.close()
        #竖排部分
        for i in range(0, (len(list_v_sorted)-1)//rec_batchsize + 1):
            rec_v_requests = BatchPredictRequest(id="recogimg_" + str(i))
            keys = list_v_sorted[i*rec_batchsize:(i+1)*rec_batchsize]
            file_idx_dict = {}
            for filename_idx in keys:
                filename, idx = filename_idx[0].split('#')
                if filename in file_idx_dict:
                    file_idx_dict[filename].append(idx)
                else:
                    file_idx_dict[filename] = [idx]

            for filename in file_idx_dict:
                rec_v_request = PredictRequest(id=filename)
                with open(filename, mode='rb') as fd:
                    rec_v_request.media.data = fd.read()
                rec_v_request.meta.int32_val = 0
                for key in file_idx_dict[filename]:
                    quad = quads_v_dict[filename+'#'+key]
                    points = points_v_dict[filename+'#'+key]
                    rec_v_request.meta.int32_val = rec_v_request.meta.int32_val + 1
                    rec_v_request.meta.int32_quads.append(quad)
                    rec_v_request.meta.int32_arrays.append(points)
                rec_v_requests.requests.append(rec_v_request)
                rec_v_response = rec_v_stub.BatchPredict(rec_v_requests)
            if rec_v_response.status == ERROR:
                print("recogimg_" + str(i), 'error')
                continue
            for filename in file_idx_dict:
                #竖排 写入文件部分
               
                path = os.path.join(ofile_path, filename.split('/')[-2])
                if not os.path.exists(path):
                    os.mkdir(path)
                ofile = open(os.path.join(path,filename.split('/')[-1].replace('.jpg','.txt').replace('.png','.txt').replace('.jpeg','.txt')),'a')
               
                for j in range(rec_v_response.result[filename].meta.int32_val):
                    #print(filename+'#'+file_idx_dict[filename][j])
                    #print('#vertical#',rec_v_response.result[filename].meta.str_array.str_elems[j])
                    points_temp = points_v_dict[filename+'#'+file_idx_dict[filename][j]]
                    points_str = str(points_temp.int32_elems[0])
                    for x in range(1,len(points_temp.int32_elems)):
                        points_str = points_str + '\t' + str(points_temp.int32_elems[x])                    
                    print(filename + '\t' +points_str + '\t' + rec_v_response.result[filename].meta.str_array.str_elems[j] + '\t' + str(rec_v_response.result[filename].meta.float_array.float_elems[j]),file=ofile)
                ofile.close()   