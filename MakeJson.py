import json
import os
import glob
import re
import numpy as np

def data_iter(data_list,step):
    data_nums = len(data_list)
    indicts = list(range(data_nums))

    for i in range(0,data_nums,step):
        batch_indicts = np.array(indicts[i:min(i+step,data_nums)])
        yield batch_indicts
def get_json_list(render_list,query_list,cate):
    data_cate_list = []
    render_list_len = len(render_list)
    query_list_len = len(query_list)
    if query_list_len > render_list_len:
        step = query_list_len // render_list_len
        j = 0
        for i, query_indicts in enumerate(data_iter(query_list, step)):
            render_indict = i
            if i >= render_list_len:
                render_indict = j
                j += 1
            for query_indict in query_indicts:
                dict = {}
                dict['query_img'] = query_list[query_indict]
                dict['render_img'] = render_list[render_indict]
                dict['category'] = cate
                data_cate_list.append(dict)
        return data_cate_list
    else:
        step = render_list_len // query_list_len
        j = 0
        for i, query_indicts in enumerate(data_iter(query_list, step)):
            render_indict = i
            if i >= render_list_len:
                render_indict = j
                j += 1
            for query_indict in query_indicts:
                dict = {}
                dict['query_img'] = query_list[query_indict]
                dict['render_img'] = render_list[render_indict]
                dict['category'] = cate
                data_cate_list.append(dict)
        return data_cate_list


def sort_files_by_number(files):
    #将文件名按数字和非数字分组
    groups = []
    for file in files:
        groups.append((file,re.split(r'(\d+)',file)))
        #将分组后的文件名列表按数字大小排序
    groups.sort(key=lambda x : [int(s) if s.isdigit() else s for s in x[1]])
    #返回排序后的文件名列表
    return [file for (file,parts) in groups]
if __name__ == '__main__':
    data_dict = {}
    label_list = os.listdir('./data/render/test')
    for i , name in enumerate(label_list):
        data_dict[name] = i
    with open('data_dict.json','w') as json_file:
        json.dump(data_dict,json_file,indent=4)
    datas_list = []
    for key in label_list:
        modelfilenames = glob.glob("%s/%s/%s/*.png"%('./data/render','test',key))
        sorted_model_filenames = sort_files_by_number(modelfilenames)
        packed_model_filenames = np.array(sorted_model_filenames)
        model_nums = len(packed_model_filenames) // 18
        packed_filenames = packed_model_filenames.reshape(model_nums,18)

        render_list = packed_filenames.tolist()
        query_list = glob.glob("%s/%s/%s/%s/*.png"%('./data/image','SHREC14LSSTB_SKETCHES',key,'test'))
        data_items = get_json_list(render_list,query_list,key)
        datas_list.append(data_items)
    datas_list = [i for j in datas_list for i in j]

    data = {}
    data['data_list'] = datas_list
    with open('data_train.json',"w") as json_file:
        json.dump(data,json_file,indent=4)

