#from handle_data import handle_kie_data
from optparse import OptionParser
from kie import my_predict

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-p",action="store",dest="path",default='./pic_data',type=str,help='path')
    parser.add_option("-o",action="store",dest="o_path",default='./output',type=str,help='o_path')
    parser.add_option("-j",action="store",dest="j_path",default='./output',type=str,help='j_path')
    parser.add_option("-m",action="store",dest="m_path",default=None,type=str,help='m_path')
    options, args = parser.parse_args()
    pic_path = options.path
    opath = options.o_path
    jpath = options.j_path
    model_path = options.m_path
    
    #shopping_data_json = handle_kie_data(pic_path) 
    my_predict(model_path,opath)
    