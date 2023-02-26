class QueryContext:
    def __init__(self, stream_id, model, input_list,scale=None):
        # input  是tensor list
        self.has_deadline=False
        self.stream_id=stream_id
        self.model=model
        self.input_list=input_list
        # shape是属性  size是方法
        # print(len(input_list),"输入长度")

        self.input_res=input_list[0].shape
        print(self.input_res)


        # 如果是sr model,query中需要包含放大倍数
        self.scale=scale

        # 这个不是query中需要指定的，而是调度器需要决策的
        self.gpu_idx=0