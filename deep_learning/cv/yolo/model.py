import torch
import torch.nn as nn
import numpy as np

from util import predict_transform


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def parse_cfg(cfg_file):
    with open(cfg_file, 'r') as f:
        lines = f.read().split('\n')
        lines = [x for x in lines if len(x) > 0]
        lines = [x for x in lines if x[0] != '#']
        lines = [x.rstrip().lstrip() for x in lines]

        block = {}
        blocks = []

        for line in lines:
            if line[0] == '[':
                if len(block) != 0:
                    blocks.append(block)
                    block = {}
                block['type'] = line[1:-1].rstrip()
            else:
                key, value = line.split('=')
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)

        return blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_norm = int(x['batch_normalize'])
                bias = False
            except:
                batch_norm = 0
                bias = True
            
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module(f'conv_{index}', conv)

            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module(f'batch_norm_{index}', bn)
            
            if activation == 'leaky':
                act_fn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module(f'leaky_{index}', act_fn)
            
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            module.add_module(f'upsample_{index}', upsample)
        
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])

            try:
                end = int(x['layers'][1])
            except:
                end = 0
            
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            
            route = EmptyLayer()
            module.add_module(f'route_{index}', route)

            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
        
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module(f'shortcut_{index}', shortcut)
        
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]
            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[a], anchors[a+1]) for a in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module(f'Detection_{index}', detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return net_info, module_list


class Darknet(nn.Module):
    def __init__(self, cfg_file, device='cuda'):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfg_file)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.device = device
    
    def load_weights(self, weight_file):
        with open(weight_file, 'rb') as f:
            # first 5 values are header info
            # 1. Major version no.
            # 2. Minor version no.
            # 3. Subversion no.
            # 4,5. Images seen by the network
            header = np.fromfile(f, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(f, dtype=np.float32)

            ptr = 0
            for i in range(len(self.module_list)):
                module_type = self.blocks[i+1]['type']
                if module_type == 'convolutional':
                    model = self.module_list[i]
                    try:
                        batch_norm = int(self.blocks[i+1]['batch_normalize'])
                    except:
                        batch_norm = 0
                    conv = model[0]

                    if batch_norm:
                        bn = model[1]

                        #Get the number of weights of Batch Norm Layer
                        num_bn_biases = bn.bias.numel()

                        #Load the weights
                        bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                        ptr += num_bn_biases

                        bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr += num_bn_biases

                        bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr += num_bn_biases

                        bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                        ptr += num_bn_biases

                        #Cast the loaded weights into dims of model weights. 
                        bn_biases = bn_biases.view_as(bn.bias.data)
                        bn_weights = bn_weights.view_as(bn.weight.data)
                        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                        bn_running_var = bn_running_var.view_as(bn.running_var)

                        #Copy the data to model
                        bn.bias.data.copy_(bn_biases)
                        bn.weight.data.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_var)
                    else:
                        num_biases = conv.bias.numel()
                        conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                        ptr = ptr + num_biases
                        
                        conv_biases = conv_biases.view_as(conv.bias.data)
                        conv.bias.data.copy_(conv_biases)
                    
                    num_weights = conv.weight.numel()
                    conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                    ptr = ptr + num_weights
                    conv_weights = conv_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)

    
    def forward(self, x):
        modules = self.blocks[1:]
        outputs = {}
        write = 0
        detections = None

        for i, module in enumerate(modules):
            module_type = (module['type'])
            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[i](x)
            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a) for a in layers]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)
            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i+from_]
            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])
                x = x.data
                x = predict_transform(x,inp_dim,anchors,num_classes,self.device)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)
            
            outputs[i] = x
            
        return detections
