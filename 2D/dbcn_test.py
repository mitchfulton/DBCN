import torch
from dcn.DCN.bayes_deform_conv import BayesDeformConv2D, BayesDeformConvPack2D
from dcn.DCN.dcn_v2 import DCN, DCNv2

deformable_groups = 1
B, inC, inH, inW = 48, 16, 128, 128
outC = 16
kH, kW = 3, 3
sH, sW = 1, 1
pH, pW = 1, 1


def example_bdconv():
    print('============using its own offsets===========')
    input = torch.randn(B, inC, inH, inW).cuda()
    dcn = BayesDeformConvPack2D(inC, outC, kernel_size=[kH, kW], stride=[sH, sW],padding=[pH, pW]).cuda()
    print('input.shape: ', input.shape)
    output = dcn(input)
    target = output.new(*output.size())
    target.data.uniform_(-0.01, 0.01)
    error = (target - output).mean()
    error.backward()
    
    print('output.shape: ', output.shape)

def example_bdconv_offset():
    print('=============using extra offsets============')
    input = torch.randn(B, inC, inH, inW).cuda()
    offset = torch.randn(B, kH*kW*2, inH, inW).cuda()
    mask = torch.randn(B, kH*kW*1, inH, inW).cuda()
    dcn = BayesDeformConv2D(inC, outC, kernel_size=[kH, kW], stride=[sH, sW],padding=[pH, pW]).cuda()
    print('input.shape: ', input.shape)
    print('offset.shape: ', offset.shape)
    output = dcn(input, offset, mask)
    target = output.new(*output.size())
    target.data.uniform_(-0.01, 0.01)
    error = (target - output).mean()
    error.backward()
    print('output.shape: ', output.shape)
    
    
def example_dconv():
    print('============using its own offsets===========')
    input = torch.randn(B, inC, inH, inW).cuda()
    dcn = DCN(inC, outC, 3, stride=[sH, sW],padding=[pH, pW]).cuda()
    print('input.shape: ', input.shape)
    output = dcn(input)
    target = output.new(*output.size())
    target.data.uniform_(-0.01, 0.01)
    error = (target - output).mean()
    error.backward()
    print('output.shape: ', output.shape)


def example_dconv_offset():
    print('=============using extra offsets============')
    input = torch.randn(B, inC, inH, inW).cuda()
    offset = torch.randn(B, kH*kW*2, inH, inW).cuda()
    mask = torch.randn(B, kH*kW*1, inH, inW).cuda()
    dcn = DCNv2(inC, outC, kernel_size=3, stride=[sH, sW],padding=[pH, pW]).cuda()
    print('input.shape: ', input.shape)
    print('offset.shape: ', offset.shape)
    output = dcn(input, offset, mask)
    target = output.new(*output.size())
    target.data.uniform_(-0.01, 0.01)
    error = (target - output).mean()
    error.backward()
    print('output.shape: ', output.shape)



if __name__ == '__main__':
    
    # bayesian deform in three dimensions
    print('=============DBCN in two dimensions===========\n')
    example_bdconv() # BDCN using its own offsets
    example_bdconv_offset() # BDCN using extra offsets
    print('\n')
    
    print('=============DCNv2 in two dimensions===========\n')
    example_dconv() # DCN using its own offsets
    example_dconv_offset() # DCN using extra offsets
    print('\n')
