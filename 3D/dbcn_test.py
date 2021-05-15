import torch
from dcn.modules.bayes_deform_conv import BayesDeformConv, _DeformConv, BayesDeformConvPack
from dcn.modules.deform_conv import DeformConvPack, DeformConv

deformable_groups = 1
B, inC, inT, inH, inW = 2, 8, 16, 16, 16
outC = 8
kT, kH, kW = 3, 3, 3
sT, sH, sW = 1, 1, 1
pT, pH, pW = 1, 1, 1


def example_bdconv():
    print('============using its own offsets===========')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    dcn = BayesDeformConvPack(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW]).cuda()
    print('input.shape: ', input.shape)
    output = dcn(input)
    target = output.new(*output.size())
    target.data.uniform_(-0.01, 0.01)
    error = (target - output).mean()
    error.backward()
    
    print('output.shape: ', output.shape)

def example_bdconv_offset():
    print('=============using extra offsets============')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    offset = torch.randn(B, kT*kH*kW*3, inT, inH, inW).cuda()
    dcn = BayesDeformConv(inC, outC, kernel_size=[kT, kH, kW], stride=[sT, sH, sW],padding=[pT, pH, pW]).cuda()
    print('input.shape: ', input.shape)
    print('offset.shape: ', offset.shape)
    output = dcn(input, offset)
    target = output.new(*output.size())
    target.data.uniform_(-0.01, 0.01)
    error = (target - output).mean()
    error.backward()
    print('output.shape: ', output.shape)
    
    
def example_dconv():
    print('============using its own offsets===========')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    dcn = DeformConvPack(inC, outC, 3, stride=[sT, sH, sW],padding=[pT, pH, pW]).cuda()
    print('input.shape: ', input.shape)
    output = dcn(input)
    target = output.new(*output.size())
    target.data.uniform_(-0.01, 0.01)
    error = (target - output).mean()
    error.backward()
    print('output.shape: ', output.shape)


def example_dconv_offset():
    print('=============using extra offsets============')
    input = torch.randn(B, inC, inT, inH, inW).cuda()
    offset = torch.randn(B, kT*kH*kW*3, inT, inH, inW).cuda()
    dcn = DeformConv(inC, outC, kernel_size=3, stride=[sT, sH, sW],padding=[pT, pH, pW]).cuda()
    print('input.shape: ', input.shape)
    print('offset.shape: ', offset.shape)
    output = dcn(input, offset)
    target = output.new(*output.size())
    target.data.uniform_(-0.01, 0.01)
    error = (target - output).mean()
    error.backward()
    print('output.shape: ', output.shape)



if __name__ == '__main__':
    
    # bayesian deform in three dimensions
    print('=============DBCN in three dimensions===========\n')
    example_bdconv() # BDCN using its own offsets
    example_bdconv_offset() # BDCN using extra offsets
    print('\n')
    
    print('=============DCN in three dimensions===========\n')
    example_dconv() # DCN using its own offsets
    example_dconv_offset() # DCN using extra offsets
    print('\n')
