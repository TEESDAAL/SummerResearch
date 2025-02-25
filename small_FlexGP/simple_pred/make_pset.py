import numpy as np
from deap import gp
import random
from functools import partial
from simple_pred.function_set import (
    root_con, root_conVector2, root_conVector3, global_hog_small,
    all_lbp, all_sift, maxP, gau, gauD, lbp, hog_feature, gab, laplace,
    gaussian_Laplace1, gaussian_Laplace2, sobelxy, sobelx, sobely, medianf,
    minf, maxf, meanf, mixconadd, mixconsub, relu, sqrt
)
from simple_pred.data_types import (
        Img, Vector, Std, Order, Orientation, Frequency, Weight, KernelSize,
        Vector1, Img1, Int1, Int2, Int3, Float1, Float2, Float3
)


# def pset():
#     pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector, prefix='Image')
#     #feature concatenation
#     pset.addPrimitive(root_conVector2, [Img, Img], Vector, name='Root2')
#     pset.addPrimitive(root_conVector3, [Img, Img, Img], Vector, name='Root3')
#
#     for n in [2, 3, 4]:
#         pset.addPrimitive(root_con, [Vector]*n, Vector, name=f"Roots{n}")
#
#     ##feature extraction
#     for feature_extractor, name in [(global_hog_small, "Global_HOG"), (all_lbp, "Global_uLBP"), (all_sift, "Global_SIFT")]:
#         pset.addPrimitive(feature_extractor, [Img], Vector, name=name)
#
#     # pooling
#     pset.addPrimitive(maxP, [Img, KernelSize, KernelSize], Img1, name='MaxP')
#
#     #filtering
#     pset.addPrimitive(gau, [Img, Std], Img1, name='Gau')
#     pset.addPrimitive(gauD, [Img, Std, Order, Order], Img1, name='GauD')
#     pset.addPrimitive(gab, [Img, Orientation, Frequency], Img1, name='GaborF')
#     image_processing_functions = [
#         (laplace, 'Lap'), (gaussian_Laplace1, 'LoG1'),
#         (gaussian_Laplace2, 'LoG2'), (sobelxy, 'Sobel'),
#         (sobelx, 'SobelX'),(sobely, 'SobelY'), (medianf, 'Med'),
#         (meanf, 'Mean'), (minf, 'Min'), (maxf, 'Max'), (lbp, 'LBP'),
#         (hog_feature,'HoG'), (sqrt, 'Sqrt'), (relu, 'ReLU')
#     ]
#     for img_processing_function, name in image_processing_functions:
#         pset.addPrimitive(img_processing_function, [Img], Img1, name=f'{name}')
#
#     pset.addPrimitive(mixconadd, [Img, Weight, Img, Weight], Img1, name='W_Add')
#     pset.addPrimitive(mixconsub, [Img, Weight, Img, Weight], Img1, name='W_Sub')
#
#     # Terminals
#     pset.renameArguments(ARG0='Image')
#     pset.addEphemeralConstant('Sigma', partial(random.randint, 1, 3), Std)
#     pset.addEphemeralConstant('Order', partial(random.randint, 0, 2), Order)
#     pset.addEphemeralConstant('Theta', partial(random.randint, 0, 7), Orientation)
#     pset.addEphemeralConstant('Frequency', partial(random.randint, 0, 5), Frequency)
#     pset.addEphemeralConstant('n', gen_weight, Weight)
#     pset.addEphemeralConstant('KernelSize', partial(random.randrange, 2, 5, 2), KernelSize)
#
#     return pset
def pset():
    pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector1, prefix='Image')
    # feature concatenation
    pset.addPrimitive(root_con, [Vector1, Vector1], Vector1, name='Root')
    pset.addPrimitive(root_conVector2, [Img1, Img1], Vector1, name='Root2')
    pset.addPrimitive(root_conVector3, [Img1, Img1, Img1], Vector1, name='Root3')
    pset.addPrimitive(root_con, [Vector, Vector], Vector1, name='Roots2')
    pset.addPrimitive(root_con, [Vector, Vector, Vector], Vector1, name='Roots3')
    pset.addPrimitive(root_con, [Vector, Vector, Vector, Vector], Vector1, name='Roots4')
    ##feature extraction
    pset.addPrimitive(global_hog_small, [Img1], Vector, name='Global_HOG')
    pset.addPrimitive(all_lbp, [Img1], Vector, name='Global_uLBP')
    pset.addPrimitive(all_sift, [Img1], Vector, name='Global_SIFT')
    pset.addPrimitive(global_hog_small, [Img], Vector, name='FGlobal_HOG')
    pset.addPrimitive(all_lbp, [Img], Vector, name='FGlobal_uLBP')
    pset.addPrimitive(all_sift, [Img], Vector, name='FGlobal_SIFT')
    # pooling
    pset.addPrimitive(maxP, [Img1, Int3, Int3], Img1, name='MaxPF')
    # filtering
    pset.addPrimitive(gau, [Img1, Int1], Img1, name='GauF')
    pset.addPrimitive(gauD, [Img1, Int1, Int2, Int2], Img1, name='GauDF')
    pset.addPrimitive(gab, [Img1, Float1, Float2], Img1, name='GaborF')
    pset.addPrimitive(laplace, [Img1], Img1, name='LapF')
    pset.addPrimitive(gaussian_Laplace1, [Img1], Img1, name='LoG1F')
    pset.addPrimitive(gaussian_Laplace2, [Img1], Img1, name='LoG2F')
    pset.addPrimitive(sobelxy, [Img1], Img1, name='SobelF')
    pset.addPrimitive(sobelx, [Img1], Img1, name='SobelXF')
    pset.addPrimitive(sobely, [Img1], Img1, name='SobelYF')
    pset.addPrimitive(medianf, [Img1], Img1, name='MedF')
    pset.addPrimitive(meanf, [Img1], Img1, name='MeanF')
    pset.addPrimitive(minf, [Img1], Img1, name='MinF')
    pset.addPrimitive(maxf, [Img1], Img1, name='MaxF')
    pset.addPrimitive(lbp, [Img1], Img1, name='LBPF')
    pset.addPrimitive(hog_feature, [Img1], Img1, name='HoGF')
    pset.addPrimitive(mixconadd, [Img1, Float3, Img1, Float3], Img1, name='W_AddF')
    pset.addPrimitive(mixconsub, [Img1, Float3, Img1, Float3], Img1, name='W_SubF')
    pset.addPrimitive(sqrt, [Img1], Img1, name='SqrtF')
    pset.addPrimitive(relu, [Img1], Img1, name='ReLUF')
    # pooling
    pset.addPrimitive(maxP, [Img, Int3, Int3], Img1, name='MaxP')
    # filtering
    pset.addPrimitive(gau, [Img, Int1], Img1, name='Gau')
    pset.addPrimitive(gauD, [Img, Int1, Int2, Int2], Img1, name='GauD')
    pset.addPrimitive(gab, [Img, Float1, Float2], Img1, name='Gabor')
    pset.addPrimitive(laplace, [Img], Img1, name='Lap')
    pset.addPrimitive(gaussian_Laplace1, [Img], Img1, name='LoG1')
    pset.addPrimitive(gaussian_Laplace2, [Img], Img1, name='LoG2')
    pset.addPrimitive(sobelxy, [Img], Img1, name='Sobel')
    pset.addPrimitive(sobelx, [Img], Img1, name='SobelX')
    pset.addPrimitive(sobely, [Img], Img1, name='SobelY')
    pset.addPrimitive(medianf, [Img], Img1, name='Med')
    pset.addPrimitive(meanf, [Img], Img1, name='Mean')
    pset.addPrimitive(minf, [Img], Img1, name='Min')
    pset.addPrimitive(maxf, [Img], Img1, name='Max')
    pset.addPrimitive(lbp, [Img], Img1, name='LBP_F')
    pset.addPrimitive(hog_feature, [Img], Img1, name='HOG_F')
    pset.addPrimitive(mixconadd, [Img, Float3, Img, Float3], Img1, name='W_Add')
    pset.addPrimitive(mixconsub, [Img, Float3, Img, Float3], Img1, name='W_Sub')
    pset.addPrimitive(sqrt, [Img], Img1, name='Sqrt')
    pset.addPrimitive(relu, [Img], Img1, name='ReLU')
    # Terminals
    pset.renameArguments(ARG0='Image')
    pset.addEphemeralConstant('Sigma', partial(random.randint, 1, 3), Int1)
    pset.addEphemeralConstant('Order', partial(random.randint, 0, 2), Int2)
    pset.addEphemeralConstant('Theta', partial(random.randint,0, 7), Float1)
    pset.addEphemeralConstant('Frequency', partial(random.randint,0, 5), Float2)
    pset.addEphemeralConstant('n', gen_weight, Float3)
    pset.addEphemeralConstant('KernelSize', partial(random.randrange,2, 5, 2), Int3)

    return pset


def gen_weight():
    return round(random.random(), 3)
