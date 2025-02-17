import numpy as np
from deap import gp
import random
from functools import partial
from simple_pred.function_set import (
    root_con_vector, maxP,  relu, sqrt, ZeromaxP, conv_filters, weighted_sub, weighted_add
)
from simple_pred.data_types import (
    Img, ConvolvedImg, ConvolvedPooledImg,
    Vector,  Weight, KernelSize, Filter,
)


def pset():
    pset = gp.PrimitiveSetTyped('MAIN',[Img], Vector, prefix='Image')
    # Concatination Layer
    pset.addPrimitive(root_con_vector, [Vector, Vector], Vector, name='vec_concat')
    for n in [2, 3, 4]:
        pset.addPrimitive(root_con_vector, [ConvolvedPooledImg]*n, Vector, name=f"concat{n}")

    # Convolution
    convolution_operators = [
        (conv_filters, [Img, Filter]), (weighted_sub, [Img, Weight, Img, Weight]),
        (weighted_add, [Img, Weight, Img, Weight]), (relu, [Img]), (sqrt, [Img]),
        (abs, [Img])
    ]
    for func, input_types in convolution_operators:
        pset.addPrimitive(func, input_types, ConvolvedImg)
        pset.addPrimitive(func, alter_input_types(input_types, {Img: ConvolvedImg}) , ConvolvedImg)
        pset.addPrimitive(func, alter_input_types(input_types, {Img: ConvolvedPooledImg}) , ConvolvedPooledImg)


    # Pooling
    for _ in range(3):
        pset.addPrimitive(maxP, [ConvolvedImg, KernelSize, KernelSize], ConvolvedPooledImg)

    pset.addPrimitive(maxP, [ConvolvedPooledImg, KernelSize, KernelSize], ConvolvedPooledImg)

    # Conv and Pooling
    pset.addPrimitive(ZeromaxP, [ConvolvedPooledImg, KernelSize, KernelSize], ConvolvedPooledImg)

    for filter_size in [3, 5, 7]:
        pset.addEphemeralConstant(f"filter{filter_size}", partial(gen_filter, filter_size), Filter)

    pset.addEphemeralConstant("weight", random.random, Weight)
    pset.addEphemeralConstant("kernelSize", partial(random.randint, 2, 4), KernelSize)

    return pset


def alter_input_types(input_types: list[type], replacements: dict[type, type]) -> list[type]:
    return [replacements.get(type_, type_) for type_ in input_types]

def gen_filter(size: int) -> list[list[int]]:
    return [
        [random.randint(-5, 5) for _ in range(size)]
        for _ in range(size)
    ]
