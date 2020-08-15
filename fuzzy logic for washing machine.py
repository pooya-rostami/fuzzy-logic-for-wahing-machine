from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D


def integral(a, b, mu, sigma):
    if b == 0:
        return 0
    else:
        return quad(Guassian, a, b, args=(mu, sigma))


def Guassian(x, mu, sigma):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))


def x_integral(a, b, mu, sigma):
    if b == 0:
        return 0
    else:
        return quad(x_Guassian, a, b, args=(mu, sigma))


def x_Guassian(x, mu, sigma):
    return x * np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))


def center_of_mass(a, b, mu, sigma):
    if b == 0:
        return 0
    else:
        return (quad(x_Guassian, a, b, args=(mu, sigma)))[0] / (quad(Guassian, a, b, args=(mu, sigma)))[0]




def calc_if(dirtiness, weight, dirtinessFire, weightFire):
    dirtinessSigma = (50 / 3)*0.28
    weightSigma = (8 / 3)*0.28
    maxWeight = 16
    maxDirtiness = 100
    # this if for calculation of weight firing
    if (0 <= weight <= (1 / 6) * maxWeight):
        weightFire[0] = Guassian(weight, 0, weightSigma)
        weightFire[1] = Guassian(weight, (8 / 3), weightSigma)
    elif ((1 / 6) * maxWeight < weight <= (2 / 6) * maxWeight):
        weightFire[1] = Guassian(weight, (8 / 3), weightSigma)
        weightFire[2] = Guassian(weight, (2 * (8 / 3)), weightSigma)
    elif ((2 / 6) * maxWeight < weight <= (3 / 6) * maxWeight):
        weightFire[2] = Guassian(weight, (2 * (8 / 3)), weightSigma)
        weightFire[3] = Guassian(weight, (3 * (8 / 3)), weightSigma)
    elif ((3 / 6) * maxWeight < weight <= (4 / 6) * maxWeight):
        weightFire[3] = Guassian(weight, (3 * (8 / 3)), weightSigma)
        weightFire[4] = Guassian(weight, (4 * (8 / 3)), weightSigma)
    elif ((4 / 6) * maxWeight < weight <= (5 / 6) * maxWeight):
        weightFire[4] = Guassian(weight, (4 * (8 / 3)), weightSigma)
        weightFire[5] = Guassian(weight, (5 * (8 / 3)), weightSigma)
    elif ((5 / 6) * maxWeight < weight <= maxWeight):
        weightFire[5] = Guassian(weight, (5 * (8 / 3)), weightSigma)
        weightFire[6] = Guassian(weight, (6 * (8 / 3)), weightSigma)
    else:
        print("there is a problem in calculating weight firing part")
    # this if for calculation of dirtiness firing
    if (0 <= dirtiness <= (1 / 6) * maxDirtiness):
        dirtinessFire[0] = Guassian(dirtiness, 0, dirtinessSigma)
        dirtinessFire[1] = Guassian(dirtiness, (50 / 3), dirtinessSigma)
    elif ((1 / 6) * maxDirtiness < dirtiness <= (2 / 6) * maxDirtiness):
        dirtinessFire[1] = Guassian(dirtiness, (50 / 3), dirtinessSigma)
        dirtinessFire[2] = Guassian(dirtiness, (2 * (50 / 3)), dirtinessSigma)
    elif ((2 / 6) * maxDirtiness < dirtiness <= (3 / 6) * maxDirtiness):
        dirtinessFire[2] = Guassian(dirtiness, (2 * (50 / 3)), dirtinessSigma)
        dirtinessFire[3] = Guassian(dirtiness, (3 * (50 / 3)), dirtinessSigma)
    elif ((3 / 6) * maxDirtiness < dirtiness <= (4 / 6) * maxDirtiness):
        dirtinessFire[3] = Guassian(dirtiness, (3 * (50 / 3)), dirtinessSigma)
        dirtinessFire[4] = Guassian(dirtiness, (4 * (50 / 3)), dirtinessSigma)
    elif ((4 / 6) * maxDirtiness < dirtiness <= (5 / 6) * maxDirtiness):
        dirtinessFire[4] = Guassian(dirtiness, (4 * (50 / 3)), dirtinessSigma)
        dirtinessFire[5] = Guassian(dirtiness, (5 * (50 / 3)), dirtinessSigma)
    elif ((5 / 6) * maxDirtiness < dirtiness <= (6 / 6) * maxDirtiness):
        dirtinessFire[5] = Guassian(dirtiness, (5 * (50 / 3)), dirtinessSigma)
        dirtinessFire[6] = Guassian(dirtiness, (6 * (50 / 3)), dirtinessSigma)
    else:
        print("there is a problem in calculating dirtiness firing part")
    return weightFire, dirtinessFire


def product(input1, input2):
    return input1 * input2


x_line = []
y_line = []
z_line = []
for i in range(1, 101):
    for j in range(2, 33):

        dirtinessFire = [0, 0, 0, 0, 0, 0, 0]  # [VL,L,LM,M,HM,H,VH]
        weightFire = [0, 0, 0, 0, 0, 0, 0]  # [VL,L,LM,M,HM,H,VH]

        weight = (j / 2)
        dirtiness = i
        x_line.append(dirtiness)
        y_line.append(weight)

        weightFire, dirtinessFire = calc_if(
            dirtiness, weight, dirtinessFire, weightFire)

        # print("dirtiness fire is : " )
        # print(dirtinessFire)
        # print("weight fire is : " )
        # print(weightFire)
        expertRule = [
            [dirtinessFire[0], weightFire[0], 0, product(
                dirtinessFire[0], weightFire[0]), 0],
            [dirtinessFire[0], weightFire[1], 1, product(
                dirtinessFire[0], weightFire[1]), 0],
            [dirtinessFire[0], weightFire[2], 2, product(
                dirtinessFire[0], weightFire[2]), 0],
            [dirtinessFire[0], weightFire[3], 3, product(
                dirtinessFire[0], weightFire[3]), 0],
            [dirtinessFire[0], weightFire[4], 4, product(
                dirtinessFire[0], weightFire[4]), 0],
            [dirtinessFire[0], weightFire[5], 5, product(
                dirtinessFire[0], weightFire[5]), 0],
            [dirtinessFire[0], weightFire[6], 6, product(
                dirtinessFire[0], weightFire[6]), 0],

            [dirtinessFire[1], weightFire[0], 1, product(
                dirtinessFire[1], weightFire[0]), 0],
            [dirtinessFire[1], weightFire[1], 2, product(
                dirtinessFire[1], weightFire[1]), 0],
            [dirtinessFire[1], weightFire[2], 3, product(
                dirtinessFire[1], weightFire[2]), 0],
            [dirtinessFire[1], weightFire[3], 4, product(
                dirtinessFire[1], weightFire[3]), 0],
            [dirtinessFire[1], weightFire[4], 5, product(
                dirtinessFire[1], weightFire[4]), 0],
            [dirtinessFire[1], weightFire[5], 6, product(
                dirtinessFire[1], weightFire[5]), 0],
            [dirtinessFire[1], weightFire[6], 7, product(
                dirtinessFire[1], weightFire[6]), 0],

            [dirtinessFire[2], weightFire[0], 2, product(
                dirtinessFire[2], weightFire[0]), 0],
            [dirtinessFire[2], weightFire[1], 3, product(
                dirtinessFire[2], weightFire[1]), 0],
            [dirtinessFire[2], weightFire[2], 4, product(
                dirtinessFire[2], weightFire[2]), 0],
            [dirtinessFire[2], weightFire[3], 5, product(
                dirtinessFire[2], weightFire[3]), 0],
            [dirtinessFire[2], weightFire[4], 6, product(
                dirtinessFire[2], weightFire[4]), 0],
            [dirtinessFire[2], weightFire[5], 7, product(
                dirtinessFire[2], weightFire[5]), 0],
            [dirtinessFire[2], weightFire[6], 8, product(
                dirtinessFire[2], weightFire[6]), 0],

            [dirtinessFire[3], weightFire[0], 3, product(
                dirtinessFire[3], weightFire[0]), 0],
            [dirtinessFire[3], weightFire[1], 4, product(
                dirtinessFire[3], weightFire[1]), 0],
            [dirtinessFire[3], weightFire[2], 5, product(
                dirtinessFire[3], weightFire[2]), 0],
            [dirtinessFire[3], weightFire[3], 6, product(
                dirtinessFire[3], weightFire[3]), 0],
            [dirtinessFire[3], weightFire[4], 7, product(
                dirtinessFire[3], weightFire[4]), 0],
            [dirtinessFire[3], weightFire[5], 8, product(
                dirtinessFire[3], weightFire[5]), 0],
            [dirtinessFire[3], weightFire[6], 9, product(
                dirtinessFire[3], weightFire[6]), 0],

            [dirtinessFire[4], weightFire[0], 4, product(
                dirtinessFire[4], weightFire[0]), 0],
            [dirtinessFire[4], weightFire[1], 5, product(
                dirtinessFire[4], weightFire[1]), 0],
            [dirtinessFire[4], weightFire[2], 6, product(
                dirtinessFire[4], weightFire[2]), 0],
            [dirtinessFire[4], weightFire[3], 7, product(
                dirtinessFire[4], weightFire[3]), 0],
            [dirtinessFire[4], weightFire[4], 8, product(
                dirtinessFire[4], weightFire[4]), 0],
            [dirtinessFire[4], weightFire[5], 9, product(
                dirtinessFire[4], weightFire[5]), 0],
            [dirtinessFire[4], weightFire[6], 10, product(
                dirtinessFire[4], weightFire[6]), 0],

            [dirtinessFire[5], weightFire[0], 5, product(
                dirtinessFire[5], weightFire[0]), 0],
            [dirtinessFire[5], weightFire[1], 6, product(
                dirtinessFire[5], weightFire[1]), 0],
            [dirtinessFire[5], weightFire[2], 7, product(
                dirtinessFire[5], weightFire[2]), 0],
            [dirtinessFire[5], weightFire[3], 8, product(
                dirtinessFire[5], weightFire[3]), 0],
            [dirtinessFire[5], weightFire[4], 9, product(
                dirtinessFire[5], weightFire[4]), 0],
            [dirtinessFire[5], weightFire[5], 10, product(
                dirtinessFire[5], weightFire[5]), 0],
            [dirtinessFire[5], weightFire[6], 11, product(
                dirtinessFire[5], weightFire[6]), 0],

            [dirtinessFire[6], weightFire[0], 6, product(
                dirtinessFire[6], weightFire[0]), 0],
            [dirtinessFire[6], weightFire[1], 7, product(
                dirtinessFire[6], weightFire[1]), 0],
            [dirtinessFire[6], weightFire[2], 8, product(
                dirtinessFire[6], weightFire[2]), 0],
            [dirtinessFire[6], weightFire[3], 9, product(
                dirtinessFire[6], weightFire[3]), 0],
            [dirtinessFire[6], weightFire[4], 10, product(
                dirtinessFire[6], weightFire[4]), 0],
            [dirtinessFire[6], weightFire[5], 11, product(
                dirtinessFire[6], weightFire[5]), 0],
            [dirtinessFire[6], weightFire[6], 12, product(
                dirtinessFire[6], weightFire[6]), 0],

        ]

        # print(expertRule)
        outputMu = (2 * (10 ** 4)) / 12
        outputSigma = ((2 * (10 ** 4)) / 12)*0.28
        lowerBound = 0
        upperBound = (2 * (10 ** 4))
        for rule in expertRule:
            if (rule[2] == 0):
                lowerBound = 0
                upperBound = 1 * outputMu
                tmp = center_of_mass(lowerBound, upperBound, 0, outputSigma)
                rule[4] = tmp
            elif (rule[2] == 1):
                rule[4] = 1 * outputMu
            elif (rule[2] == 2):
                rule[4] = 2 * outputMu
            elif (rule[2] == 3):
                rule[4] = 3 * outputMu
            elif (rule[2] == 4):
                rule[4] = 4 * outputMu
            elif (rule[2] == 5):
                rule[4] = 5 * outputMu
            elif (rule[2] == 6):
                rule[4] = 6 * outputMu
            elif (rule[2] == 7):
                rule[4] = 7 * outputMu
            elif (rule[2] == 8):
                rule[4] = 8 * outputMu
            elif (rule[2] == 9):
                rule[4] = 9 * outputMu
            elif (rule[2] == 10):
                rule[4] = 10 * outputMu
            elif (rule[2] == 11):
                rule[4] = 11 * outputMu
            elif (rule[2] == 12):
                lowerBound = 11 * outputMu
                upperBound = 12 * outputMu
                tmp = center_of_mass(lowerBound, upperBound, outputMu * 12, outputSigma)
                rule[4] = tmp
            else:
                print("we have a problem in integral part")

        # for rule in expertRule:
        # print(rule)
        sum1 = 0.0
        sum2 = 0.0
        for rule in expertRule:
            sum1 += (rule[3] * rule[4])
            sum2 += rule[3]

        output = sum1 / sum2
        # print("output is : ")
        # print(output)
        z_line.append(output)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_line, y_line, z_line)

ax.set_xlabel('Dirtiness')
ax.set_ylabel('Weight')
ax.set_zlabel('RPM')
plt.show()
