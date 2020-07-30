import matplotlib.pyplot as plt

CART_sens = {'max_depth = 1': 0.7075645756457565, 'max_depth = 2': 0.7075645756457565, 'max_depth = 3': 0.6848281642917016, 'max_depth = 4': 0.7167338709677419, 'max_depth = 5': 0.681084162964692,
             'max_depth = 6': 0.6120210446589671, 'max_depth = 7': 0.548042308494594, 'max_depth = 8': 0.5830469947998613, 'max_depth = 9': 0.5836993751962348, 'max_depth = 10': 0.5704718123303881,
             'max_depth = 11': 0.5667153054825267, 'max_depth = 12': 0.5573394045880148, 'max_depth = 13': 0.5460280601150955, 'max_depth = 14': 0.5371509117244047, 'max_depth = 15': 0.538250631621157,
             'max_depth = 16': 0.5338191088085892, 'max_depth = 17': 0.5431255373777171, 'max_depth = 18': 0.5326840196200495, 'max_depth = 19': 0.5412253961344842, 'max_depth = 20': 0.5382306488025898,
             'max_depth = 21': 0.5414353567181838, 'max_depth = 22': 0.5475267558946374, 'max_depth = 23': 0.5406215091612951, 'max_depth = 24': 0.5378144769693309, 'max_depth = 25': 0.5362581639901902,
             'max_depth = 26': 0.5371149677519214, 'max_depth = 27': 0.5341910232067091, 'max_depth = 28': 0.5353567697100761, 'max_depth = 29': 0.5418057087678096}
CART_spec = {'max_depth = 1': 0.6862326574172892, 'max_depth = 2': 0.6862326574172892, 'max_depth = 3': 0.7053140096618357, 'max_depth = 4': 0.6598639455782311, 'max_depth = 5': 0.6565529495414513,
             'max_depth = 6': 0.6451632527780117, 'max_depth = 7': 0.5055347814165908, 'max_depth = 8': 0.5603387664900324, 'max_depth = 9': 0.5549427180216829, 'max_depth = 10': 0.5309449965335329,
             'max_depth = 11': 0.5240184328028766, 'max_depth = 12': 0.5130560278256788, 'max_depth = 13': 0.49929709591228566, 'max_depth = 14': 0.4880052691868359, 'max_depth = 15': 0.49045945352120723,
             'max_depth = 16': 0.4857583123778141, 'max_depth = 17': 0.49572105947410955, 'max_depth = 18': 0.48490307135675986, 'max_depth = 19': 0.49253422230339855, 'max_depth = 20': 0.49039682567934256,
             'max_depth = 21': 0.49418589751727715, 'max_depth = 22': 0.5017346879449858, 'max_depth = 23': 0.49322878474508175, 'max_depth = 24': 0.489664254004623, 'max_depth = 25': 0.4886050197269219,
             'max_depth = 26': 0.4892302117594798, 'max_depth = 27': 0.4856969243548591, 'max_depth = 28': 0.4863106667600591, 'max_depth = 29': 0.4923098233672083}
CART_gm = {'max_depth = 1': 0.6968169910670404, 'max_depth = 2': 0.6968169910670404, 'max_depth = 3': 0.6949956104076733, 'max_depth = 4': 0.6877113057281615, 'max_depth = 5': 0.6687060524751933,
           'max_depth = 6': 0.6283728596611124, 'max_depth = 7': 0.5263539082875657, 'max_depth = 8': 0.5715647532994916, 'max_depth = 9': 0.5691238132309352, 'max_depth = 10': 0.5503452810034581,
           'max_depth = 11': 0.5449329486232329, 'max_depth = 12': 0.5347290901498083, 'max_depth = 13': 0.5221273778991693, 'max_depth = 14': 0.5119840998995786, 'max_depth = 15': 0.5137978323169541,
           'max_depth = 16': 0.5092172195124637, 'max_depth = 17': 0.5188771654610027, 'max_depth = 18': 0.5082280665772176, 'max_depth = 19': 0.5163010307785167, 'max_depth = 20': 0.5137470207996764,
           'max_depth = 21': 0.5172617793343247, 'max_depth = 22': 0.5241232524321615, 'max_depth = 23': 0.5163770738088027, 'max_depth = 24': 0.5131702726501928, 'max_depth = 25': 0.511872141933048,
           'max_depth = 26': 0.5126093773267318, 'max_depth = 27': 0.5093585554508511, 'max_depth = 28': 0.5102377990480607, 'max_depth = 29': 0.5164583914161505}

sens = list()
spec = list()
gm = list()
for i in range(len(list(CART_sens.keys()))):
    sens.append(CART_sens[list(CART_sens.keys())[i]])
    spec.append(CART_spec[list(CART_spec.keys())[i]])
    gm.append(CART_gm[list(CART_gm.keys())[i]])
x_axis = list(range(1, 30))

plt.plot(x_axis, sens, color = 'r', label = 'sensitivity')
plt.plot(x_axis, spec, color = 'b', label = 'specificity')
plt.plot(x_axis, gm, color = 'g', label = 'gmean')
plt.xlabel('maximum depth')
plt.title('CART')
plt.legend()
plt.show()