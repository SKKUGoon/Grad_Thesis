import matplotlib.pyplot as plt

# Average of 20 results.
# Tree # 10
tree10_res_sens = {'max_depth = 1': 0.6212879780722261, 'max_depth = 2': 0.6553951681946081, 'max_depth = 3': 0.6790706556106092, 'max_depth = 4': 0.6964155182299768, 'max_depth = 5': 0.6949320575959002,
                   'max_depth = 6': 0.6924347837041129, 'max_depth = 7': 0.689408121005939, 'max_depth = 8': 0.6718812958857647, 'max_depth = 9': 0.6595705877554771, 'max_depth = 10': 0.6342585390118902,
                   'max_depth = 11': 0.6267781127289632, 'max_depth = 12': 0.6091860453035581, 'max_depth = 13': 0.5924851438133032, 'max_depth = 14': 0.594726352321993, 'max_depth = 15': 0.5837444328504313}
tree10_res_spec =  {'max_depth = 1': 0.6726626533969929, 'max_depth = 2': 0.6854476281888309, 'max_depth = 3': 0.6694667431592889, 'max_depth = 4': 0.6768172743999527, 'max_depth = 5': 0.657663223673059,
                    'max_depth = 6': 0.6498319139645801, 'max_depth = 7': 0.6189013520968587, 'max_depth = 8': 0.6057204797197134, 'max_depth = 9': 0.586573325765419, 'max_depth = 10': 0.5753921823902302,
                    'max_depth = 11': 0.5671883069432473, 'max_depth = 12': 0.5436947921958863, 'max_depth = 13': 0.5331848446638189, 'max_depth = 14': 0.5293574447926807, 'max_depth = 15': 0.5118289571300952}
tree10_gm ={'max_depth = 1': 0.6464213883512908, 'max_depth = 2': 0.6701539338515684, 'max_depth = 3': 0.6740360496308596, 'max_depth = 4': 0.6864326445386539, 'max_depth = 5': 0.67581094954664,
            'max_depth = 6': 0.6705502038631687, 'max_depth = 7': 0.6528624667206115, 'max_depth = 8': 0.6377467007710282, 'max_depth = 9': 0.6217329034519603, 'max_depth = 10': 0.6039237478867803,
            'max_depth = 11': 0.5961146172665173, 'max_depth = 12': 0.5754013994503999, 'max_depth = 13': 0.5619726038319406, 'max_depth = 14': 0.5610109045734869, 'max_depth = 15': 0.5465432564006252}


# Tree # 30
tree30_res_sens = {'max_depth = 1': 0.6308912013661138, 'max_depth = 2': 0.6677771204561376, 'max_depth = 3': 0.6829647180313476, 'max_depth = 4': 0.7007694817445943, 'max_depth = 5': 0.7096499429863209,
                   'max_depth = 6': 0.7138747667794156, 'max_depth = 7': 0.7161956018968486, 'max_depth = 8': 0.7098000482823495, 'max_depth = 9': 0.6998799298091953, 'max_depth = 10': 0.6827556824011127,
                   'max_depth = 11': 0.675334718663283, 'max_depth = 12': 0.6588380763779575, 'max_depth = 13': 0.6502615666047069, 'max_depth = 14': 0.6287602366226266, 'max_depth = 15': 0.6383383983722178}
tree30_res_spec = {'max_depth = 1': 0.7087437966160379, 'max_depth = 2': 0.7020179823302295, 'max_depth = 3': 0.6893647907038304, 'max_depth = 4': 0.6778868985262979, 'max_depth = 5': 0.6726043181220429,
                   'max_depth = 6': 0.6582592801930199, 'max_depth = 7': 0.6458067372393563, 'max_depth = 8': 0.6435878381857155, 'max_depth = 9': 0.630193502991769, 'max_depth = 10': 0.6181267117734104,
                   'max_depth = 11': 0.6003011362880725, 'max_depth = 12': 0.5910130916566485, 'max_depth = 13': 0.5753333168558903, 'max_depth = 14': 0.5564723067003348, 'max_depth = 15': 0.554387218206285}
tree30_gm = {'max_depth = 1': 0.6686638277950585, 'max_depth = 2': 0.6846134566697074, 'max_depth = 3': 0.6860986425829603, 'max_depth = 4': 0.6891690917259606, 'max_depth = 5': 0.6908219246290486,
             'max_depth = 6': 0.6853725229222519, 'max_depth = 7': 0.6799724846016472, 'max_depth = 8': 0.6757758585741667, 'max_depth = 9': 0.6639511826912238, 'max_depth = 10': 0.6494893058851154,
             'max_depth = 11': 0.6366127097321335, 'max_depth = 12': 0.6238156739554869, 'max_depth = 13': 0.6114738215233018, 'max_depth = 14': 0.5914540857351791, 'max_depth = 15': 0.5947182251297545}


# Tree # 100
tree100_res_sens = {'max_depth = 1': 0.6253846199712896, 'max_depth = 2': 0.6622100006446657, 'max_depth = 3': 0.6887761123441709, 'max_depth = 4': 0.7034636853498623, 'max_depth = 5': 0.7114118166294783,
                    'max_depth = 6': 0.7167721220230411, 'max_depth = 7': 0.718503265279512, 'max_depth = 8': 0.7241100689076866, 'max_depth = 9': 0.7254952285922973, 'max_depth = 10': 0.717149050603287,
                    'max_depth = 11': 0.7192111140786867, 'max_depth = 12': 0.7094981440219228, 'max_depth = 13': 0.6937121846041382, 'max_depth = 14': 0.68890522502357, 'max_depth = 15': 0.6709743786579894,
                    'max_depth = 16': 0.6647356055511114, 'max_depth = 17': 0.6572112399009133, 'max_depth = 18': 0.6452943078094412, 'max_depth = 19': 0.6551213069427599, 'max_depth = 20': 0.6546740169312983,
                    'max_depth = 21': 0.6551284935799659, 'max_depth = 22': 0.6513962932782302, 'max_depth = 23': 0.6527466834707402, 'max_depth = 24': 0.6522886235044152, 'max_depth = 25': 0.6566446331319786,
                    'max_depth = 26': 0.6471923132927551, 'max_depth = 27': 0.6495361424727024, 'max_depth = 28': 0.6510168684590878, 'max_depth = 29': 0.6464302761167764}

tree100_res_spec = {'max_depth = 1': 0.7067207767887238, 'max_depth = 2': 0.7049732167067059, 'max_depth = 3': 0.6955415276524521, 'max_depth = 4': 0.6838564936636257, 'max_depth = 5': 0.6784796311600476,
                    'max_depth = 6': 0.6722602542116511, 'max_depth = 7': 0.6667558121087657, 'max_depth = 8': 0.6610773334442064, 'max_depth = 9': 0.6541051345285428, 'max_depth = 10': 0.6403739416359966,
                    'max_depth = 11': 0.6320507021310029, 'max_depth = 12': 0.6129220718753949, 'max_depth = 13': 0.5965573178121517, 'max_depth = 14': 0.5853793593951816, 'max_depth = 15': 0.5830920846135538,
                    'max_depth = 16': 0.5718400880231682, 'max_depth = 17': 0.5603824062453922, 'max_depth = 18': 0.5517167211483421, 'max_depth = 19': 0.5549419848609658, 'max_depth = 20': 0.5524470116612856,
                    'max_depth = 21': 0.5581245449682843, 'max_depth = 22': 0.5416064056482093, 'max_depth = 23': 0.5541545648112968, 'max_depth = 24': 0.541262776005422, 'max_depth = 25': 0.5519695323851816,
                    'max_depth = 26': 0.5410274643612405, 'max_depth = 27': 0.5411382054971683, 'max_depth = 28': 0.5477559213229541, 'max_depth = 29': 0.5383343280323594}
tree100_gm = {'max_depth = 1': 0.6647954937519355, 'max_depth = 2': 0.6832401535085475, 'max_depth = 3': 0.692137296834093, 'max_depth = 4': 0.6935773028816642, 'max_depth = 5': 0.6947234996412177,
              'max_depth = 6': 0.6941177337563535, 'max_depth = 7': 0.6921145102564285, 'max_depth = 8': 0.6918175143167622, 'max_depth = 9': 0.6888347688894609, 'max_depth = 10': 0.6775613388151334,
              'max_depth = 11': 0.674088850721786, 'max_depth = 12': 0.6593337767687907, 'max_depth = 13': 0.6430881938688946, 'max_depth = 14': 0.6349358899343662, 'max_depth = 15': 0.6254112990226278,
              'max_depth = 16': 0.6164897742806876, 'max_depth = 17': 0.6067719448417943, 'max_depth = 18': 0.596528905614033, 'max_depth = 19': 0.6029075689878044, 'max_depth = 20': 0.6013146849291816,
              'max_depth = 21': 0.6045727725764445, 'max_depth = 22': 0.5939218798387366, 'max_depth = 23': 0.6013524522808291, 'max_depth = 24': 0.5941116658872405, 'max_depth = 25': 0.6019396677808787,
              'max_depth = 26': 0.5916670362694716, 'max_depth = 27': 0.5927741272169016, 'max_depth = 28': 0.5970461571801005, 'max_depth = 29': 0.5898242406006822}


# Sensitivity Robustness Check
temp10 = list()
temp30 = list()
temp100 = list()
for i in range(len(list(tree10_res_sens.keys()))):
    temp10.append(tree10_res_sens[list(tree10_res_sens.keys())[i]])
    temp30.append(tree30_res_sens[list(tree30_res_sens.keys())[i]])
    temp100.append(tree100_res_sens[list(tree100_res_sens.keys())[i]])
x_axis = list(range(1, 16))

plt.plot(x_axis, temp10, color = 'r', label = '10 trees')
plt.plot(x_axis, temp30, color = 'b', label = '30 trees')
plt.plot(x_axis, temp100, color = 'g', label = '100 trees')
plt.xlabel('maximum depth')
plt.ylabel('sensitivity')
plt.title('Sensitivity difference')
plt.show()

# Specificity Robustness Check
temp10 = list()
temp30 = list()
temp100 = list()
for i in range(len(list(tree10_res_spec.keys()))):
    temp10.append(tree10_res_spec[list(tree10_res_spec.keys())[i]])
    temp30.append(tree30_res_spec[list(tree30_res_spec.keys())[i]])
    temp100.append(tree100_res_spec[list(tree100_res_spec.keys())[i]])
x_axis = list(range(1, 16))

plt.plot(x_axis, temp10, color = 'r', label = '10 trees')
plt.plot(x_axis, temp30, color = 'b', label = '30 trees')
plt.plot(x_axis, temp100, color = 'g', label = '100 trees')
plt.xlabel('maximum depth')
plt.ylabel('specificity')
plt.title('Specificity difference')
plt.show()

# Gmean Robustness Check
temp10 = list()
temp30 = list()
temp100 = list()
for i in range(len(list(tree10_res_spec.keys()))):
    temp10.append(tree10_gm[list(tree10_gm.keys())[i]])
    temp30.append(tree30_gm[list(tree30_gm.keys())[i]])
    temp100.append(tree100_gm[list(tree100_gm.keys())[i]])
x_axis = list(range(1, 16))

plt.plot(x_axis, temp10, color = 'r', label = '10 trees')
plt.plot(x_axis, temp30, color = 'b', label = '30 trees')
plt.plot(x_axis, temp100, color = 'g', label = '100 trees')
plt.xlabel('maximum depth')
plt.ylabel('Gmean')
plt.title('Gmean difference')
plt.show()

x_ = list(range(1, 30))
sens100 = list()
spec100 = list()
gm100 = list()
for i in range(len(list(tree100_res_spec.keys()))):
    sens100.append(tree100_res_sens[list(tree100_res_sens.keys())[i]])
    spec100.append(tree100_res_spec[list(tree100_res_spec.keys())[i]])
    gm100.append(tree100_gm[list(tree100_gm.keys())[i]])
x_axis = list(range(1, 16))

plt.plot(x_, sens100, color = 'r', label = 'sensitivity')
plt.plot(x_, spec100, color = 'b', label = 'specificity')
plt.plot(x_, gm100, color = 'g', label = 'gmean')
plt.xlabel('maximum depth')
plt.title('Number of Trees : 100')
plt.legend()
plt.show()