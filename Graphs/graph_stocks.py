# Graph
for i in range(len(list(Asian_dataset.keys()))): # Asian Graph
    Asian_dataset[list(Asian_dataset.keys())[i]].plot(label = list(Asian_dataset.keys())[i])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

for i in range(len(list(European_dataset.keys()))): # European Graph
    European_dataset[list(European_dataset.keys())[i]].plot(label = list(European_dataset.keys())[i])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()