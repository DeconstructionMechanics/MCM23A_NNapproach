import weight_network as wn
import numpy as np
import gc


TRAIN_TIMES = 7
TOTAL_TIMES = 0

plant_dataset = np.load('.\\solutions\\data\\plant_dataset.npy')
environment_dataset = np.load('.\\solutions\\data\\environment_dataset.npy')
plant_dataset = np.where(np.isnan(plant_dataset),0,plant_dataset)
environment_dataset = np.where(np.isnan(environment_dataset),0,environment_dataset)


our_network = wn.WeightNetwork([225,201],4,0.1)
our_network.load(".\\solutions\\src\\WEIGHT_exp3.npy")

#train data
for t in range(TOTAL_TIMES):
    for i in range(1,10):
        for j in range(6):
            print("total %d, the %d th square, year %d" %(t,i,j))
            temp_inp = np.concatenate((plant_dataset[i][j],environment_dataset[j]))
            our_network.load_input(temp_inp)
            del temp_inp
            gc.collect()
            our_network.load_expected_output(plant_dataset[i][j + 1])
            for k in range(TRAIN_TIMES):
                our_network.run_input()
                our_network.change_weight()
                print(our_network.return_cost())

    #our_network.save(".\\solutions\\src\\WEIGHT_exp3.npy")
    print('saved')


#test data
plant_input = plant_dataset[0][0]
for j in range(6):
    i = 0
    temp_inp = np.concatenate((plant_input,environment_dataset[j]))
    our_network.load_input(temp_inp)
    del temp_inp
    gc.collect()
    our_network.run_input()
    plant_input = our_network.out_output()
    our_network.load_expected_output(plant_dataset[i][j + 1])
    print("at year %d:"%j)
    print(our_network.return_cost())
    # print('outputed:')
    # print(our_network.out_output())
    





