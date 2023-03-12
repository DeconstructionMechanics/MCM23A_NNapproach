import weight_network as wn
import numpy as np
import cook
import matplotlib.pyplot as plt

plant_dataset_raw = np.load('.\\solutions\\data\\plant_dataset_raw.npy')
plant_dataset = np.load('.\\solutions\\data\\plant_dataset.npy')
environment_dataset = np.load('.\\solutions\\data\\environment_dataset.npy')
avg_environment_raw_reorder = np.load('.\\solutions\\data\\avg_environment_raw_reorder.npy')
avg_plant_raw = np.load('.\\solutions\\data\\avg_plant_raw.npy')
avg_environment = np.load('.\\solutions\\data\\avg_environment.npy')
avg_plant = np.load('.\\solutions\\data\\avg_plant.npy')

our_network = wn.WeightNetwork([225,201],4,0.001)
our_network.load(".\\solutions\\src\\WEIGHT_exp3.npy")


'''avg_environment to ripe'''
# avg_environment = np.zeros(avg_environment_raw.shape)
# for i in range(12):
#     avg_environment[i] = cook.to_ripe(4,avg_environment_raw[i])

# for i in range(12,24):
#     avg_environment[i] = cook.to_ripe(3,avg_environment_raw[i])

# print(avg_environment)
# avg_environment_reorder = np.concatenate((avg_environment[20:],avg_environment[12:20],avg_environment[8:12],avg_environment[:8]))
# print(avg_environment_reorder)
# np.save('.\\solutions\\data\\avg_environment.npy',avg_environment_reorder)




'''
plant predict 2016 2017
'''
# plant_predict_1617 = np.zeros([10,2,201])

# for i in range(10):
#     temp_inp = np.concatenate((plant_dataset[i][6],environment_dataset[6]))
#     print(cook.to_raw_plant(plant_dataset[i][6]))
#     print(cook.to_raw_environment(environment_dataset[6]))

#     our_network.load_input(temp_inp)
#     our_network.run_input()
#     plant_predict_1617[i][0] = our_network.out_output()
#     print(cook.to_raw_plant(our_network.out_output()))

# for i in range(10):
#     temp_inp = np.concatenate((plant_predict_1617[i][0],environment_dataset[7]))
#     our_network.load_input(temp_inp)
#     our_network.run_input()
#     plant_predict_1617[i][1] = our_network.out_output()

# np.save('.\\solutions\\data\\plant_predict_1617.npy',plant_predict_1617)


# plant_predict_1617 = np.load('.\\solutions\\data\\plant_predict_1617.npy')
# for i in range(10):
#     for j in range(2009,2018):
#         print('the %d th square, year %d'%(i,j))
#         if j < 2016:
#             print(cook.to_raw_plant(plant_dataset[i][j-2009][2::3]))
#         else:
#             print(cook.to_raw_plant(plant_predict_1617[i][j-2016][2::3]))
        




'''
average output water 200, 190, ... 100, 90, 80 ... 10 %
'''
# avg_output = []
# avg_plant = np.load('.\\solutions\\data\\avg_plant.npy')



# for i in range(20):
#     rate = np.ones(24,dtype=float)
#     for j in range(12,24):
#         rate[j] *= (20 - i) * 0.1

#     #print(rate)

#     our_network.load_input(np.concatenate((avg_plant,cook.to_ripe_environment(avg_environment_raw_reorder * rate))))
#     our_network.run_input()
#     avg_output.append(our_network.out_output())

# avg_output = np.asarray(avg_output)
# print(avg_output.shape)


# np.save('.\\solutions\\data\\result\\avg_output_descenting_water.npy',avg_output)

# avg_output = np.load('.\\solutions\\data\\result\\avg_output_descenting_water.npy')
# x = np.linspace(2,0.1,20)
# y = np.zeros(20)
# for i in range(20):
#     y[i] = sum(cook.to_raw_plant(avg_output[i])[2::3])

# p = np.poly1d(np.polyfit(x,y,3))
# t = np.linspace(2,0.1,100)

# plt.plot(t,p(t))
# plt.scatter(x,y,s=10,c='g')
# print('y = %d x^3 + %d x^2 + %d x + %d'%(p[0],p[1],p[2],p[3]))
# plt.xlabel('scale of precipitation')
# plt.ylabel('sum of plant\'s weight (g/m^2)')
# plt.show()

# print("---")
# for i in range(len(x)):
#     print('%f,%f'%(x[i],y[i]))


'''
pre\plant num:1,2...33 groups
100
90        data plants
...
10
'''
index = np.argsort(avg_plant_raw[1::3])
group = set()
for i in range(33):
    group.add((index[i*2 + 1],index[i*2 + 2]))

group = list(group)
print(group)

sum_1 = sum(avg_plant_raw[1::3])
sum_2 = sum(avg_plant_raw[2::3])

plant_asc_specesnum = []
for i in range(33):
    plant_in = np.zeros(201)
    for j in range(i + 1):
        for k in range(3):
            plant_in[group[j][0]*3 + k] = avg_plant_raw[group[j][0]*3 + k]
            plant_in[group[j][1]*3 + k] = avg_plant_raw[group[j][1]*3 + k]
        
    plant_asc_specesnum.append(plant_in)

plant_asc_specesnum = np.asarray(plant_asc_specesnum)


for i in range(33):
    sum_s_1 = sum(plant_asc_specesnum[i][1::3])
    sum_s_2 = sum(plant_asc_specesnum[i][2::3])
    if(sum_s_1 == 0 or sum_s_2 == 0):
        print(i)
        print(plant_asc_specesnum[i])
        continue

    plant_asc_specesnum[i][1::3] *= sum_1 / sum_s_1
    plant_asc_specesnum[i][2::3] *= sum_2 / sum_s_2


print(plant_asc_specesnum.shape)
np.save('.\\solutions\\data\\plant_asc_speciesnum_raw.npy',plant_asc_specesnum) 

for i in range(len(plant_asc_specesnum)):
    for j in range(len(plant_asc_specesnum[i])):
        plant_asc_specesnum[i][j] = cook.to_ripe(j%3,plant_asc_specesnum[i][j])

np.save('.\\solutions\\data\\plant_asc_speciesnum.npy',plant_asc_specesnum)



'''predict'''
plant_asc_specesnum = np.load('.\\solutions\\data\\plant_asc_speciesnum.npy')
speces_number_water_predict = []
for i in range(33):
    different_water = []
    for j in range(10):
        our_network.load_input(np.concatenate((plant_asc_specesnum[i],avg_environment[:12],(avg_environment[12:]*(10 - j)*0.1))))
        our_network.run_input()
        different_water.append(our_network.out_output())
    
    speces_number_water_predict.append(different_water)

speces_number_water_predict = np.asarray(speces_number_water_predict)
np.save('.\\solutions\\data\\result\\speces_number_water_predict.npy',speces_number_water_predict)


'''
avg plant
avg env ((1,0.9,0.8,...)water/irregular water)
many years (10)
water:     1  0.9  0.8  ...
  year 1:
  year 2:  plant data
  ...
'''

# plant_des_water_longterm_prediction = []

# for i in np.linspace(1,0.1,10):
#     process = []
#     environment_input = avg_environment
#     for j in range(12,24):
#         environment_input[j] *= i

#     plant_input = avg_plant
#     for y in range(10):
#         our_network.load_input(np.concatenate((plant_input,environment_input)))
#         our_network.run_input()
#         output = our_network.out_output()
#         process.append(our_network.out_output())
#         plant_input = output

#     plant_des_water_longterm_prediction.append(process)

# np.save('.\\solutions\\data\\result\\plant_desc_water_longterm_prediction.npy',plant_des_water_longterm_prediction)


'''random weather'''
# plant_various_cycle_longterm_prediction = []
# environment_variate_cycle = []

# plant_input = avg_plant
# temp_var = np.random.uniform(0.8,1.5,10)
# water_var = np.random.uniform(0.5,1.5,10)

# env_data = np.load('.\\solutions\\data\\result\\environment_variate_cycle.npy')
# for y in range(10):
#     # environment_input = avg_environment.copy()
#     # environment_input[:12] *= temp_var[y]
#     # environment_input[12:] *= water_var[y]
#     # environment_variate_cycle.append(environment_input)


#     our_network.load_input(np.concatenate((plant_input,env_data[y])))
#     our_network.run_input()
#     output = our_network.out_output()
#     plant_various_cycle_longterm_prediction.append(output.copy())
#     plant_input = output
    

# environment_variate_cycle = np.asarray(environment_variate_cycle)
# plant_various_cycle_longterm_prediction = np.asarray(plant_various_cycle_longterm_prediction)

# np.save('.\\solutions\\data\\result\\plant_various_cycle_longterm_prediction.npy',plant_various_cycle_longterm_prediction)
#np.save('.\\solutions\\data\\result\\environment_variate_cycle.npy',environment_variate_cycle)



'''
avg
67 species......
  10years
  ......    plant_data
'''

# single_species_longterm = np.zeros((67,10,201))


# for i in range(67):
#     plant_dt = np.zeros(201)
#     plant_dt[i*3] = avg_plant[i*3]
#     plant_dt[i*3 + 1] = sum(avg_plant[1::3])
#     plant_dt[i*3 + 2] = sum(avg_plant[2::3])

#     for y in range(10):
#         temp_inp = np.concatenate((cook.to_ripe_plant(plant_dt),avg_environment))
#         our_network.load_input(temp_inp)       
#         our_network.run_input()
#         plant_dt = our_network.out_output()
#         single_species_longterm[i][y] = plant_dt.copy()

# np.save('.\\solutions\\data\\result\\single_species_longterm.npy',single_species_longterm)
    




