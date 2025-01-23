import multiprocessing
import os
import pickle
from new_baby_feedforwardnetwork import FeedForwardNetwork
from convolutional import Convolutional
from reshape import Reshape
#import tensorflow as tf
import numpy as np
from improv_trading_engine import Sim
import math
from activations import Sigmoid
from network import predict
from special_LSTM import LSTM
import neat
import neat.checkpoint

#loading the datast
#(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    
def eval_genome(genome, config):
    sim1 = Sim()
    #init all the network 
    if len(genome.custom_connections) != 5:
        print("Error connections :",len(genome.custom_connections))
    depth = len(genome.kernels)
    size = genome.kernels.shape[2]
    conv = Convolutional((1, 28, 28), size, depth)
    conv.kernels = genome.kernels
    conv.biases = genome.biases
    sigmod = Sigmoid()
    total_output_shape  = conv.output_shape[0] * conv.output_shape[1] * conv.output_shape[2]
    reshape = Reshape(conv.output_shape, (total_output_shape, 1))
    new_dense = FeedForwardNetwork.create(genome,config,0)
    sigmod2 = Sigmoid()
    new_dense2 = FeedForwardNetwork.create(genome,config,1)
    sigmod3 = Sigmoid()
    init_kernel = np.copy(genome.kernels)
    network = [
    conv,
    sigmod,
    reshape,
    new_dense,
    sigmod2,    
    new_dense2,
    sigmod3
    ]

    #construct the network base on genome information

    # for item in genome.custom_nodes[1].items():
    #     ng = item[1]
    #     print(item[0]," : ",ng.bias)

    #compare
    if conv.kernels.shape != genome.kernels.shape or conv.biases.shape != genome.biases.shape:
        RuntimeError("Big problemo")


    # print("\n First 10 connections of custom NETWORK [0]")
    # counter = 0
    # for key in iter(genome.custom_connections[0]):
    #     if counter < 10:
    #         print(key)
    #     else:
    #         break
    #     counter += 1

    #print("\n Kernel output\n",genome.kernels)


    #setting LSTM 
    dense1 = FeedForwardNetwork.create(genome,config,2)
    dense2 = FeedForwardNetwork.create(genome,config,3)
    final_dense = FeedForwardNetwork.create(genome,config,4)
    lstm = LSTM(500)
    lstm.Uf = genome.Uf
    lstm.Ui = genome.Ui 
    lstm.Uo = genome.Uo 
    lstm.Ug = genome.Ug 

    lstm.Wf = genome.Wf 
    lstm.Wi = genome.Wi 
    lstm.Wo = genome.Wo
    lstm.Wg = genome.Wg

    lstm.bf = genome.bf 
    lstm.bi = genome.bi 
    lstm.bo = genome.bo
    lstm.bg = genome.bg

    total_Score = 0
    markdown = 0
    for i in range(500):
        # inputs = x_train[i]
        inputs = sim1.get_state()
        min = sim1.calculate_local(period=28)[0]
        max = sim1.calculate_local(period=28)[1]
        cnn_inputs_raw = np.zeros((1,28,28))
        for i in range(28):
            position = math.floor(((sim1.df[sim1.current_step-28+i][3] - min) / (max - min))*28)-1
            if position > 27:
                position = 27
            elif position < 0:
                position = 0
            #TODO : somethign probably wrong with this
            cnn_inputs_raw[0][i][position] = 1 
        cnn_inputs = cnn_inputs_raw
        #print(cnn_inputs)

        X_t_cut = np.array([inputs[0]])
        lstm.forward(X_t_cut)
        H = np.array(lstm.H)
        H = H.reshape((H.shape[0],H.shape[1]))
        outputdense1 = dense1.forward(H[1:,:],1)
        output = dense2.forward(outputdense1)
        lstm_decision = 0
        if output[0] > 0.7:
            #print("buy")
            lstm_decision = 1
        elif output[1] > 0.7:
            #print("sell")
            lstm_decision = 2
        elif output[2] > 0.7:
            #print("hold")
            lstm_decision  = 0


        output = predict(network,cnn_inputs)
        cnn_decision = 0
        if output[0] > 0.7:
            #print("buy")
            cnn_decision = 1
        elif output[1] > 0.7:
            #print("sell")
            cnn_decision = 2
        elif output[2] > 0.7:
            #print("hold")
            cnn_decision  = 0
        inputs.extend([cnn_decision,lstm_decision])
        try:
            final_decision = final_dense.forward(np.array(inputs))
        except Exception as e:
            print(f"error in final dense {e}")
            return 500
        final_step = 0
        if final_decision[0] > 0.7:
            #print("buy")
            final_step = 0
        if final_decision[1] > 0.7:
            #print("sell")
            final_step = 1
        if final_decision[2] > 0.7:
            #print("hold")
            final_step  = 2
        #print("okay does it work then?",final_decision)
        sim1.step(final_step,final_decision[3])
        true_output = np.array([[1 if j == sim1.calulate_optimal() else 0 for j in range(3)]]).T
        grad = true_output - output
        # internal backpropogation

        # TODO: have to figure out how to calculate the output for trading for back progation. output = [buy,sell,do nothing].
        # what would be ideal is to have a way to manually calulate the best possible outcome for a trading period to identify the perfect buy sell positions. to train the network base off that.
        # so i suppose the main task to do rightnow is to code an algorithm that identify the perfect positions. TREND REVERSAL IS THE KEYY.
        # CODE ALGO TO IDENTIFY TREND REVERSAL AND TRAIN AI OFF THAT    


        for layer in reversed(network):
            try:
                grad = layer.backward(grad, 0.2)
                if grad.any() == -1:
                    #stop back propogation ignore this genome practically useless
                    #markdown -= 0.2
                    break
            except Exception as e:
                #print(f"ERROR : {e}, {layer.weights.shape},{layer.weights},{len(layer.node_evals)}")
                print(f"error skipping... [{e}]")
                break
                # TODO: somehow somewhere there is an error node_evals == 0, WHY?? idk find it out pls

        if sim1.current_capital < 0 :
            return 0
    genome.kernels = conv.kernels
    genome.biases = conv.biases 
    if (init_kernel == genome.kernels).all():
        print("ERROR kernels not modified")
    print('MISSION SUCCESS')
    sim1.finalize()

    #fitness ruless
    if sim1.current_capital == 1000 and genome.prev_fitness == 1000:
        fitness = genome.prev_fitness * 0.9 
    else:
        fitness = sim1.current_capital
    genome.prev_fitness = fitness
    return fitness

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ctrnn')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    #pop = neat.Checkpointer.restore_checkpoint("V2(no limit)-neat-checkpoint-11")
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(50,filename_prefix='V2(no limit)-neat-checkpoint-'))
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count()-10, eval_genome)
    winner = pop.run(pe.evaluate,20)
    input()
    # Save the winner.
    with open('winner-ctrnn', 'wb') as f:
        pickle.dump(winner, f)
    # saving the config
    with open("winner-config", 'w') as config_file:
        with open(config_path,"r") as f:
            config_file.write(str(f.read()))
    print(winner)
    print(f"WINNER FOUND and saved")




if __name__ == '__main__':
    run()