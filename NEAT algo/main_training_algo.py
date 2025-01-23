import multiprocessing
import os
import pickle
import neat
import neat.graphs
import neat.checkpoint
import trading_engine

runs_per_net = 25
#time constant in ticks
time_const = 1

# Use the CTRNN network phenotype and the discrete actuator force function.
#fitness score not base by capital 

def eval_genome(genome, config):
    #print("evaluating...")
    #net = neat.nn.RecurrentNetwork.create(genome,config)
    net = neat.ctrnn.CTRNN.create(genome, config, time_const)
    config.genome
    fitnesses = []
    for runs in range(runs_per_net):

        #created the enviroment
        sim = trading_engine.sim()

        #make sure it is a reseted state of the neural network
        net.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 100.0
        
        volume_trigger = 0
        #simulation testing per geome
        for i in range(500):
            #get current stock market info
            inputs = sim.get_state()

            #get predictions
            #action = net.activate(inputs)
            action = net.advance(inputs, time_const, time_const)

            #get results
            if action[0] > 0.8:
                #buy
                sim.step(0,action[1])
                # print(f"hitting buy volume : {action[2]}")
            else:
                #sell
                sim.step(1,action[1])
                # print(f"hitting sell volume : {action[2]}")

            #print(f"status : {sim.current_capital}\nholdings : {sim.current_holdings}")
            # Break out function
            if sim.current_capital < 0:
                break

            # if action[2] < 0.01 or action[2] > 0.9:
            #     volume_trigger += 1

        #fitness calculation
        # if volume_trigger > 5:
        #     fitness -= (volume_trigger/500)*40
        # if 990 < sim.current_capital and sim.current_capital < 1010:
        #     fitness -= 20
        # elif sim.current_capital > 1010:
        #     #profit hence reward
        #     fitness += (sim.current_capital-1000)
        # else:
        #     fitness -= (1010 - sim.current_capital)/1010 * 35
        
        #code this to train to hold positions

        fitnesses.append(sim.current_capital+())
        # print("{0} fitness {1}".format(net, fitness))

    # The genome's fitness is its worst performance across all runs.
    #print("evaluation completed!")
    return max(fitnesses)

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ctrnn')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    #pop = neat.Population(config)
    pop = neat.Checkpointer.restore_checkpoint("V2(no limit)-neat-checkpoint-1988")
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(50,filename_prefix='V2(no limit)-neat-checkpoint-'))
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count()-8, eval_genome)
    winner = pop.run(pe.evaluate,10)
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