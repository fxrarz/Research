"""
Generating A Gait For A Biped Robot Using Actor Critic Method In Reinforcement Learning

Requirement: CPU/GPU

Written on: 23 March 2022

Tested on: Linux(debian)

Author: AS Faraz Ahmed

"""
import gym
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution() # fast performance

def run():
	#Initalize gym environment
    env = gym.make("BipedalWalker-v3")
    
    #Load trained model
    Actor = tf.keras.models.load_model('./models/model.h5', compile=False)
    for i in range(101):
    	#reset environment
        state = env.reset()
        
        #reshape state/observation
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        done = False
        score = 0
        while not done:
            env.render()    #render output
            #predict action for state/observation
            action = Actor.predict(state)[0]
            state, reward, done, _ = env.step(action)    #do the prediction/action
            
            #reshape state/observation
            state = np.reshape(state, [1, env.observation_space.shape[0]])
            
            #sumup the reward
            score += reward
            if done:           #reset of environment         
            	print("episode: {}/{}, score: {}".format(i, 100, score))
            	break
    env.close()

run()
