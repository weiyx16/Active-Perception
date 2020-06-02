# Deep Reinforcement Learning for Robotic Pushing and Picking in Cluttered Environment

Tensorflow implementation of DQN for our active exploration strategy. It includes two kinds of model, with one outputs 8\*8\*8 kinds of action and the other 18\*18\*8, the only difference between them is the resolution of the output action and the last **8** stands for 8 directions. And the code is motivated by [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow).  

## Environments

### For our model

- Python 3.6
- [tqdm](https://github.com/tqdm/tqdm)
- [SciPy](http://www.scipy.org/install.html)
- [OpenCV3](http://opencv.org/)
- [Numpy](http://www.numpy.org/)
- [Pillow](https://python-pillow.org/)
- [TensorFlow 1.4.1](https://github.com/tensorflow/)  
- [urx](https://github.com/SintefManufacturing/python-urx)  
- [pyserial](https://github.com/pyserial/pyserial)  
- [pypcl](https://github.com/cmpute/pypcl)  
- [h5py](https://github.com/h5py/h5py)

The model is test on ubuntu 16.04 only, with CUDA 8.0, CUDNN 6.0 and this [file](../environment.yaml) is the conda environment we used.

### Combine with Affordance map

- Torch 7 & Lua support  
- For more details, ref to [Affordance map](http://arc.cs.princeton.edu)

## Usage

### Prepare

We trained the model in a simulation environment named [V-Rep](http://coppeliarobotics.com/). The first thing is to prepare the V-Rep. Related Code is in *simulation* directory and our simulation scene file is in [Simulation.ttt](./simulation/Simulation.ttt).

The *blocks* directory in *simulation* directory stores the simulation objects we used (**Notice Change it can't Change the Objects, You have to reload objects in the Simulation.ttt**) and the *bg* directory stores the background depth and RGB image which you can obtain by uncomment the *bg_init* function L116 of [environment.py](simulation/environment.py). The *scene* and *test_scenes* directory stores the training and testing clusterd scene.  

### Training and Evaluation

Firstly, you have to open the V-Rep software and open the simulation file in [Simulation.ttt](./simulation/Simulation.ttt) and in the command console, please type in the following code for building connection between python script and V-Rep.

```Lua
    simRemoteApi.start(19999)
```
Laterly, you can run the training code. Usually you can directly see the moving robotic arms in the simulation. Notice, to obtain the affordance map, we will call a torch process in the python script and save the map in your local machine.  

```sh
    $ python main.py --is_train=True
```

To evaluate the model:  

- In simulation environment  

```sh
    $ python main.py --is_train=False --is_sim=True  
```

### Real Experiment

We also add in the code for real experiment with a real ur5 manipulator and kinect V2 camera in *experiment* directory. Notice the two files: [environment_kinectfree.py](./experiment/environment_kinectfree.py) seperating the process of read images from kinect camera, so you should read the image from the buffer in other places. The file [environment.py](./experiment/environment.py)
combine these two things, but actually not recommended....

Besides, you have to change all the related matrix in the file [environment_kinectfree.py](./experiment/environment_kinectfree.py), including the intri-matrix of camera(L80), relative transpoce between camera and arms(L93) and relative rotation(L97). We use urx lib to make connection to ur5 arms and a USB with the manipulator we designed. You have to replace the function of manipulator control in L261....

In sum, lots of things need to be checked if you want to do experiment in real world, this code is only for reference. Notice, **Keep Safe!**  

The running code can be:
```sh
    $ python main.py --is_train=False --is_sim=False  
```

## Trouble Shooting

We recommend you to **Read src code** and **Search in the issues** first, or you can open a new issue, we will try best to help.

## Citation

If you feel it useful, please cite:
```bibtex
@INPROCEEDINGS{Wei2019ActiveExplore,
  author={Y. {Deng} and X. {Guo} and Y. {Wei} and K. {Lu} and B. {Fang} and D. {Guo} and H. {Liu} and F. {Sun}},
  booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Deep Reinforcement Learning for Robotic Pushing and Picking in Cluttered Environment}, 
  year={2019},
  volume={},
  number={},
  pages={619-626},}
```

## License

MIT License.

