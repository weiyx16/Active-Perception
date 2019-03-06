# Deep Reinforcement Learning for robotic manuplation

Tensorflow implementation of DQN for our active exploration strategy. It includes two kinds of model, with one outputs 8*8*8 kinds of action and the other 18*18*8. And the code is motivated by [DQN-tensorflow](https://github.com/devsisters/DQN-tensorflow).  

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

### Combine with Affordance map

- Torch 7  
- For more details, ref to [Affordance map](http://arc.cs.princeton.edu)

## Usage

To train a model:

    $ python main.py --is_train=True

To evaluate the model:  

- In simulation environment  

    $ python main.py --is_train=False --is_sim=True

- In real ur5 manipulator and kinect V2 camera:  

    $ python main.py --is_train=False --is_sim=False

## Details

The DQN structure has 8 parallel U-Net-like modules to output subpixel-wised operation locations.  
We trained the model in a simulation environment named [V-Rep](http://coppeliarobotics.com/) and our simulation scene file is in './simulation/environment.ttt'.  
The model is test on ubuntu 16.04 only, with CUDA 8.0, CUDNN 6.0.  

## License

MIT License.
